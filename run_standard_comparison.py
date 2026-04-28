from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ASSET_DIR = Path("assets")
ASSET_DIR.mkdir(exist_ok=True)

BASE_SEED = 42
NUM_RUNS = 10
DIM_LIST = [10, 30, 50]
NUM_AGENTS = 30
MAX_ITERS = 1000
BOUNDS = (-5.0, 5.0)

GWO_RANDOM_STEP = 1.0
GWO_A0 = 2.0
GWO_NOISE_SCALE = 0.03

PSO_W_START = 0.9
PSO_W_END = 0.4
PSO_C1 = 2.0
PSO_C2 = 2.0
PSO_VMAX = 1.0

RAS_TOL_PER_DIM = 1e-2
ST_OPTIMUM_PER_DIM = -39.16616570377142
ST_TOL_PER_DIM = 5e-1
ACKLEY_TOL = 1e-2
GRIEWANK_TOL = 1e-2


def rastrigin(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return 10.0 * x.shape[1] + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=1)


def styblinski_tang(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return 0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=1)


def ackley(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    n = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x), axis=1)
    return -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20.0 + np.e


def griewank(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    i = np.arange(1, x.shape[1] + 1)
    return np.sum(x**2, axis=1) / 4000.0 - np.prod(np.cos(x / np.sqrt(i)), axis=1) + 1.0


FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Rastrigin": rastrigin,
    "Styblinski-Tang": styblinski_tang,
    "Ackley": ackley,
    "Griewank": griewank,
}


def is_success(function_name: str, dim: int, best_value: float) -> bool:
    if function_name == "Rastrigin":
        return best_value <= RAS_TOL_PER_DIM * dim
    if function_name == "Styblinski-Tang":
        target = ST_OPTIMUM_PER_DIM * dim
        return abs(best_value - target) <= ST_TOL_PER_DIM * dim
    if function_name == "Ackley":
        return best_value <= ACKLEY_TOL
    if function_name == "Griewank":
        return best_value <= GRIEWANK_TOL
    return False


def levy_flight_step(rng: np.random.Generator, shape: tuple[int, int], beta: float = 1.5) -> np.ndarray:
    sigma_u = (
        (math.gamma(1 + beta) * math.sin(math.pi * beta / 2))
        / (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
    ) ** (1 / beta)
    u = rng.normal(0, sigma_u, shape)
    v = rng.normal(0, 1, shape)
    return u / (np.abs(v) ** (1 / beta) + 1e-12)


def gwo_candidate_update(
    rng: np.random.Generator,
    positions: np.ndarray,
    leaders: list[np.ndarray],
    a: float,
    progress: float,
) -> np.ndarray:
    new_parts = []
    for leader in leaders:
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        a_vec = 2.0 * a * r1 - a
        c_vec = 2.0 * r2
        distance = np.abs(c_vec * leader - positions)
        new_parts.append(leader - a_vec * distance)

    new_positions = np.mean(new_parts, axis=0)
    perturb = rng.uniform(-GWO_RANDOM_STEP, GWO_RANDOM_STEP, positions.shape) * GWO_NOISE_SCALE * (1.0 - progress)
    return np.clip(new_positions + perturb, BOUNDS[0], BOUNDS[1])


def run_original_gwo(objective: Callable[[np.ndarray], np.ndarray], dim: int, seed: int) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(BOUNDS[0], BOUNDS[1], (NUM_AGENTS, dim))
    best_score = float("inf")
    history = []

    for t in range(MAX_ITERS):
        fitness = objective(positions)
        sorted_idx = np.argsort(fitness)
        positions = positions[sorted_idx]
        fitness = fitness[sorted_idx]

        if fitness[0] < best_score:
            best_score = float(fitness[0])

        progress = t / max(1, MAX_ITERS - 1)
        a = GWO_A0 * (1.0 - progress)
        leaders = [positions[0].copy(), positions[1].copy(), positions[2].copy()]
        positions = gwo_candidate_update(rng, positions, leaders, a, progress)
        history.append(best_score)

    return best_score, np.asarray(history)


def roulette_index(rng: np.random.Generator, leader_fitness: np.ndarray) -> int:
    max_fit = np.max(leader_fitness)
    weights = (max_fit - leader_fitness) + 1e-12
    probs = weights / np.sum(weights)
    return int(rng.choice(len(leader_fitness), p=probs))


def run_hpro(objective: Callable[[np.ndarray], np.ndarray], dim: int, seed: int) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(BOUNDS[0], BOUNDS[1], (NUM_AGENTS, dim))
    particle_ids = np.arange(NUM_AGENTS)
    omega_streak = np.zeros(NUM_AGENTS, dtype=int)
    leader_group_size = max(1, int(np.ceil(0.1 * NUM_AGENTS)))
    omega_drop_threshold = max(1, math.ceil(0.05 * MAX_ITERS))
    drop_activation_iter = max(0, math.ceil(0.3 * MAX_ITERS))
    best_score = float("inf")
    history = []

    for t in range(MAX_ITERS):
        fitness = objective(positions)
        sorted_idx = np.argsort(fitness)
        positions = positions[sorted_idx]
        fitness = fitness[sorted_idx]
        particle_ids = particle_ids[sorted_idx]

        if fitness[0] < best_score:
            best_score = float(fitness[0])

        alpha = positions[0].copy()
        beta_group = positions[leader_group_size : min(2 * leader_group_size, NUM_AGENTS)]
        if len(beta_group) == 0:
            beta_group = positions[:1]
        delta_group = positions[2 * leader_group_size : min(3 * leader_group_size, NUM_AGENTS)]
        if len(delta_group) == 0:
            delta_group = beta_group

        beta = beta_group[int(rng.integers(len(beta_group)))].copy()
        delta = delta_group[int(rng.integers(len(delta_group)))].copy()
        leaders = [alpha, beta, delta]
        leader_fitness = fitness[: len(leaders)]

        drop_active = t >= drop_activation_iter
        if drop_active:
            omega_mask = np.arange(NUM_AGENTS) >= 3
            omega_streak[particle_ids[omega_mask]] += 1
            omega_streak[particle_ids[~omega_mask]] = 0
        else:
            omega_streak[particle_ids] = 0

        drop_mask = drop_active & (omega_streak[particle_ids] >= omega_drop_threshold)

        progress = t / max(1, MAX_ITERS - 1)
        a = GWO_A0 * (1.0 - progress)
        updated_positions = gwo_candidate_update(rng, positions, leaders, a, progress)

        drop_indices = np.where(drop_mask)[0]
        if len(drop_indices) > 0:
            for idx in drop_indices:
                base_idx = roulette_index(rng, leader_fitness)
                base_pos = leaders[base_idx]
                new_pos = base_pos + levy_flight_step(rng, (1, dim))[0] * GWO_RANDOM_STEP
                updated_positions[idx] = np.clip(new_pos, BOUNDS[0], BOUNDS[1])
                omega_streak[particle_ids[idx]] = 0

        positions = updated_positions
        history.append(best_score)

    return best_score, np.asarray(history)


def run_pso(objective: Callable[[np.ndarray], np.ndarray], dim: int, seed: int) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(BOUNDS[0], BOUNDS[1], (NUM_AGENTS, dim))
    velocities = rng.uniform(-PSO_VMAX, PSO_VMAX, (NUM_AGENTS, dim))
    pbest_pos = positions.copy()
    pbest_scores = objective(positions)
    best_idx = int(np.argmin(pbest_scores))
    gbest_pos = pbest_pos[best_idx].copy()
    gbest_score = float(pbest_scores[best_idx])
    history = []

    for t in range(MAX_ITERS):
        progress = t / max(1, MAX_ITERS - 1)
        inertia = PSO_W_START - (PSO_W_START - PSO_W_END) * progress
        r1 = rng.random((NUM_AGENTS, dim))
        r2 = rng.random((NUM_AGENTS, dim))
        velocities = (
            inertia * velocities
            + PSO_C1 * r1 * (pbest_pos - positions)
            + PSO_C2 * r2 * (gbest_pos - positions)
        )
        velocities = np.clip(velocities, -PSO_VMAX, PSO_VMAX)
        positions = np.clip(positions + velocities, BOUNDS[0], BOUNDS[1])

        scores = objective(positions)
        improved = scores < pbest_scores
        pbest_pos[improved] = positions[improved]
        pbest_scores[improved] = scores[improved]

        best_idx = int(np.argmin(pbest_scores))
        if pbest_scores[best_idx] < gbest_score:
            gbest_score = float(pbest_scores[best_idx])
            gbest_pos = pbest_pos[best_idx].copy()

        history.append(gbest_score)

    return gbest_score, np.asarray(history)


ALGORITHMS: dict[str, Callable[[Callable[[np.ndarray], np.ndarray], int, int], tuple[float, np.ndarray]]] = {
    "Original GWO": run_original_gwo,
    "PSO": run_pso,
    "HPRO": run_hpro,
}


def summarize(run_best: list[float], run_hist: list[np.ndarray], run_time: list[float], function_name: str, dim: int) -> dict:
    best_array = np.asarray(run_best, dtype=float)
    success = np.asarray([is_success(function_name, dim, value) for value in best_array], dtype=bool)
    return {
        "best": float(np.min(best_array)),
        "mean": float(np.mean(best_array)),
        "std": float(np.std(best_array)),
        "worst": float(np.max(best_array)),
        "success_rate": float(np.mean(success) * 100.0),
        "mean_time_sec": float(np.mean(run_time)),
        "history": np.mean(np.vstack(run_hist), axis=0),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    results: dict[str, dict[str, dict[int, dict]]] = {alg: {fname: {} for fname in FUNCTIONS} for alg in ALGORITHMS}

    for alg_name, runner in ALGORITHMS.items():
        print(f"=== Running {alg_name} ===")
        for fname, objective in FUNCTIONS.items():
            for dim in DIM_LIST:
                run_best = []
                run_hist = []
                run_time = []
                for run in range(NUM_RUNS):
                    seed = BASE_SEED + run
                    start = time.perf_counter()
                    best, hist = runner(objective, dim, seed)
                    elapsed = time.perf_counter() - start
                    run_best.append(best)
                    run_hist.append(hist)
                    run_time.append(elapsed)
                summary = summarize(run_best, run_hist, run_time, fname, dim)
                results[alg_name][fname][dim] = summary
                print(
                    f"{fname:16s} dim={dim:2d} "
                    f"best={summary['best']:.6f} mean={summary['mean']:.6f} "
                    f"success={summary['success_rate']:.1f}%"
                )

    rows = []
    for alg_name in ALGORITHMS:
        for fname in FUNCTIONS:
            for dim in DIM_LIST:
                s = results[alg_name][fname][dim]
                rows.append(
                    {
                        "algorithm": alg_name,
                        "function": fname,
                        "dim": dim,
                        "best": s["best"],
                        "mean": s["mean"],
                        "std": s["std"],
                        "worst": s["worst"],
                        "success_rate_percent": s["success_rate"],
                        "mean_time_sec": s["mean_time_sec"],
                    }
                )

    save_csv(ASSET_DIR / "standard_algorithm_comparison_results.csv", rows)
    save_csv(ASSET_DIR / "standard_algorithm_comparison_dim30.csv", [row for row in rows if row["dim"] == 30])

    for fname in FUNCTIONS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for alg_name in ALGORITHMS:
            ax.plot(results[alg_name][fname][30]["history"], label=alg_name)
        ax.set_title(f"{fname} Convergence Comparison (30D)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best fitness")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        file_name = fname.lower().replace("-", "_") + "_standard_comparison_dim30.png"
        fig.savefig(ASSET_DIR / file_name, dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, fname in zip(axes.flatten(), FUNCTIONS):
        for alg_name in ALGORITHMS:
            ax.plot(results[alg_name][fname][30]["history"], label=alg_name)
        ax.set_title(f"{fname} (30D)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best fitness")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "standard_algorithm_convergence_comparison_dim30.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    settings = ASSET_DIR / "standard_algorithm_comparison_settings.txt"
    settings.write_text(
        "Compared algorithms: Original GWO, PSO, HPRO\n"
        "Benchmark functions: Rastrigin, Styblinski-Tang, Ackley, Griewank\n"
        f"Population size: {NUM_AGENTS}\n"
        f"Maximum iterations: {MAX_ITERS}\n"
        f"Independent runs: {NUM_RUNS}\n"
        f"Dimensions: {DIM_LIST}\n"
        f"Bounds: {BOUNDS}\n"
        f"Base seed: {BASE_SEED}; run seed: BASE_SEED + run\n"
        f"PSO parameters: w {PSO_W_START}->{PSO_W_END}, c1={PSO_C1}, c2={PSO_C2}, vmax={PSO_VMAX}\n"
        "Success criteria: Rastrigin <= 1e-2 * dim; "
        "Styblinski-Tang within 5e-1 * dim of -39.16616570377142 * dim; "
        "Ackley <= 1e-2; Griewank <= 1e-2\n",
        encoding="utf-8",
    )
    print(f"Saved comparison assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()
