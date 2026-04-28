# Soft Computing Assignment 2

This project implements and evaluates the **Hierarchical Pack Regeneration Optimizer (HPRO)**, a swarm-based optimizer inspired by hierarchical pack leadership and population regeneration.

HPRO is implemented from a GWO-style search framework, but changes the interaction logic by using group-based leadership, omega memory, birth-death regeneration, and Levy flight renewal.

## Project Structure

| File / Folder | Purpose |
|---|---|
| `GWO_hybrid0.3.ipynb` | Final HPRO notebook and final-method result generation. |
| `GWO_original.ipynb` | Original GWO baseline notebook. |
| `GWO_group.ipynb` | Ablation notebook for group-based leadership. |
| `GWO_borndeath.ipynb` | Ablation notebook for omega birth-death regeneration. |
| `GWO_hybrid.ipynb` | Earlier hybrid/group draft kept for comparison. |
| `run_standard_comparison.py` | Main reproducible script for HPRO vs Original GWO vs PSO. |
| `assets/` | Generated result tables, convergence graphs, and report-ready table files. |
| `Assignment-2 (1).docx` | Assignment requirements. |
| `Soft_computing A2.pdf` | Original presentation/report draft. |

## Algorithms Compared

The main benchmarking comparison uses:

- Original Grey Wolf Optimizer (GWO)
- Particle Swarm Optimization (PSO)
- Hierarchical Pack Regeneration Optimizer (HPRO)

This satisfies the assignment requirement of comparing the proposed method against at least two standard algorithms.

## Benchmark Functions

The experiments use four benchmark functions:

- Rastrigin
- Styblinski-Tang
- Ackley
- Griewank

## Parameter Settings

| Parameter | Value |
|---|---|
| Population size | 30 |
| Maximum iterations | 1000 |
| Independent runs | 10 |
| Dimensions | 10, 30, 50 |
| Search bounds | `[-5, 5]` |
| Base random seed | 42 |
| Run seed rule | `BASE_SEED + run` |
| HPRO leader group size | `ceil(0.1 * N)` |
| HPRO birth-death activation | `0.3 * T` |
| HPRO omega threshold | `ceil(0.05 * T)` |
| PSO inertia weight | `0.9 -> 0.4` |
| PSO acceleration coefficients | `c1 = 2.0`, `c2 = 2.0` |
| PSO maximum velocity | `1.0` |

## How to Reproduce the Standard Comparison

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the standard comparison:

```bash
python run_standard_comparison.py
```

The script regenerates the comparison between HPRO, original GWO, and PSO on all four benchmark functions.

## Outputs

Generated outputs are stored in `assets/`.

Important report assets:

| Output | Description |
|---|---|
| `assets/standard_algorithm_comparison_results.csv` | Full HPRO vs GWO vs PSO results for 10D, 30D, and 50D. |
| `assets/standard_algorithm_comparison_dim30.csv` | 30D results used with convergence graphs. |
| `assets/standard_algorithm_convergence_comparison_dim30.png` | Combined 2x2 convergence graph for all functions. |
| `assets/rastrigin_standard_comparison_dim30.png` | Rastrigin convergence graph. |
| `assets/styblinski_tang_standard_comparison_dim30.png` | Styblinski-Tang convergence graph. |
| `assets/ackley_standard_comparison_dim30.png` | Ackley convergence graph. |
| `assets/griewank_standard_comparison_dim30.png` | Griewank convergence graph. |
| `assets/benchmark_results_tables_embedded.tex` | Report-ready LaTeX tables generated from CSV files. |

## Notes

- The final proposed algorithm name used in the report is **HPRO**.
- `GWO_hybrid0.3.ipynb` is the final HPRO notebook.
- `run_standard_comparison.py` is the recommended main script for reproducible benchmarking.
