# Metric k-Median Approximation

This project implements:

- a naive greedy baseline for the metric `k`-median problem,
- a local-search approximation based on one-swap improvements,
- an experiment runner that records runtime, costs, and convergence metrics,
- plot generation for the empirical evaluation,
- a LaTeX report template tied to the generated figures.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m kmedian_project.experiments
```

The command writes:

- `results/experiment_results.csv`
- `results/summary.json`
- `results/summary_table.csv`
- `figures/cost_comparison.png`
- `figures/runtime_comparison.png`
- `figures/swap_iterations.png`
- `figures/improvement_pct.png`

## Compile the report

```bash
pdflatex report.tex
```
