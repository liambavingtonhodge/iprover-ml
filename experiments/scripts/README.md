# Experiments

This directory contains the scripts and results used to evaluate the performance of standard iProver and machine-learning-guided clause selection approaches.

## Directory Structure

- **`scripts/`** — Contains scripts for running experiments and collecting performance data.
- **`results/`** — Contains CSV files recording experimental results and JSON files containing summary statistics.

## Scripts

| Script | Description |
|---|---|
| `run_experiment.py` | Runs standard iProver or externally guided iProver on a specified set of TPTP problems, recording solving outcomes, runtimes and proof-search statistics. |

## Experimental Evaluation

Experiments can be conducted using either standard iProver or an external clause-selection agent.

The script supports configurable time limits, iProver parameters and clause-selection queue settings.

For each problem, the results include solving status, runtime, number of generated and processed clauses, instantiations and search iterations.

Results are saved as CSV files, with accompanying JSON files summarising solve rates, timeouts and overall runtime.

For further details on experimental methodology and performance comparisons, see the accompanying dissertation.