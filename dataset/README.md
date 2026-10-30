# Dataset

This directory contains the scripts and data used to construct and prepare the TPTP problem datasets for training and evaluating machine-learning-guided clause selection in iProver.

## Directory Structure

- **`problems/`** — Contains a small selection of example TPTP problems in `examples/`, alongside directories for the training, development and test datasets.
- **`results/`** — Contains CSV files recording dataset construction results, problem selections and dataset splits.
- **`scripts/`** — Contains scripts for dataset construction, analysis and preparation.

## Scripts

| Script | Description |
|---|---|
| `build_dataset.py` | Filters TPTP problems by domain, rating and status, evaluates them using standard iProver, and records the results in a CSV file. |
| `count_valid_problems.py` | Counts TPTP problems satisfying the specified rating and status restrictions, reporting counts and average ratings by domain. |
| `csv_stats.py` | Calculates summary statistics from dataset results CSVs, including solve rates, timeouts and runtimes. |
| `merge_datasets.py` | Combines multiple dataset CSV files into a single dataset. |
| `split_dataset.py` | Splits a dataset into training, development and test sets, stratified by domain and solved status. |
| `stratified_sample.py` | Creates a stratified subset of a dataset, preserving the approximate distribution of selected characteristics. |
| `copy_split_problems.py` | Copies TPTP problems listed in split CSVs into their corresponding training, development and test directories. |
| `copy_problem_subset.py` | Copies TPTP problems listed in a specified CSV into a user-specified directory under `problems/`. |