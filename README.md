# Machine Learning for Clause Selection in First-Order Automated Theorem Proving

**University of Manchester — Final-Year Computer Science Project (2025–2026)**  
**Author:** Liam Bavington-Hodge

This repository contains a Python framework for developing and evaluating machine-learning-guided clause selection in [iProver](https://gitlab.com/korovin/iprover), a first-order automated theorem prover.

The framework uses iProver's interactive external-agent interface to extract clause features, assign scores to candidate clauses and influence proof search. It supports hand-crafted heuristics, supervised machine learning, pairwise ranking and hybrid approaches.

## Features

The repository provides functionality for:

- **External clause selection** — Communicating with iProver through a TCP socket to receive clause information and return selection scores.
- **Hand-crafted heuristics** — Scoring clauses using weighted combinations of syntactic and structural features.
- **Training data collection** — Collecting labelled clause-level data from iProver proof-search runs.
- **Supervised learning** — Training logistic regression and multilayer perceptron (MLP) models for clause selection.
- **Pairwise ranking** — Training models to compare pairs of clauses rather than scoring them independently.
- **Hybrid approaches** — Combining machine-learning predictions with hand-crafted heuristic scores.
- **Dataset preparation** — Constructing and splitting benchmark datasets from the TPTP problem library.
- **Experimental evaluation** — Comparing clause-selection approaches using solving outcomes, runtimes and proof-search statistics.

## Repository Structure

```text
iprover-ml/
├── agents/          # External clause-selection agents
├── common/          # Shared feature extraction utilities
├── dataset/         # TPTP dataset construction and preparation
├── training/        # Training scripts, results and trained models
├── experiments/     # Experimental evaluation and results
└── README.md        # Project overview and documentation
```

Each directory contains its own README with further details on its contents and functionality.

## Requirements

The project was developed on Ubuntu using Python 3.10.

The following are required:

- Python 3.10
- A working iProver executable (`iproveropt`) with interactive external-agent support
- TPTP problem library (v9.2.1)
- Python packages including NumPy, pandas, scikit-learn and joblib

The examples below assume the following directory structure:

```text
workspace/
├── iproveropt
├── TPTPLibrary/
│   └── TPTP-v9.2.1/
└── iprover-ml/
```

### Setup

From the `iprover-ml/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pandas scikit-learn joblib matplotlib
export TPTP="$(realpath ../TPTPLibrary/TPTP-v9.2.1)"
```

Adjust the paths to iProver and TPTP as necessary.

## Usage

The following examples demonstrate the main functionality of the repository. All commands should be executed from the `iprover-ml/` directory.

### 1. Run Standard iProver

Run iProver on the included example TPTP problems without an external clause-selection agent:

```bash
python3 experiments/scripts/run_experiment.py \
  --problems dataset/problems/examples \
  --iprover-bin ../iproveropt \
  --tptp_root ../TPTPLibrary/TPTP-v9.2.1 \
  --results-csv baseline_examples.csv \
  --timeout 15
```

The script records solving outcomes, runtimes and proof-search statistics in `experiments/results/`.

### 2. Collect Training Data

Collect labelled clause-level training data using an external data-collection agent:

```bash
python3 training/scripts/collect_training_data.py \
  --problems dataset/problems/train_200 \
  --iprover-bin ../iproveropt \
  --tptp_root ../TPTPLibrary/TPTP-v9.2.1 \
  --collector-agent agents/collect_clause_data_given_passive.py \
  --collection-mode external_score \
  --clause-csv training/results/clause_train_200.csv \
  --summary-csv training/results/collection_summary.csv \
  --timeout 15 \
  --max-negatives-per-positive 10
```

Alternative collection agents are available in `agents/`, supporting different approaches to generating clause-selection labels.

The full training problem sets are not included in the repository but can be reconstructed using the scripts and dataset information in `dataset/`.

### 3. Train Machine Learning Models

The repository supports logistic regression and MLP models trained on extracted clause features.

**Logistic Regression**

```bash
python3 training/scripts/train_logreg.py \
  --train-csv training/results/clause_train_200.csv \
  --model-out training/models/example_logreg.joblib \
  --split-mode row \
  --c 1.0
```

**Multilayer Perceptron (MLP)**

```bash
python3 training/scripts/train_mlp.py \
  --train-csv training/results/clause_train_200.csv \
  --model-out training/models/example_mlp.joblib \
  --hidden-layer-sizes 64,32 \
  --learning-rate-init 0.001 \
  --alpha 0.0001
```

Both approaches support configurable model parameters and training/development splits.

Additional scripts support training on complete datasets and constructing pairwise ranking models.

Trained models are saved in `.joblib` format for use by the external agents.

### 4. Evaluate an External Agent

Run iProver using a trained MLP model to guide clause selection:

```bash
python3 experiments/scripts/run_experiment.py \
  --problems dataset/problems/examples \
  --iprover-bin ../iproveropt \
  --tptp_root ../TPTPLibrary/TPTP-v9.2.1 \
  --agent-cmd "python3 agents/mlpEA.py --host {host} --port {port} --model training/models/example_mlp.joblib" \
  --results-csv example_mlp_results.csv \
  --timeout 15
```

The experiment runner launches the external agent and manages communication with iProver. The `{host}` and `{port}` placeholders are populated automatically.

Other available agents include:

- `handEA.py` — Hand-crafted heuristic
- `logregEA.py` — Logistic regression
- `pairwise_logregEA.py` — Pairwise logistic regression
- `pairwise_mlpEA.py` — Pairwise MLP
- `hybrid_logregEA.py` — Hybrid heuristic and logistic regression
- `hybrid_mlpEA.py` — Hybrid heuristic and MLP

The experiment runner also supports configurable clause-selection queues through `--inst-passive-queues` and `--inst-passive-queues-freq`.

## Dataset

The project uses problems from the **TPTP (Thousands of Problems for Theorem Provers)** library.

The original benchmark dataset contains **1,701 problems** from five domains:

- ALG — Algebra
- COL — Combinatorics
- GEO — Geometry
- LAT — Lattices
- NUM — Number Theory

The dataset is divided into training, development and test sets.

Only a small selection of example TPTP problems is included in this repository. Scripts and CSV files for reconstructing the complete datasets are provided in `dataset/`.

## Results

The framework was used to compare standard iProver against several machine-learning-guided clause-selection strategies.

On a held-out test set of 264 problems, using a 15-second timeout:

| Strategy | Problems solved |
|---|---:|
| Standard iProver | **162 / 264** |
| Pure machine-learning clause selection | 64 / 264 |
| Machine learning with an age queue | 64 / 264 |

Although some learned approaches reduced search effort on individual problems, they did not outperform standard iProver in overall solving performance.

The repository includes experimental results and trained models for further inspection and comparison.

## Report

This repository accompanies the University of Manchester final-year project:

**Machine Learning for Clause Selection in First-Order Automated Theorem Proving** (2026).

The report provides further details on the methodology, implementation, experiments and findings.

**Report:** [https://doi.org/10.48420/34036326](https://doi.org/10.48420/34036326)