# External Agents

This directory contains the external agents used to guide clause selection in iProver. The agents communicate with iProver through a TCP socket, receiving clause information and returning scores that influence proof search.

The agents include hand-crafted heuristics, supervised machine learning models, hybrid approaches, and data collection utilities.

## Scripts

### Core Infrastructure

- **`baseEA.py`** — Provides the shared infrastructure for external agents, including socket communication, JSON message handling, clause registration, and scoring requests.

- **`dummyEA.py`** — Baseline agent that assigns every clause a score of zero. Used to test communication with iProver without applying a clause-selection heuristic.

### Data Collection Agents

- **`collect_clause_data_given_passive.py`** — Collects clause-level training data using iProver's given-clause selections. Reconstructs candidate sets from passive-clause information and labels clauses according to whether they were selected.

- **`collect_clause_data_given_scoresreq.py`** — Collects training data using the candidate clauses supplied in scoring requests. Labels the selected given clause as positive and the remaining candidates as negative.

- **`collect_clause_data_proof.py`** — Collects proof-based training data by identifying clauses appearing in completed proofs. Labels proof clauses as positive and other registered clauses as negative.

### Heuristic Agent

- **`handEA.py`** — Implements a hand-crafted clause-selection heuristic using a weighted combination of syntactic features, including clause length, symbol count, depth, age, and conjecture overlap.

### Machine Learning Agents

- **`logregEA.py`** — Uses a trained logistic regression model to score clauses based on extracted features. Supports both pointwise classification and pairwise ranking.

- **`mlpEA.py`** — Uses a trained multilayer perceptron (MLP) to score clauses based on extracted features. Supports both pointwise classification and pairwise ranking.

- **`pairwise_logregEA.py`** — Uses a pairwise logistic regression model to compare candidate clauses and rank them according to their average predicted preference over other candidates.

- **`pairwise_mlpEA.py`** — Uses a pairwise MLP model to compare candidate clauses and rank them according to their average predicted preference over other candidates.

### Hybrid Agents

- **`hybrid_logregEA.py`** — Combines logistic regression predictions with the hand-crafted heuristic using a weighted scoring function. Supports configurable weighting, score normalisation, and pairwise ranking.

- **`hybrid_mlpEA.py`** — Combines MLP predictions with the hand-crafted heuristic using a weighted scoring function. Supports configurable weighting, score normalisation, and pairwise ranking.

## Notes

All agents use the external scoring interface provided by iProver.

When using the `[-external_score]` selection criterion, lower scores indicate higher clause priority.

The machine learning agents require trained models, which are produced by the scripts in the `training/` directory. Clause features are extracted using the shared utilities in `common/`.

The data collection agents are intended to be used alongside the training data collection scripts, while the scoring agents can be evaluated using the scripts in `experiments/`.