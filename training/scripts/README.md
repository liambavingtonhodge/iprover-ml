# Model Training

This directory contains the scripts, results and trained models used to develop supervised machine-learning approaches for clause selection in iProver.

## Directory Structure

- **`scripts/`** — Contains scripts for collecting training data, preparing datasets and training machine-learning models.
- **`results/`** — Contains training results, performance metrics and related outputs.
- **`models/`** — Contains saved trained models for use in clause selection.

## Scripts

| Script | Description |
|---|---|
| `collect_training_data.py` | Collects clause-level training data from interactive iProver runs, recording clause features and selection labels. |
| `build_pairwise_dataset.py` | Converts clause-level training data into pairwise comparisons using differences between clause features. |
| `train_logreg.py` | Trains and evaluates a logistic regression model using a training/development split. |
| `train_logreg_no_split.py` | Trains a logistic regression model on the full dataset without a separate development split. |
| `train_mlp.py` | Trains and evaluates a multilayer perceptron (MLP) model using a training/development split. |
| `train_mlp_no_split.py` | Trains an MLP model on the full dataset without a separate development split. |
| `train_pairwise_logreg.py` | Trains a pairwise logistic regression ranking model using differences between clause features. |
| `train_pairwise_mlp.py` | Trains a pairwise MLP ranking model using differences between clause features. |

## Training Approach

The models use nine numerical and structural clause features extracted using the shared utilities in `common/`.

Both logistic regression and MLP models are supported, with approaches based on individual clause classification and pairwise clause ranking.

The training scripts support configurable model parameters and evaluation using training and development datasets. Trained models are saved in `.joblib` format.