#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.baseEA import BaseEA
from common.feature_extraction import ClauseFeatureExtractor


FEATURE_COLS = [
    "num_literals",
    "num_symbols",
    "max_depth",
    "num_variables",
    "age",
    "shared_conj_symbols",
    "is_ground",
    "has_equality",
    "clause_weight",
]


def pairwise_scores_from_features(model, feature_rows: List[Dict[str, float]]) -> List[float]:
    n = len(feature_rows)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    pair_rows: List[Dict[str, float]] = []
    pair_index: List[int] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row = {
                f"delta_{feat}": float(feature_rows[i][feat]) - float(feature_rows[j][feat])
                for feat in FEATURE_COLS
            }
            pair_rows.append(row)
            pair_index.append(i)

    delta_cols = [f"delta_{feat}" for feat in FEATURE_COLS]
    X_pair = pd.DataFrame(pair_rows, columns=delta_cols)
    probs = model.predict_proba(X_pair)[:, 1]

    wins = [0.0 for _ in range(n)]
    counts = [0 for _ in range(n)]

    for idx, prob in zip(pair_index, probs):
        wins[idx] += float(prob)
        counts[idx] += 1

    win_rates = [
        wins[i] / counts[i] if counts[i] > 0 else 0.0
        for i in range(n)
    ]

    return [-float(w) for w in win_rates]


class MLPEA(BaseEA):
    def __init__(self, host: str, port: int, model_path: str, pairwise: bool = False):
        super().__init__(host=host, port=port)
        self.model = joblib.load(model_path)
        self.pairwise = bool(pairwise)
        self.feature_extractor = ClauseFeatureExtractor()
        self.clause_ages: Dict[int, int] = {}
        self.next_age = 0
        self.conjecture_symbols: Set[str] = set()

    def reset_connection_state(self) -> None:
        super().reset_connection_state()
        self.clause_ages.clear()
        self.next_age = 0
        self.conjecture_symbols.clear()

    def _handle_register_clauses(self, msg: Dict[str, Any]) -> None:
        clauses = msg.get("clauses", [])
        added = 0

        if isinstance(clauses, list):
            for item in clauses:
                if not isinstance(item, dict):
                    continue

                cid = item.get("clause_id")
                ctext = item.get("clause", "")
                if cid is None:
                    continue

                try:
                    cid_int = int(cid)
                except Exception:
                    continue

                ctext = str(ctext)
                self.clauses_by_id[cid_int] = ctext

                if cid_int not in self.clause_ages:
                    self.clause_ages[cid_int] = self.next_age
                    self.next_age += 1

                role = str(item.get("role", "")).strip().lower()
                if role in {"conjecture", "negated_conjecture"}:
                    self.conjecture_symbols |= self.feature_extractor._extract_symbols(ctext)

                added += 1

        print(
            f"register_clauses: stored/updated {added} clauses "
            f"(total={len(self.clauses_by_id)})",
            file=sys.stderr,
        )

    def extract_features(self, clause_id: int) -> Dict[str, float]:
        cid = int(clause_id)
        text = self.clauses_by_id.get(cid, "")
        age = int(self.clause_ages.get(cid, self.next_age))

        feats = self.feature_extractor.extract_features(
            clause_text=text,
            age=age,
            conjecture_symbols=self.conjecture_symbols,
        )

        return {
            "num_literals": float(feats.num_literals),
            "num_symbols": float(feats.num_symbols),
            "max_depth": float(feats.max_depth),
            "num_variables": float(feats.num_variables),
            "age": float(feats.age),
            "shared_conj_symbols": float(feats.shared_conj_symbols),
            "is_ground": float(feats.is_ground),
            "has_equality": float(feats.has_equality),
            "clause_weight": float(feats.clause_weight),
        }

    def score_clause_ids(self, clause_ids: List[Any]) -> List[float]:
        rows: List[Dict[str, float]] = []

        for cid in clause_ids:
            try:
                rows.append(self.extract_features(int(cid)))
            except Exception:
                rows.append({c: 0.0 for c in FEATURE_COLS})

        if self.pairwise:
            return pairwise_scores_from_features(self.model, rows)

        X = pd.DataFrame(rows, columns=FEATURE_COLS)
        probs = self.model.predict_proba(X)[:, 1]

        return [-float(p) for p in probs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--model",
        default="training/models/mlp_pipeline.joblib",
        help="Path to trained MLP model.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Treat loaded model as pairwise model trained on delta_* features.",
    )
    args = parser.parse_args()

    agent = MLPEA(
        host=args.host,
        port=args.port,
        model_path=args.model,
        pairwise=args.pairwise,
    )
    agent.run()