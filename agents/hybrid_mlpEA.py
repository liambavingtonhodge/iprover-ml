#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.baseEA import BaseEA
from common.feature_extraction import ClauseFeatures, ClauseFeatureExtractor


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)


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

    # Lower score is better for [[-external_score]]
    return [-float(w) for w in win_rates]


class SafeHeuristic:
    """
    Hand-coded linear heuristic copied from handEA.py.

    Lower score = better when using [[-external_score]].
    """

    def __init__(
        self,
        num_literals_weight: float = 5.0,
        num_symbols_weight: float = 1.0,
        max_depth_weight: float = 3.0,
        num_variables_weight: float = 2.0,
        age_weight: float = 0.02,
        shared_conj_symbols_weight: float = -4.0,
        is_ground_weight: float = -2.0,
        has_equality_weight: float = 1.5,
        clause_weight_weight: float = 1.0,
    ):
        self.num_literals_weight = float(num_literals_weight)
        self.num_symbols_weight = float(num_symbols_weight)
        self.max_depth_weight = float(max_depth_weight)
        self.num_variables_weight = float(num_variables_weight)
        self.age_weight = float(age_weight)
        self.shared_conj_symbols_weight = float(shared_conj_symbols_weight)
        self.is_ground_weight = float(is_ground_weight)
        self.has_equality_weight = float(has_equality_weight)
        self.clause_weight_weight = float(clause_weight_weight)

    def score_clause(self, feats: ClauseFeatures) -> float:
        return float(
            self.num_literals_weight * feats.num_literals
            + self.num_symbols_weight * feats.num_symbols
            + self.max_depth_weight * feats.max_depth
            + self.num_variables_weight * feats.num_variables
            + self.age_weight * feats.age
            + self.shared_conj_symbols_weight * feats.shared_conj_symbols
            + self.is_ground_weight * feats.is_ground
            + self.has_equality_weight * feats.has_equality
            + self.clause_weight_weight * feats.clause_weight
        )


class HybridMLPEA(BaseEA):
    def __init__(
        self,
        host: str,
        port: int,
        model_path: str,
        alpha: float = 0.1,
        num_literals_weight: float = 5.0,
        num_symbols_weight: float = 1.0,
        max_depth_weight: float = 3.0,
        num_variables_weight: float = 2.0,
        age_weight: float = 0.02,
        shared_conj_symbols_weight: float = -4.0,
        is_ground_weight: float = -2.0,
        has_equality_weight: float = 1.5,
        clause_weight_weight: float = 1.0,
        normalize: bool = True,
        pairwise: bool = False,
    ):
        super().__init__(host=host, port=port)

        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1")

        self.model = joblib.load(model_path)
        self.pairwise = bool(pairwise)
        self.feature_extractor = ClauseFeatureExtractor()

        self.heuristic = SafeHeuristic(
            num_literals_weight=num_literals_weight,
            num_symbols_weight=num_symbols_weight,
            max_depth_weight=max_depth_weight,
            num_variables_weight=num_variables_weight,
            age_weight=age_weight,
            shared_conj_symbols_weight=shared_conj_symbols_weight,
            is_ground_weight=is_ground_weight,
            has_equality_weight=has_equality_weight,
            clause_weight_weight=clause_weight_weight,
        )

        self.alpha = float(alpha)
        self.normalize = bool(normalize)

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

        logging.info(
            "register_clauses: stored/updated %d clauses (total=%d)",
            added,
            len(self.clauses_by_id),
        )

    def extract_clause_features(self, clause_id: int) -> ClauseFeatures:
        cid = int(clause_id)
        text = self.clauses_by_id.get(cid, "")
        age = int(self.clause_ages.get(cid, self.next_age))

        return self.feature_extractor.extract_features(
            clause_text=text,
            age=age,
            conjecture_symbols=self.conjecture_symbols,
        )

    @staticmethod
    def _minmax(values: List[float]) -> List[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi - lo < 1e-12:
            return [0.0 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    def score_clause_ids(self, clause_ids: List[Any]) -> List[float]:
        feature_rows: List[Dict[str, float]] = []
        structured_feats: List[ClauseFeatures] = []

        for cid in clause_ids:
            try:
                feats = self.extract_clause_features(int(cid))
                structured_feats.append(feats)
                feature_rows.append({
                    "num_literals": float(feats.num_literals),
                    "num_symbols": float(feats.num_symbols),
                    "max_depth": float(feats.max_depth),
                    "num_variables": float(feats.num_variables),
                    "age": float(feats.age),
                    "shared_conj_symbols": float(feats.shared_conj_symbols),
                    "is_ground": float(feats.is_ground),
                    "has_equality": float(feats.has_equality),
                    "clause_weight": float(feats.clause_weight),
                })
            except Exception:
                zero_feats = ClauseFeatures(
                    num_literals=0,
                    num_symbols=0,
                    max_depth=0,
                    num_variables=0,
                    age=0,
                    shared_conj_symbols=0,
                    is_ground=0,
                    has_equality=0,
                    clause_weight=0,
                )
                structured_feats.append(zero_feats)
                feature_rows.append({c: 0.0 for c in FEATURE_COLS})

        safe_scores = [self.heuristic.score_clause(feats) for feats in structured_feats]

        if self.pairwise:
            ml_scores = pairwise_scores_from_features(self.model, feature_rows)
        else:
            X = pd.DataFrame(feature_rows, columns=FEATURE_COLS)
            probs = self.model.predict_proba(X)[:, 1]
            ml_scores = [-float(p) for p in probs]

        if self.normalize:
            ml_scores = self._minmax(ml_scores)
            safe_scores = self._minmax(safe_scores)

        return [
            (1.0 - self.alpha) * safe + self.alpha * ml
            for safe, ml in zip(safe_scores, ml_scores)
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid MLP external agent using ML + handEA-style heuristic."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)

    parser.add_argument("--num-literals-weight", type=float, default=5.0)
    parser.add_argument("--num-symbols-weight", type=float, default=1.0)
    parser.add_argument("--max-depth-weight", type=float, default=3.0)
    parser.add_argument("--num-variables-weight", type=float, default=2.0)
    parser.add_argument("--age-weight", type=float, default=0.02)
    parser.add_argument("--shared-conj-symbols-weight", type=float, default=-4.0)
    parser.add_argument("--is-ground-weight", type=float, default=-2.0)
    parser.add_argument("--has-equality-weight", type=float, default=1.5)
    parser.add_argument("--clause-weight-weight", type=float, default=1.0)

    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Treat loaded model as pairwise model trained on delta_* features.",
    )

    args = parser.parse_args()

    agent = HybridMLPEA(
        host=args.host,
        port=args.port,
        model_path=args.model,
        alpha=args.alpha,
        num_literals_weight=args.num_literals_weight,
        num_symbols_weight=args.num_symbols_weight,
        max_depth_weight=args.max_depth_weight,
        num_variables_weight=args.num_variables_weight,
        age_weight=args.age_weight,
        shared_conj_symbols_weight=args.shared_conj_symbols_weight,
        is_ground_weight=args.is_ground_weight,
        has_equality_weight=args.has_equality_weight,
        clause_weight_weight=args.clause_weight_weight,
        normalize=not args.no_normalize,
        pairwise=args.pairwise,
    )

    agent.run()