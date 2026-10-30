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


BASE_FEATURE_COLS = [
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

DELTA_FEATURE_COLS = [f"delta_{c}" for c in BASE_FEATURE_COLS]


class PairwiseMLPEA(BaseEA):
    """
    Pairwise ranking agent using within-batch aggregation.

    For each candidate clause c_i in a scores_req batch:
      - compare it against every other candidate c_j
      - compute P(c_i beats c_j)
      - average these probabilities
      - return negative average preference as the external score
        (lower = better)
    """

    def __init__(self, host: str, port: int, model_path: str):
        super().__init__(host=host, port=port)
        self.model = joblib.load(model_path)
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
            f"(total={len(self.clauses_by_id)})"
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

    @staticmethod
    def _zero_feature_dict() -> Dict[str, float]:
        return {c: 0.0 for c in BASE_FEATURE_COLS}

    def score_clause_ids(self, clause_ids: List[Any]) -> List[float]:
        parsed_ids: List[int] = []
        features_by_id: Dict[int, Dict[str, float]] = {}

        for cid in clause_ids:
            try:
                cid_int = int(cid)
                parsed_ids.append(cid_int)
                features_by_id[cid_int] = self.extract_features(cid_int)
            except Exception:
                continue

        n = len(parsed_ids)
        if n == 0:
            return [0.0 for _ in clause_ids]

        if n == 1:
            single_id = parsed_ids[0]
            pref_by_id = {single_id: 1.0}
        else:
            pair_rows: List[Dict[str, float]] = []
            pair_meta: List[tuple[int, int]] = []

            for cid_i in parsed_ids:
                fi = features_by_id.get(cid_i, self._zero_feature_dict())
                for cid_j in parsed_ids:
                    if cid_i == cid_j:
                        continue
                    fj = features_by_id.get(cid_j, self._zero_feature_dict())

                    delta = {
                        f"delta_{k}": float(fi[k]) - float(fj[k])
                        for k in BASE_FEATURE_COLS
                    }
                    pair_rows.append(delta)
                    pair_meta.append((cid_i, cid_j))

            X = pd.DataFrame(pair_rows, columns=DELTA_FEATURE_COLS)
            probs = self.model.predict_proba(X)[:, 1]

            pref_sum: Dict[int, float] = {cid: 0.0 for cid in parsed_ids}
            pref_cnt: Dict[int, int] = {cid: 0 for cid in parsed_ids}

            for (cid_i, _cid_j), p in zip(pair_meta, probs):
                pref_sum[cid_i] += float(p)
                pref_cnt[cid_i] += 1

            pref_by_id = {
                cid: (pref_sum[cid] / pref_cnt[cid]) if pref_cnt[cid] > 0 else 0.5
                for cid in parsed_ids
            }

        scores: List[float] = []
        for cid in clause_ids:
            try:
                cid_int = int(cid)
                pref = pref_by_id.get(cid_int, 0.5)
                scores.append(-float(pref))
            except Exception:
                scores.append(0.0)

        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained pairwise MLP model",
    )
    args = parser.parse_args()

    agent = PairwiseMLPEA(
        host=args.host,
        port=args.port,
        model_path=args.model,
    )
    agent.run()