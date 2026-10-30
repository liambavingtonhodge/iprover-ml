#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Make sibling folders like common/ importable when running:
# python3 agents/handEA.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.baseEA import BaseEA, HOST, PORT
from common.feature_extraction import ClauseFeatures, ClauseFeatureExtractor


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)


class HandHeuristic:
    """
    Hand-coded linear heuristic.

    IMPORTANT:
    With:

        --sup_passive_queues "[[-external_score]]"

    LOWER returned score = better.

    So this class returns a cost directly.
    """

    # Cost weights: larger cost = worse clause
    W_NUM_LITERALS = 5.0
    W_NUM_SYMBOLS = 1.0
    W_MAX_DEPTH = 3.0
    W_NUM_VARIABLES = 2.0
    W_AGE = 0.02
    W_SHARED_CONJ_SYMBOLS = -4.0
    W_IS_GROUND = -2.0
    W_HAS_EQUALITY = 1.5
    W_CLAUSE_WEIGHT = 1.0

    def score_clause(self, feats: ClauseFeatures) -> float:
        return float(
            self.W_NUM_LITERALS * feats.num_literals
            + self.W_NUM_SYMBOLS * feats.num_symbols
            + self.W_MAX_DEPTH * feats.max_depth
            + self.W_NUM_VARIABLES * feats.num_variables
            + self.W_AGE * feats.age
            + self.W_SHARED_CONJ_SYMBOLS * feats.shared_conj_symbols
            + self.W_IS_GROUND * feats.is_ground
            + self.W_HAS_EQUALITY * feats.has_equality
            + self.W_CLAUSE_WEIGHT * feats.clause_weight
        )


class HandHeuristicEA(BaseEA):
    def __init__(self, host: str = HOST, port: int = PORT):
        super().__init__(host=host, port=port)

        self.feature_extractor = ClauseFeatureExtractor()
        self.heuristic = HandHeuristic()

        self.current_age: int = 0
        self.first_seen_age_by_id: Dict[int, int] = {}
        self.current_conjecture_symbols: Set[str] = set()

    def reset_connection_state(self) -> None:
        super().reset_connection_state()
        self.current_age = 0
        self.first_seen_age_by_id.clear()
        self.current_conjecture_symbols.clear()

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

                if cid_int not in self.first_seen_age_by_id:
                    self.first_seen_age_by_id[cid_int] = self.current_age

                role = str(item.get("role", "")).strip().lower()
                if role in {"conjecture", "negated_conjecture"}:
                    self.current_conjecture_symbols |= self.feature_extractor._extract_symbols(ctext)

                added += 1

        logging.info(
            "register_clauses: stored/updated %d clauses (total=%d, conjecture_symbols=%d)",
            added,
            len(self.clauses_by_id),
            len(self.current_conjecture_symbols),
        )

    def score_clause_ids(self, clause_ids: List[Any]) -> List[float]:
        scores: List[float] = []

        # Advance age once per scoring round
        self.current_age += 1

        for raw_cid in clause_ids:
            try:
                cid = int(raw_cid)
            except Exception:
                logging.warning("Could not parse clause id: %r", raw_cid)
                scores.append(1e9)
                continue

            clause_text = self.clauses_by_id.get(cid, "")
            if not clause_text:
                logging.warning("No text found for clause id %s", cid)
                scores.append(1e6)
                continue

            born_age = self.first_seen_age_by_id.get(cid, self.current_age)
            age = max(0, self.current_age - born_age)

            feats = self.feature_extractor.extract_features(
                clause_text=clause_text,
                age=age,
                conjecture_symbols=self.current_conjecture_symbols,
            )

            score = self.heuristic.score_clause(feats)
            scores.append(score)

            logging.info(
                "Clause %s: score=%.3f | lits=%d syms=%d depth=%d vars=%d age=%d conj=%d ground=%d eq=%d weight=%d | text=%r",
                cid,
                score,
                feats.num_literals,
                feats.num_symbols,
                feats.max_depth,
                feats.num_variables,
                feats.age,
                feats.shared_conj_symbols,
                feats.is_ground,
                feats.has_equality,
                feats.clause_weight,
                clause_text[:160],
            )

        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    agent = HandHeuristicEA(host=args.host, port=args.port)
    agent.run()