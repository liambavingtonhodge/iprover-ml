#!/usr/bin/env python3

import argparse
import json
import logging
import random
import re
import signal
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.baseEA import BaseEA, HOST, PORT
from common.feature_extraction import ClauseFeatureExtractor


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)


def iter_null_delimited_frames(conn: socket.socket):
    buf = b""
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            return
        buf += chunk
        while b"\x00" in buf:
            raw, buf = buf.split(b"\x00", 1)
            raw = raw.strip()
            if raw:
                yield raw


def send_json_line(conn: socket.socket, obj: Dict[str, Any]) -> None:
    conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def extract_conjecture_symbols(
    problem_file: Optional[Path],
    feature_extractor: ClauseFeatureExtractor,
) -> Set[str]:
    if problem_file is None or not problem_file.is_file():
        return set()

    try:
        text = problem_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return set()

    matches = re.findall(
        r"(?:fof|cnf|tff)\s*\([^,]+,\s*(?:conjecture|negated_conjecture)\s*,(.*?)\)\s*\.",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not matches:
        return set()

    merged = " ".join(matches)
    return feature_extractor._extract_symbols(merged)


class ClauseDatasetProofEA(BaseEA):
    FIELDNAMES = [
        "problem",
        "decision_index",
        "candidate_clause_id",
        "selected_clause_id",
        "selected",
        "num_literals",
        "num_symbols",
        "max_depth",
        "num_variables",
        "age",
        "shared_conj_symbols",
        "is_ground",
        "has_equality",
        "clause_weight",
        "clause_text",
    ]

    SOLVED_STATUSES = {"Theorem", "Unsatisfiable"}

    def __init__(
        self,
        host: str,
        port: int,
        problem_name: str = "",
        conjecture_symbols: Optional[Set[str]] = None,
        max_negatives_per_positive: int = 10,
        random_seed: int = 42,
        keep_unsolved: bool = False,
    ):
        super().__init__(host=host, port=port, reset_on_disconnect=False)

        self.problem_name = problem_name
        self.conjecture_symbols = conjecture_symbols or set()
        self.feature_extractor = ClauseFeatureExtractor()

        self.event_counter: int = 0
        self.first_seen_event_by_id: Dict[int, int] = {}

        self.proof_clause_ids: Set[int] = set()
        self.szs_status: str = ""

        self.max_negatives_per_positive = max_negatives_per_positive
        self.rng = random.Random(random_seed)
        self.keep_unsolved = keep_unsolved

        self.pending_scores_req: Optional[Dict[str, Any]] = None

        self.collected_rows: List[Dict[str, Any]] = []
        self._dumped = False

    def reset_connection_state(self) -> None:
        super().reset_connection_state()
        self.event_counter = 0
        self.first_seen_event_by_id.clear()
        self.proof_clause_ids.clear()
        self.szs_status = ""
        self.pending_scores_req = None
        self.collected_rows.clear()
        self._dumped = False

    def _normalise_clause_ids(self, clause_ids: Any) -> List[int]:
        out: List[int] = []
        if not isinstance(clause_ids, list):
            return out

        for cid in clause_ids:
            try:
                out.append(int(cid))
            except Exception:
                continue

        return out

    def _handle_register_clauses(self, msg: Dict[str, Any]) -> None:
        clauses = msg.get("clauses", [])
        added = 0

        if not isinstance(clauses, list):
            return

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

            self.clauses_by_id[cid_int] = str(ctext)

            if cid_int not in self.first_seen_event_by_id:
                self.first_seen_event_by_id[cid_int] = self.event_counter

            added += 1

        logging.info(
            "register_clauses: stored/updated %d clauses (total=%d)",
            added,
            len(self.clauses_by_id),
        )

    def _handle_szs_result(self, msg: Dict[str, Any]) -> None:
        raw_status = str(msg.get("szs_status", ""))
        self.szs_status = raw_status

        for status in self.SOLVED_STATUSES:
            if status in raw_status:
                self.szs_status = status
                return

    def _make_row(self, cid: int, selected: int) -> Optional[Dict[str, Any]]:
        clause_text = self.clauses_by_id.get(cid, "")
        if not clause_text:
            return None

        first_seen = self.first_seen_event_by_id.get(cid, self.event_counter)
        age = max(0, self.event_counter - first_seen)

        feats = self.feature_extractor.extract_features(
            clause_text=clause_text,
            age=age,
            conjecture_symbols=self.conjecture_symbols,
        )

        return {
            "problem": self.problem_name,
            "decision_index": -1,
            "candidate_clause_id": cid,
            "selected_clause_id": cid if selected else -1,
            "selected": selected,
            "num_literals": feats.num_literals,
            "num_symbols": feats.num_symbols,
            "max_depth": feats.max_depth,
            "num_variables": feats.num_variables,
            "age": feats.age,
            "shared_conj_symbols": feats.shared_conj_symbols,
            "is_ground": feats.is_ground,
            "has_equality": feats.has_equality,
            "clause_weight": feats.clause_weight,
            "clause_text": clause_text,
        }

    def _sample_negatives(self, negatives: List[int], num_positives: int) -> List[int]:
        if self.max_negatives_per_positive < 0:
            return negatives

        max_negs = self.max_negatives_per_positive * max(1, num_positives)

        if len(negatives) > max_negs:
            return self.rng.sample(negatives, max_negs)

        return negatives

    def _build_rows_from_proof_membership(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        if not self.proof_clause_ids and not self.keep_unsolved:
            logging.info(
                "No proof_out received; producing no rows for unsolved/non-proof run."
            )
            return rows

        positives = [
            cid for cid in sorted(self.proof_clause_ids)
            if cid in self.clauses_by_id
        ]

        negatives = [
            cid for cid in sorted(self.clauses_by_id.keys())
            if cid not in self.proof_clause_ids
        ]

        negatives = self._sample_negatives(negatives, num_positives=len(positives))

        kept_ids = positives + negatives

        for cid in kept_ids:
            selected = 1 if cid in self.proof_clause_ids else 0
            row = self._make_row(cid=cid, selected=selected)
            if row is not None:
                rows.append(row)

        logging.info(
            "Built %d proof-membership rows: positives=%d negatives=%d total_registered=%d",
            len(rows),
            len(positives),
            len(negatives),
            len(self.clauses_by_id),
        )

        return rows

    def dump_rows_to_stdout(self) -> None:
        if self._dumped:
            return

        self.collected_rows = self._build_rows_from_proof_membership()
        self._dumped = True

        payload = {
            "rows": self.collected_rows,
            "summary": {
                "problem": self.problem_name,
                "szs_status": self.szs_status,
                "registered_clauses": len(self.clauses_by_id),
                "proof_clauses": len(self.proof_clause_ids),
                "rows": len(self.collected_rows),
            },
        }

        sys.stdout.write(json.dumps(payload))
        sys.stdout.flush()

    def handle(self, conn: socket.socket):
        for raw in iter_null_delimited_frames(conn):
            self.event_counter += 1

            try:
                msg = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                logging.warning("Bad JSON: %r", raw[:200])
                continue

            tag = msg.get("tag")
            if not tag:
                continue

            if tag == "register_clauses":
                self._handle_register_clauses(msg)
                continue

            if tag == "scores_req":
                self.pending_scores_req = msg
                continue

            if tag == "server_queries_start":
                send_json_line(conn, {"tag": "server_queries_end"})

                if self.pending_scores_req is None:
                    send_json_line(conn, {"tag": "scores_res", "scores": []})
                else:
                    clause_ids = self.pending_scores_req.get("clause_ids", [])
                    scores = self.score_clause_ids(clause_ids)
                    send_json_line(conn, {"tag": "scores_res", "scores": scores})
                    self.pending_scores_req = None

                continue

            if tag == "proof_out":
                proof_ids = self._normalise_clause_ids(msg.get("clause_ids", []))
                self.proof_clause_ids.update(proof_ids)
                logging.info("proof_out: received %d proof clause ids", len(proof_ids))
                continue

            if tag == "szs_result_out":
                self._handle_szs_result(msg)
                continue

            continue

    def score_clause_ids(self, clause_ids: List[Any]) -> List[float]:
        return [0.0 for _ in clause_ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pure proof-based collector: all registered clauses are labelled by "
            "membership in iProver's proof_out clause-id set."
        )
    )

    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--problem-name", default="")
    parser.add_argument("--problem-file", default="")
    parser.add_argument("--debug-log", default="")

    parser.add_argument(
        "--max-negatives-per-positive",
        type=int,
        default=10,
        help=(
            "Keep at most this many negative clauses per positive proof clause. "
            "Use -1 to keep all non-proof clauses."
        ),
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--keep-unsolved",
        action="store_true",
        help=(
            "If set, output all registered clauses as negatives when no proof_out "
            "is received. Normally leave this off for proof-based learning."
        ),
    )

    args = parser.parse_args()

    feature_extractor = ClauseFeatureExtractor()
    problem_file = Path(args.problem_file).resolve() if args.problem_file else None
    conjecture_symbols = extract_conjecture_symbols(problem_file, feature_extractor)

    agent = ClauseDatasetProofEA(
        host=args.host,
        port=args.port,
        problem_name=args.problem_name,
        conjecture_symbols=conjecture_symbols,
        max_negatives_per_positive=args.max_negatives_per_positive,
        random_seed=args.random_seed,
        keep_unsolved=args.keep_unsolved,
    )

    def handle_exit(signum, frame):
        agent.dump_rows_to_stdout()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)

    try:
        agent.run()
    finally:
        agent.dump_rows_to_stdout()


if __name__ == "__main__":
    main()