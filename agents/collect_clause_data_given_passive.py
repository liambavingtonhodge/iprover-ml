#!/usr/bin/env python3

import argparse
import json
import logging
import random
import signal
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

    import re

    matches = re.findall(
        r"(?:fof|cnf|tff)\s*\([^,]+,\s*(?:conjecture|negated_conjecture)\s*,(.*?)\)\s*\.",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not matches:
        return set()

    merged = " ".join(matches)
    return feature_extractor._extract_symbols(merged)


class ClauseDatasetGivenEA(BaseEA):
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

    def __init__(
        self,
        host: str,
        port: int,
        problem_name: str = "",
        conjecture_symbols: Optional[Set[str]] = None,
        max_negatives_per_positive: int = 10,
        random_seed: int = 42,
        debug_log: Optional[Path] = None,
    ):
        super().__init__(host=host, port=port)
        self.problem_name = problem_name
        self.conjecture_symbols = conjecture_symbols or set()

        self.feature_extractor = ClauseFeatureExtractor()

        self.pending_scores_req: Optional[Dict[str, Any]] = None

        self.event_counter: int = 0
        self.decision_index: int = 0
        self.first_seen_event_by_id: Dict[int, int] = {}

        self.max_negatives_per_positive = max_negatives_per_positive
        self.rng = random.Random(random_seed)

        self.collected_rows: List[Dict[str, Any]] = []
        self._dumped = False

        self.latest_passive_by_component: Dict[Tuple[str, int], List[int]] = {}
        self.latest_scores_req_by_component: Dict[Tuple[str, int], List[int]] = {}

        self.debug_log = debug_log

    def _trace(self, message: str) -> None:
        if self.debug_log is None:
            return
        self.debug_log.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_log.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def reset_connection_state(self) -> None:
        super().reset_connection_state()
        self.pending_scores_req = None
        self.event_counter = 0
        self.decision_index = 0
        self.first_seen_event_by_id.clear()
        self.collected_rows.clear()
        self.latest_passive_by_component.clear()
        self.latest_scores_req_by_component.clear()

    def dump_rows_to_stdout(self) -> None:
        if self._dumped:
            return
        self._dumped = True
        payload = {"rows": self.collected_rows}
        sys.stdout.write(json.dumps(payload))
        sys.stdout.flush()

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

    def _component_key_from_msg(self, msg: Dict[str, Any]) -> Tuple[str, int]:
        component = str(msg.get("component", "unknown"))
        try:
            component_id = int(msg.get("component_id", -1))
        except Exception:
            component_id = -1
        return (component, component_id)

    def _extract_clause_id(self, msg: Dict[str, Any]) -> Optional[int]:
        clause_ids = msg.get("clause_ids")
        if isinstance(clause_ids, list) and clause_ids:
            try:
                return int(clause_ids[0])
            except Exception:
                pass

        direct_candidates = [
            msg.get("clause_id"),
            msg.get("id"),
            msg.get("given_clause_id"),
            msg.get("selected_clause_id"),
        ]

        for key in ["clause", "given_clause", "selected_clause"]:
            obj = msg.get(key)
            if isinstance(obj, dict):
                nested_clause_ids = obj.get("clause_ids")
                if isinstance(nested_clause_ids, list) and nested_clause_ids:
                    try:
                        return int(nested_clause_ids[0])
                    except Exception:
                        pass
                direct_candidates.extend([obj.get("clause_id"), obj.get("id")])

        for value in direct_candidates:
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                pass

        return None

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

                self.clauses_by_id[cid_int] = str(ctext)
                if cid_int not in self.first_seen_event_by_id:
                    self.first_seen_event_by_id[cid_int] = self.event_counter
                added += 1

        logging.info(
            "register_clauses: stored/updated %d clauses (total=%d)",
            added,
            len(self.clauses_by_id),
        )

    def _make_row(self, cid: int, chosen_clause_id: int, selected: int) -> Optional[Dict[str, Any]]:
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
            "decision_index": self.decision_index,
            "candidate_clause_id": cid,
            "selected_clause_id": chosen_clause_id,
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

    def _candidate_ids_for_given_clause(
        self,
        chosen_clause_id: int,
        component_key: Tuple[str, int],
    ) -> List[int]:
        passive_ids = self.latest_passive_by_component.get(component_key)
        if passive_ids:
            return list(dict.fromkeys(passive_ids))

        scores_req_ids = self.latest_scores_req_by_component.get(component_key)
        if scores_req_ids:
            logging.warning(
                "No passive_clauses found for component=%r; falling back to scores_req clause_ids",
                component_key,
            )
            return list(dict.fromkeys(scores_req_ids))

        logging.warning(
            "No passive_clauses or scores_req candidates for component=%r; falling back to chosen clause only",
            component_key,
        )
        return [chosen_clause_id]

    def _build_rows_for_decision(
        self,
        chosen_clause_id: int,
        component_key: Tuple[str, int],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        self.decision_index += 1

        candidate_ids = self._candidate_ids_for_given_clause(chosen_clause_id, component_key)

        positives = [cid for cid in candidate_ids if cid == chosen_clause_id]
        negatives = [cid for cid in candidate_ids if cid != chosen_clause_id]

        if not positives:
            positives = [chosen_clause_id]

        if self.max_negatives_per_positive >= 0 and len(negatives) > self.max_negatives_per_positive:
            negatives = self.rng.sample(negatives, self.max_negatives_per_positive)

        kept_ids = positives + negatives

        for cid in kept_ids:
            selected = 1 if cid == chosen_clause_id else 0
            row = self._make_row(cid, chosen_clause_id, selected)
            if row is not None:
                rows.append(row)

        return rows

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
                component_key = self._component_key_from_msg(msg)
                self.latest_scores_req_by_component[component_key] = self._normalise_clause_ids(
                    msg.get("clause_ids", [])
                )
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

            if tag == "passive_clauses":
                component_key = self._component_key_from_msg(msg)
                clause_ids = self._normalise_clause_ids(msg.get("clause_ids", []))
                existing = self.latest_passive_by_component.get(component_key, [])
                merged = existing + [cid for cid in clause_ids if cid not in existing]
                self.latest_passive_by_component[component_key] = merged
                continue

            if tag == "simplified_clauses":
                component_key = self._component_key_from_msg(msg)
                removed_ids = set(self._normalise_clause_ids(msg.get("clause_ids", [])))

                if component_key in self.latest_passive_by_component:
                    self.latest_passive_by_component[component_key] = [
                        cid
                        for cid in self.latest_passive_by_component[component_key]
                        if cid not in removed_ids
                    ]
                continue

            if tag == "moved_from_active_to_passive":
                component_key = self._component_key_from_msg(msg)
                moved_ids = self._normalise_clause_ids(msg.get("clause_ids", []))
                if moved_ids:
                    existing = self.latest_passive_by_component.get(component_key, [])
                    merged = existing + [cid for cid in moved_ids if cid not in existing]
                    self.latest_passive_by_component[component_key] = merged
                continue

            if tag == "given_clause":
                chosen_clause_id = self._extract_clause_id(msg)
                if chosen_clause_id is None:
                    logging.warning("given_clause received but no clause id could be extracted")
                    logging.warning("Raw given_clause message: %r", msg)
                    continue

                component_key = self._component_key_from_msg(msg)
                rows = self._build_rows_for_decision(chosen_clause_id, component_key)
                self.collected_rows.extend(rows)

                if component_key in self.latest_passive_by_component:
                    self.latest_passive_by_component[component_key] = [
                        cid
                        for cid in self.latest_passive_by_component[component_key]
                        if cid != chosen_clause_id
                    ]
                continue

            continue

    def score_clause_ids(self, clause_ids: List[Any]) -> List[float]:
        return [0.0 for _ in clause_ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect clause-level training data with target = selected next by iProver (given_clause)."
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
        help="Keep at most this many negative candidate clauses per positive (default: 10). Use -1 to keep all.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for negative sampling (default: 42).",
    )
    args = parser.parse_args()

    feature_extractor = ClauseFeatureExtractor()
    problem_file = Path(args.problem_file).resolve() if args.problem_file else None
    conjecture_symbols = extract_conjecture_symbols(problem_file, feature_extractor)

    debug_log = Path(args.debug_log).resolve() if args.debug_log else None

    agent = ClauseDatasetGivenEA(
        host=args.host,
        port=args.port,
        problem_name=args.problem_name,
        conjecture_symbols=conjecture_symbols,
        max_negatives_per_positive=args.max_negatives_per_positive,
        random_seed=args.random_seed,
        debug_log=debug_log,
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