#!/usr/bin/env python3

import json
import logging
import re
import socket
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set

# Default values (will be overridden by CLI in practice)
HOST = "127.0.0.1"
PORT = 5000

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)


# ============================================================
# Wire protocol helpers
# ============================================================

def iter_null_delimited_frames(conn: socket.socket):
    """
    Reads a raw byte stream from the TCP socket that iProver sends,
    where each message is terminated by a NUL byte (\x00),
    and turns them into individual JSON frames.
    """
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
    """
    Sends JSON message from agent to iProver through TCP socket,
    by converting a Python dictionary into JSON text,
    adding a newline at the end,
    converting it to bytes,
    and sending it over the TCP connection.
    """
    conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))


# ============================================================
# Base EA
# ============================================================

class BaseEA:
    """
    Base external agent for iProver.

    This keeps the protocol handling as close as possible to the working dummy EA.
    Subclasses only need to override score_clause_ids().

    Host and port are passed in (typically from CLI) so that experiment scripts
    can choose a free port dynamically. The constants above are only defaults.

    By default, reset_on_disconnect=True to preserve the original behaviour.
    Collection agents that need to keep state until process shutdown can set
    reset_on_disconnect=False.
    """

    def __init__(
        self,
        host: str = HOST,
        port: int = PORT,
        reset_on_disconnect: bool = True,
    ):
        self.host = host
        self.port = port
        self.reset_on_disconnect = reset_on_disconnect
        self.clauses_by_id: Dict[int, str] = {}
        self.pending_scores_req: Optional[Dict[str, Any]] = None

    def run(self):
        """
        Opens a socket, waits for iProver to connect,
        and then runs the communication loop (handle) for that connection.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(1)
            logging.info(f"EA listening on {self.host}:{self.port}")

            while True:
                conn, addr = s.accept()
                logging.info(f"Connection from {addr}")
                try:
                    self.handle(conn)
                except Exception as e:
                    logging.exception(e)
                finally:
                    conn.close()
                    logging.info("Connection closed")
                    if self.reset_on_disconnect:
                        self.reset_connection_state()

    def reset_connection_state(self) -> None:
        """
        Reset per-connection state.
        Subclasses can extend this if they store extra state.
        """
        self.clauses_by_id.clear()
        self.pending_scores_req = None

    def handle(self, conn: socket.socket):
        for raw in iter_null_delimited_frames(conn):
            try:
                msg = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                logging.warning(f"Bad JSON: {raw[:200]!r}")
                continue

            tag = msg.get("tag")
            if not tag:
                continue

            logging.info(f"Received tag: {tag}")

            if tag == "register_clauses":
                self._handle_register_clauses(msg)
                continue

            # In PSM: scores_req comes BEFORE server_queries_start
            if tag == "scores_req":
                self.pending_scores_req = msg
                continue

            # Now iProver opens the query window; EA must close it, then send scores.
            if tag == "server_queries_start":
                # If we don't need any queries, immediately end query phase.
                send_json_line(conn, {"tag": "server_queries_end"})

                # Then reply with scores_res for the latest scores_req
                if self.pending_scores_req is None:
                    # Nothing to score; reply empty (defensive)
                    send_json_line(conn, {"tag": "scores_res", "scores": []})
                else:
                    clause_ids = self.pending_scores_req.get("clause_ids", [])
                    scores = self.score_clause_ids(clause_ids)
                    send_json_line(conn, {"tag": "scores_res", "scores": scores})
                    self.pending_scores_req = None
                continue

            # Everything else can be ignored for the temporary agent
            continue

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
                added += 1

        logging.info(
            f"register_clauses: stored/updated {added} clauses (total={len(self.clauses_by_id)})"
        )

    def score_clause_ids(self, clause_ids: List[Any]) -> List[float]:
        """
        Override in subclasses.
        Must return one score per clause id, in the same order.
        """
        raise NotImplementedError