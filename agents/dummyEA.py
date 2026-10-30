#!/usr/bin/env python3

import argparse
from typing import List, Any
from baseEA import BaseEA


class DummyEA(BaseEA):
    def score_clause_ids(self, clause_ids: List[Any]) -> List[float]:
        return [0.0 for _ in clause_ids]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    agent = DummyEA(host=args.host, port=args.port)
    agent.run()