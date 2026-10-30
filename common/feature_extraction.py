#!/usr/bin/env python3

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Set


@dataclass
class ClauseFeatures:
    num_literals: int
    num_symbols: int
    max_depth: int
    num_variables: int
    age: int
    shared_conj_symbols: int
    is_ground: int
    has_equality: int
    clause_weight: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


class ClauseFeatureExtractor:
    """
    Canonical lightweight text-based feature extractor for clause strings.

    This is still heuristic rather than a full TPTP parser, but it is designed
    to be:
      - consistent across training and inference
      - robust to nested terms
      - better aligned with clause structure than naive string splitting
    """

    TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    VAR_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*\b")
    NUMBER_RE = re.compile(r"\b\d+\b")

    RESERVED_WORDS = {
        "cnf",
        "fof",
        "tff",
        "axiom",
        "plain",
        "lemma",
        "theorem",
        "unknown",
        "conjecture",
        "negated_conjecture",
        "file",
        "include",
        "inference",
        "status",
        "thm",
        "clausifier",
        "true",
        "false",
    }

    LOGICAL_WORDS = {
        "and",
        "or",
        "not",
        "implies",
        "iff",
        "forall",
        "exists",
    }

    def _extract_variables(self, text: str) -> Set[str]:
        return set(self.VAR_RE.findall(text))

    def _extract_symbols(self, text: str) -> Set[str]:
        if not text:
            return set()

        toks = set(self.TOKEN_RE.findall(text))
        vars_ = self._extract_variables(text)
        nums = set(self.NUMBER_RE.findall(text))

        return {
            t
            for t in toks
            if t not in vars_
            and t not in nums
            and t.lower() not in self.RESERVED_WORDS
            and t.lower() not in self.LOGICAL_WORDS
        }

    def _extract_symbol_occurrences(self, text: str) -> List[str]:
        """
        Returns non-variable, non-reserved, non-logical symbol token occurrences.
        Unlike _extract_symbols(), repeated occurrences are preserved.
        """
        if not text:
            return []

        toks = self.TOKEN_RE.findall(text)
        vars_ = self._extract_variables(text)

        out: List[str] = []
        for tok in toks:
            tl = tok.lower()
            if tok in vars_:
                continue
            if tl in self.RESERVED_WORDS:
                continue
            if tl in self.LOGICAL_WORDS:
                continue
            out.append(tok)

        return out

    def _strip_outer_clause_wrapper(self, text: str) -> str:
        """
        Try to isolate the actual clause body from something like:
            cnf(name, role, (p(X) | q(X))).
        If no wrapper is matched, return the original stripped text.
        """
        s = text.strip()
        m = re.search(r",\s*\((.*)\)\s*\)\s*\.?\s*$", s)
        if m:
            return m.group(1).strip()
        return s

    def split_top_level_literals(self, clause_text: str) -> List[str]:
        """
        Split a clause into top-level literals on top-level '|'.
        This avoids incorrectly splitting inside nested terms.
        """
        s = self._strip_outer_clause_wrapper(clause_text)

        lits: List[str] = []
        buf: List[str] = []
        depth = 0
        i = 0

        while i < len(s):
            ch = s[i]

            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)

            if depth == 0 and ch == "|":
                lit = "".join(buf).strip()
                if lit:
                    lits.append(lit)
                buf = []
                i += 1
                continue

            buf.append(ch)
            i += 1

        tail = "".join(buf).strip()
        if tail:
            lits.append(tail)

        return lits if lits else ([s] if s else [])

    def count_literals(self, clause_text: str) -> int:
        return max(1, len(self.split_top_level_literals(clause_text)))

    def _term_depth(self, s: str, start: int = 0):
        i = start

        while i < len(s) and s[i].isspace():
            i += 1

        while i < len(s) and not (s[i].isalnum() or s[i] == "_"):
            i += 1

        while i < len(s) and (s[i].isalnum() or s[i] == "_"):
            i += 1

        if i < len(s) and s[i] == "(":
            i += 1
            max_child = 0

            while i < len(s):
                child_depth, i = self._term_depth(s, i)
                max_child = max(max_child, child_depth)

                while i < len(s) and s[i].isspace():
                    i += 1

                if i < len(s) and s[i] == ",":
                    i += 1
                    continue

                if i < len(s) and s[i] == ")":
                    i += 1
                    break

                if i >= len(s):
                    break

                i += 1

            return 1 + max_child, i

        return 0, i

    def estimate_max_depth(self, clause_text: str) -> int:
        """
        Estimate maximum term nesting depth rather than raw parenthesis depth.
        """
        text = self._strip_outer_clause_wrapper(clause_text)

        best = 0
        i = 0
        while i < len(text):
            if text[i].isalnum() or text[i] == "_":
                d, j = self._term_depth(text, i)
                best = max(best, d)
                i = max(i + 1, j)
            else:
                i += 1

        return best

    def compute_clause_weight(self, clause_text: str) -> int:
        """
        Clause weight intended as a rough structural size measure.

        We combine:
          - number of unique non-variable symbols
          - number of variables
          - extra literals beyond the first
          - parentheses count as a proxy for structural complexity
        """
        text = clause_text.strip()
        literals = self.split_top_level_literals(text)
        variables = self._extract_variables(text)
        symbols = self._extract_symbols(text)

        return (
            len(symbols)
            + len(variables)
            + max(0, len(literals) - 1)
            + text.count("(")
            + text.count(")")
        )

    def extract_features(
        self,
        clause_text: str,
        age: int,
        conjecture_symbols: Set[str],
    ) -> ClauseFeatures:
        text = clause_text.strip()

        literals = self.split_top_level_literals(text)
        num_literals = max(1, len(literals))

        variables = self._extract_variables(text)
        symbols = self._extract_symbols(text)

        num_symbols = len(symbols)
        max_depth = self.estimate_max_depth(text)
        num_variables = len(variables)
        shared_conj_symbols = len(symbols.intersection(conjecture_symbols))
        is_ground = 1 if num_variables == 0 else 0
        has_equality = 1 if "=" in text else 0
        clause_weight = self.compute_clause_weight(text)

        return ClauseFeatures(
            num_literals=num_literals,
            num_symbols=num_symbols,
            max_depth=max_depth,
            num_variables=num_variables,
            age=age,
            shared_conj_symbols=shared_conj_symbols,
            is_ground=is_ground,
            has_equality=has_equality,
            clause_weight=clause_weight,
        )