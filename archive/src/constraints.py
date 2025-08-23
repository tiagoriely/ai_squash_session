# src/grammar_tools/constraints.py
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Optional

@dataclass
class Constraint:
    kind: str           # e.g., "landing_zone"
    scope: str          # e.g., "all_except_boast" | "deep_shots"
    bounce: str         # "first" | "second" | "any"
    zone: str           # "behind_t_line" | ...

_PATTERN_1 = re.compile(r"all shots excluding the boast must land behind the t-line\.?", re.I)
_PATTERN_2 = re.compile(r"second bounce behind the t-line\.?", re.I)

def parse_constraint(text: str) -> Optional[Constraint]:
    t = (text or "").strip()
    if _PATTERN_1.fullmatch(t):
        return Constraint(kind="landing_zone", scope="all_except_boast", bounce="any", zone="behind_t_line")
    if _PATTERN_2.fullmatch(t):
        return Constraint(kind="landing_zone", scope="deep_shots", bounce="second", zone="behind_t_line")
    return None
