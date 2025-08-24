# rag/pipelines/generation/grammar_enforcer.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set

# If you prefer, you can import base_action from your semantic_checks.
# Here we localise it so this module has no dependency on src/.
def _norm(s: str) -> str:
    return " ".join(s.lower().split())

def _base_action(name: str) -> str:
    n = _norm(name)
    if "boast" in n: return "boast"
    if "drive" in n: return "drive"
    if "drop"  in n: return "drop"
    if "cross" in n: return "cross"
    if "lob"   in n: return "lob"
    if "kill"  in n: return "kill"
    if "serve" in n: return "serve"
    if "nick"  in n: return "nick"
    if "volley" in n: return "volley"
    return n

@dataclass
class EnforceReport:
    changed: bool
    disallowed_found: List[str]      # unique, normalised
    policy: str                      # "exact" | "family" | "exact_or_family"

def _extract_action_candidates(text: str) -> List[str]:
    """
    Very lightweight surface extractor for likely shot/action tokens.
    Catches single- and 2-word phrases like 'extra drive', 'cross lob'.
    """
    # Two-word combos we care about
    two_word = r"(?:straight|deep|hard|cross(?:-court)?|counter|volley|reverse|2-wall|3-wall|open)\s+(?:drive|drop|lob|kill|boast|cross|nick)"
    one_word = r"\b(boast|drive|drop|cross|lob|kill|serve|nick|volley)\b"
    pat = re.compile(fr"{two_word}|{one_word}", re.I)
    return [m.group(0) for m in pat.finditer(text)]

def enforce(text: str,
            allowed_actions: List[str],
            policy: str = "exact_or_family") -> Tuple[str, EnforceReport]:
    """
    Report any action tokens in `text` that aren't permitted by `allowed_actions`.
    Returns (unchanged_text, report).
    """
    allowed_exact: Set[str] = {_norm(a) for a in allowed_actions}
    allowed_fams: Set[str]  = {_base_action(a) for a in allowed_actions}

    offenders: List[str] = []
    for tok in _extract_action_candidates(text):
        n = _norm(tok)
        fam = _base_action(tok)
        ok = (
            (policy == "exact" and n in allowed_exact) or
            (policy == "family" and fam in allowed_fams) or
            (policy == "exact_or_family" and (n in allowed_exact or fam in allowed_fams))
        )
        if not ok:
            offenders.append(n)

    unique = sorted(set(offenders))
    return text, EnforceReport(
        changed=False,
        disallowed_found=unique,
        policy=policy,
    )
