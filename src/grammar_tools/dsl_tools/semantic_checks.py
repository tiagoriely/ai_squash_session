# src/grammar_tools/semantic_checks.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

Step = Dict[str, Any]  # AST node

# ---------- Results ----------

@dataclass
class CheckResult:
    check: str                    # machine name, e.g. "no_drops_after_cross"
    ok: bool
    severity: str = "hard"        # "hard" or "soft"
    message: str = ""
    details: Optional[dict] = None


# ---------- Helpers over the AST ----------

def iter_actions(node: Step) -> Iterable[Step]:
    """Yield Action nodes reachable under `node`."""
    t = node.get("type")
    if t == "Action":
        yield node
    elif t == "Optional":
        yield from iter_actions(node["body"])
    elif t == "Repeat":
        yield from iter_actions(node["body"])
    elif t == "Choice":
        for opt in node["options"]:
            yield from iter_actions(opt)
    elif t == "Restart":
        return
    else:
        return

def iter_actions_linear(ast: List[Step]) -> Iterable[Tuple[int, Step]]:
    """Yield (index, Action) in the visual/linear order of the top-level steps."""
    idx = 0
    for step in ast:
        for act in iter_actions(step):
            yield (idx, act)
        idx += 1

def norm(s: str) -> str:
    return " ".join(s.lower().split())

def base_action(name: str) -> str:
    """Map concrete names to a canonical action family. Tweak as needed."""
    n = norm(name)
    if "boast" in n: return "boast"
    if "drive" in n: return "drive"      # covers "extra drive", "volley drive", etc.
    if "drop"  in n: return "drop"       # covers "counter drop", "straight drop"
    if "cross" in n: return "cross"      # covers "cross lob/kill"
    if "lob"   in n: return "lob"
    if "kill"  in n: return "kill"
    return n


# ---------- Checks you asked for ----------

def check_no_drops_after_cross(ast: List[Step], severity: str = "hard") -> CheckResult:
    """
    Enforce: once a CROSS appears in the sequence, no DROP may appear afterwards.
    Works across Optional/Repeat/Choice, in the linear order of top-level steps.
    """
    cross_seen = False
    offenders: List[str] = []

    def scan(step: Step):
        nonlocal cross_seen, offenders
        t = step.get("type")
        if t == "Action":
            name = step.get("name", "")
            fam = base_action(name)
            if fam == "cross":
                cross_seen = True
            elif cross_seen and fam == "drop":
                offenders.append(name)
        elif t == "Optional":
            scan(step["body"])
        elif t == "Repeat":
            # The body occurs BEFORE we advance to the next step, so scanning it now is fine.
            scan(step["body"])
        elif t == "Choice":
            for opt in step["options"]:
                scan(opt)
        elif t == "Restart":
            return

    for step in ast:
        scan(step)

    if offenders:
        return CheckResult(
            check="no_drops_after_cross",
            ok=False,
            severity=severity,
            message=f"Found drop actions after a cross: {sorted(set(map(norm, offenders)))}",
            details={"offenders": offenders},
        )
    return CheckResult(check="no_drops_after_cross", ok=True, severity=severity)


def check_allowed_actions(ast: List[Step],
                          allowed_actions: List[str],
                          policy: str = "exact_or_family",
                          severity: str = "hard") -> CheckResult:
    """
    Enforce: every possible Action in the AST is permitted.
    `policy`:
      - "exact": action name must be exactly listed.
      - "family": the base action family must be listed (e.g., 'drive' covers 'extra drive').
      - "exact_or_family" (default): passes if either exact name OR family is listed.
    """
    allowed_exact: Set[str] = {norm(a) for a in allowed_actions}
    allowed_families: Set[str] = {base_action(a) for a in allowed_actions}

    disallowed: List[str] = []
    for _, act in iter_actions_linear(ast):
        name = act.get("name", "")
        n = norm(name)
        fam = base_action(name)

        ok = False
        if policy == "exact":
            ok = n in allowed_exact
        elif policy == "family":
            ok = fam in allowed_families
        else:  # exact_or_family
            ok = (n in allowed_exact) or (fam in allowed_families)

        if not ok:
            disallowed.append(name)

    if disallowed:
        return CheckResult(
            check="allowed_actions",
            ok=False,
            severity=severity,
            message=f"Actions not permitted by allowed_actions: {sorted(set(map(norm, disallowed)))}",
            details={"disallowed": disallowed, "policy": policy},
        )
    return CheckResult(check="allowed_actions", ok=True, severity=severity)


def check_actor_dependencies(ast: List[Step], severity: str = "hard") -> CheckResult:
    """
    Enforce: references like 'opponent of cross' must appear only AFTER a 'cross' exists.
    (Extensible if you later add 'opponent of drive', etc.)
    """
    cross_seen = False
    violations: List[Dict[str, str]] = []

    def scan(step: Step):
        nonlocal cross_seen
        t = step.get("type")
        if t == "Action":
            name = step.get("name", "")
            actor = norm(step.get("actor", "")) if step.get("actor") else ""
            fam = base_action(name)
            if fam == "cross":
                cross_seen = True
            if actor == "opponent of cross" and not cross_seen:
                violations.append({"action": name, "actor": actor})
        elif t == "Optional":
            scan(step["body"])
        elif t == "Repeat":
            scan(step["body"])
        elif t == "Choice":
            for opt in step["options"]:
                scan(opt)

    for step in ast:
        scan(step)

    if violations:
        return CheckResult(
            check="actor_dependencies",
            ok=False,
            severity=severity,
            message="Found actor references before their antecedent 'cross'.",
            details={"violations": violations},
        )
    return CheckResult(check="actor_dependencies", ok=True, severity=severity)


def check_restart_present(ast: List[Step], severity: str = "soft") -> CheckResult:
    """Warn if no Restart appears (nice for planning loops)."""
    has_restart = any(step.get("type") == "Restart" for step in ast)
    if not has_restart:
        return CheckResult(
            check="restart_present",
            ok=False,
            severity=severity,
            message="No 'restart' directive found at end of sequence.",
        )
    return CheckResult(check="restart_present", ok=True, severity=severity)


# ---------- Runner ----------

def run_semantic_checks(ast: List[Step],
                        allowed_actions: Optional[List[str]] = None,
                        policy: str = "exact_or_family") -> List[CheckResult]:
    """
    Bundle of checks we run for squash variants.
    """
    results: List[CheckResult] = []
    results.append(check_no_drops_after_cross(ast, severity="hard"))
    results.append(check_actor_dependencies(ast, severity="hard"))
    results.append(check_restart_present(ast, severity="soft"))
    if allowed_actions:
        results.append(check_allowed_actions(ast, allowed_actions, policy=policy, severity="hard"))
    return results
