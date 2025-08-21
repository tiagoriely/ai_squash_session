from src.grammar_tools.dsl_tools.semantic_checks import (
    run_semantic_checks, check_allowed_actions
)

def test_allowed_actions_family_policy():
    ast = [
        {"type": "Action", "name": "boast", "actor": "A"},
        {"type": "Action", "name": "straight drive", "actor": "B"},
    ]
    r = check_allowed_actions(ast, ["drive", "boast"], policy="family")
    assert r.ok

def test_semantic_bundle_has_soft_restart_warning():
    ast = [{"type": "Action", "name": "boast"}]
    results = run_semantic_checks(ast, allowed_actions=["boast"])
    assert any(not r.ok and r.severity == "soft" for r in results)  # restart warning
