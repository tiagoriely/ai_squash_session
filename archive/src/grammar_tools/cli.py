from __future__ import annotations
import argparse, json
from src.grammar_tools.engine.loader import load_yaml_dir
from src.grammar_tools.dsl_tools.semantic_checks import run_semantic_checks

def cmd_ast(args):
    acts = load_yaml_dir()
    for a in acts.values():
        if a.sequence_ast:
            print(f"== {a.id} :: {a.name}")
            print(json.dumps(a.sequence_ast, indent=2, ensure_ascii=False))
            print()

def cmd_validate(args):
    acts = load_yaml_dir()
    hard = soft = 0
    for a in acts.values():
        if not a.sequence_ast:
            continue
        allowed = (a.allowed_actions or
                   (a.rules or {}).get("allowed_actions") or [])
        results = run_semantic_checks(a.sequence_ast, allowed_actions=allowed, policy=args.policy)
        for r in results:
            if not r.ok and r.severity == "hard":
                hard += 1
                print(f"[HARD] {a.id} :: {r.check} :: {r.message}")
            elif not r.ok:
                soft += 1
                print(f"[SOFT] {a.id} :: {r.check} :: {r.message}")
    print(f"\nSummary → hard:{hard} soft:{soft}")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("ast", help="print parsed sequence ASTs")
    pa.set_defaults(fn=cmd_ast)

    pv = sub.add_parser("validate", help="run semantic & allowed_actions checks")
    pv.add_argument("--policy", choices=["exact","family","exact_or_family"], default="exact_or_family")
    pv.set_defaults(fn=cmd_validate)

    args = p.parse_args(); args.fn(args)

if __name__ == "__main__":
    main()
