# rag/pipelines/generation/run_generation_grammar.py
import argparse
import json
import random
import re
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from flashrag.retriever.retriever import DenseRetriever

# ---- Grammar plumbing -------------------------------------------------------
from rag.pipelines.generation.grammar_constraints_integration import (
    ConstraintsMode,
    load_family_constraints,
    build_constraints_block,
    load_structure,
    build_structure_block,
    validate_structure,
    load_archetype,
    build_archetype_block,
    validate_archetype,
)
from rag.pipelines.generation.grammar_enforcer import enforce as enforce_grammar
# -----------------------------------------------------------------------------


load_dotenv()

# Patterns for session type inference (only for picking base prompt template)
SESSION_TYPES_PATTERNS = {
    "drill": re.compile(r"\bdrill\b", re.I),
    "conditioned_game": re.compile(r"\bconditi(?:on|oned)|game\b", re.I),
    "solo": re.compile(r"\bsolo\b", re.I),
    "ghosting": re.compile(r"\bghosting\b", re.I),
    "mix": re.compile(
        r"\b(drill|conditi(?:on|oned)|game|solo|ghosting)\b.*\b(drill|conditi(?:on|oned)|game|solo|ghosting)\b",
        re.I,
    ),
}


def load_cfg(path: str | Path) -> dict:
    data = yaml.safe_load(open(path, "r", encoding="utf-8"))
    return data or {}


def build_context(docs, max_tokens=3000):
    parts, total = [], 0
    for doc in docs:
        tokens = len(doc["contents"].split())  # rough estimate
        if total + tokens > max_tokens:
            break
        parts.append(doc["contents"])
        total += tokens
    return "\n---\n".join(parts)


def generate_answer(prompt: str, model="gpt-4o") -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful squash training assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def infer_type(query: str) -> str:
    for key, pattern in SESSION_TYPES_PATTERNS.items():
        if pattern.search(query):
            return key
    return "conditioned_game"


def resolve_session_type(user_type: str, query: str) -> str:
    """
    If user passed 'auto', infer from the query; otherwise return the user choice.
    """
    if (user_type or "").lower() == "auto":
        return infer_type(query)
    return user_type


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAG generation with grammar constraints")
    ap.add_argument("cfg", help="YAML config for retriever")
    ap.add_argument(
        "--query",
        required=False,
        default=None,
        help="If omitted, sample a random prompt from data/squash_session_queries_prompts.json",
    )
    ap.add_argument("--model", default="gpt-4o")

    # Session type selection (auto = infer from query, else pick a prompt file explicitly)
    ap.add_argument(
        "--type",
        choices=["auto", "drill", "conditioned_game", "solo", "ghosting", "mix"],
        default="auto",
        help="Which base prompt template to use. 'auto' infers from the query.",
    )

    # Grammar controls
    ap.add_argument(
        "--constraints",
        choices=["soft", "hard", "hybrid"],
        default="hybrid",
        help="Grammar application mode",
    )
    ap.add_argument(
        "--policy",
        choices=["exact", "family", "exact_or_family"],
        default="exact_or_family",
        help="Allowed-actions policy for post-hoc enforcement",
    )
    ap.add_argument(
        "--family-id",
        default="squash.family.pattern.diagonal.boast_cross_drive",
        help="Family id from grammar/sports/squash/exercises/*.yaml",
    )
    ap.add_argument(
        "--structure-id",
        default="conditioned_game_v1",
        help="File stem under grammar/sports/squash/session_structures (or dotted id)",
    )
    ap.add_argument(
        "--archetype-id",
        default="progressive_family",
        help="File stem under grammar/sports/squash/session_archetypes (or dotted id)",
    )
    args = ap.parse_args()

    # ---------------- retrieval ----------------
    cfg = load_cfg(args.cfg)
    cfg.update({"use_reranker": True})
    retriever = DenseRetriever(cfg)

    # pick or sample query
    if args.query:
        query = args.query
        print(f"▶️ Using provided query: '{query}'")
    else:
        prompts_path = Path("data/squash_session_queries_prompts.json")
        if not prompts_path.exists():
            raise SystemExit(
                f"Prompts file missing at {prompts_path}. "
                "Run: python evaluation/generation/create_prompts_for_generation.py"
            )
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        if not prompts:
            raise SystemExit("Prompts file is empty.")
        query = random.choice(prompts)
        print(f"🎲 Random prompt: '{query}'")

    t0 = time.perf_counter()
    docs, _ = retriever.search(query, return_score=True)
    dt = 1000 * (time.perf_counter() - t0)
    print(f"\n⏱  {dt:.1f} ms   |   top-k={len(docs)}\n")
    for i, d in enumerate(docs, 1):
        print(f"{i:2d}. id={d['id']:<4} source={d['source']}")

    # ---------------- base prompt ----------------
    context = build_context(docs)
    session_type = resolve_session_type(args.type, query)
    prompt_path = Path("prompts/rag") / f"session_{session_type}.txt"
    base_template = prompt_path.read_text(encoding="utf-8")

    # ---------------- grammar: load + prompt blocks ----------------
    mode = ConstraintsMode(args.constraints)

    # family -> constraints
    fam = load_family_constraints(args.family_id)
    constraints_block = build_constraints_block(fam, mode=mode, include_sequences=True)

    # structure
    struct = load_structure(args.structure_id)
    structure_block = build_structure_block(struct)

    # archetype
    arch = load_archetype(args.archetype_id)
    archetype_block = build_archetype_block(arch)

    # --- Build final prompt (supports inline placeholders or append style) ---
    fmt_vars = {
        "question": query,
        "context": context,
        "STRUCTURE_SPEC": structure_block,
        "ARCHETYPE_RULES": archetype_block,
        "CONSTRAINTS": constraints_block,  # you can use {CONSTRAINTS} in your template
    }
    try:
        # If the template has {STRUCTURE_SPEC}/{ARCHETYPE_RULES}/{CONSTRAINTS}, this fills them.
        filled_prompt = base_template.format(**fmt_vars)
    except KeyError:
        # Template doesn’t define grammar placeholders → append blocks after filling q/context.
        filled_prompt = (
            base_template.format(question=query, context=context)
            + "\n\n"
            + structure_block
            + "\n\n"
            + archetype_block
            + "\n\n"
            + constraints_block
        )

    # ---------------- generate ----------------
    print("\n=== Answer ===\n")
    answer = generate_answer(filled_prompt, model=args.model)

    # ---------------- post-hoc grammar checks (hard/hybrid) ----------------
    enforcer_applied = False
    disallowed = []
    violations_structure = []
    violations_archetype = []

    if mode in (ConstraintsMode.HARD, ConstraintsMode.HYBRID):
        allowed_union = fam.allowed_actions_union()
        answer, report = enforce_grammar(answer, allowed_union, policy=args.policy)
        enforcer_applied = getattr(report, "changed", False)
        disallowed = getattr(report, "disallowed_found", [])

        # soft text validators
        violations_structure = validate_structure(answer, struct)
        violations_archetype = validate_archetype(answer, arch)

        # Optional: print a brief audit summary
        if enforcer_applied or disallowed or violations_structure or violations_archetype:
            print("\n--- Grammar audit ---")
            if enforcer_applied:
                print("• Enforcer modified output to remove disallowed actions.")
            if disallowed:
                print(f"• Disallowed actions encountered: {sorted(set(disallowed))}")
            for v in violations_structure:
                print(f"• STRUCTURE: {v}")
            for v in violations_archetype:
                print(f"• ARCHETYPE: {v}")
            print("--- end audit ---\n")

    print(answer)

    # ---------------- persist row for RAGAS / analysis ----------------
    out_path = Path("data/processed/eval_dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    current = []
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            current = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(current, list):
                current = []
        except Exception:
            current = []

    next_case_id = 1 + max(
        [int(r.get("case_id", 0)) for r in current if str(r.get("case_id", "0")).isdigit()] or [0]
    )

    retrieved_documents_info = [{"id": d["id"], "source": d["source"]} for d in docs]
    row = {
        "case_id": next_case_id,
        "question": query,
        "answer": answer,
        "contexts": [d["contents"] for d in docs],
        "retrieved_documents_info": retrieved_documents_info,
        # grammar audit
        "constraints_mode": mode.value,
        "policy": args.policy,
        "family_id": fam.family_id,
        "structure_id": struct.get("_id"),
        "archetype_id": arch.get("_id"),
        "enforcer_applied": enforcer_applied,
        "disallowed_found": disallowed,
        "violations_structure": violations_structure,
        "violations_archetype": violations_archetype,
        # traceability
        "prompt_files": {
            "base": str(prompt_path),
            "structure_yaml": struct.get("_file"),
            "archetype_yaml": arch.get("_file"),
        },
    }
    current.append(row)
    out_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 1 row added to {out_path} with Case ID: {next_case_id}")
