# rag/pipelines/generation/run_generation_grammar_with_random_prompt.py
import argparse, yaml, re, json, random
from pathlib import Path
from flashrag.retriever.retriever import DenseRetriever
from openai import OpenAI
from dotenv import load_dotenv

from archive.rag.generation.grammar_constraints_integration import (
    ConstraintsMode,
    load_family_constraints, build_constraints_block,
    load_structure, build_structure_block, validate_structure,
    load_archetype, build_archetype_block, validate_archetype,
    load_session_type, build_session_type_block, validate_session_type,
)
from archive.rag.generation.grammar_enforcer import enforce as enforce_grammar

load_dotenv()

SESSION_TYPES_PATTERNS = {
    "drill":            re.compile(r"\bdrill\b", re.I),
    "conditioned_game": re.compile(r"\bconditi(?:on|oned)|game\b", re.I),
    "solo":             re.compile(r"\bsolo\b", re.I),
    "ghosting":         re.compile(r"\bghosting\b", re.I),
    "mix":              re.compile(r"\b(drill|conditi(?:on|oned)|game|solo|ghosting)\b.*\b(drill|conditi(?:on|oned)|game|solo|ghosting)\b", re.I),
}

def load_cfg(path: str | Path) -> dict:
    data = yaml.safe_load(open(path, "r", encoding="utf-8"))
    return data or {}

def build_context(docs, max_tokens=3000):
    out, total = [], 0
    for d in docs:
        n = len(d["contents"].split())
        if total + n > max_tokens:
            break
        out.append(d["contents"])
        total += n
    return "\n---\n".join(out)

def infer_type_from_query(q: str) -> str:
    for k, pat in SESSION_TYPES_PATTERNS.items():
        if pat.search(q):
            return k
    return "conditioned_game"

def _pick_base_prompt_file(session_type_id: str) -> Path:
    p = Path("prompts/rag") / f"session_{session_type_id}.txt"
    return p if p.exists() else Path("prompts/rag") / "session_conditioned_game.txt"

def generate_answer(prompt: str, model="gpt-4o") -> str:
    client = OpenAI()
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful squash training assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return r.choices[0].message.content


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAG + Grammar (random prompt)")
    ap.add_argument("cfg", help="YAML config for retriever")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--constraints", choices=["soft", "hard", "hybrid"], default="hybrid")
    ap.add_argument("--policy", choices=["exact", "family", "exact_or_family"], default="exact_or_family")
    ap.add_argument("--family-id", default="squash.family.pattern.diagonal.boast_cross_drive")
    ap.add_argument("--structure-id", default="conditioned_game_v1")
    ap.add_argument("--archetype-id", default="progressive_family")
    ap.add_argument("--session-type", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    cfg.update({"use_reranker": True})
    retriever = DenseRetriever(cfg)

    prompts_path = Path("data/squash_session_queries_prompts.json")
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    query = random.choice(prompts)
    print(f"🎲 Random prompt: '{query}'")

    docs, _ = retriever.search(query, return_score=True)
    print(f"\n⏱  Retrieved top-k={len(docs)}\n")
    for i, d in enumerate(docs, 1):
        print(f"{i:2d}. id={d['id']:<4} source={d['source']}")

    context = build_context(docs)
    session_type_id = (args.session_type or "").strip().lower() or infer_type_from_query(query)
    base_prompt_path = _pick_base_prompt_file(session_type_id)
    base_template = base_prompt_path.read_text(encoding="utf-8")

    mode = ConstraintsMode(args.constraints)
    fam = load_family_constraints(args.family_id)
    struct = load_structure(args.structure_id)
    arch = load_archetype(args.archetype_id)
    st = load_session_type(session_type_id)

    prompt = (
        base_template
        + "\n\n" + build_structure_block(struct)
        + "\n\n" + build_archetype_block(arch)
        + "\n\n" + build_session_type_block(st)
        + "\n\n" + build_constraints_block(fam, mode=mode, include_sequences=True)
    ).format(question=query, context=context)

    print("\n=== Answer ===\n")
    answer = generate_answer(prompt, model=args.model)

    enforcer_applied = False
    disallowed = []
    v_struct = v_arch = v_type = []
    if mode in (ConstraintsMode.HARD, ConstraintsMode.HYBRID):
        allowed_union = fam.allowed_actions_union()
        answer, report = enforce_grammar(answer, allowed_union, policy=args.policy)
        enforcer_applied = getattr(report, "changed", False)
        disallowed = getattr(report, "disallowed_found", [])
        v_struct = validate_structure(answer, struct)
        v_arch = validate_archetype(answer, arch)
        v_type = validate_session_type(answer, st)

        if enforcer_applied or disallowed or v_struct or v_arch or v_type:
            print("\n--- Grammar audit ---")
            if enforcer_applied: print("• Enforcer modified output to remove disallowed actions.")
            if disallowed: print(f"• Disallowed actions encountered: {sorted(set(disallowed))}")
            for v in v_struct: print(f"• STRUCTURE: {v}")
            for v in v_arch: print(f"• ARCHETYPE: {v}")
            for v in v_type: print(f"• TYPE: {v}")
            print("--- end audit ---\n")

    print(answer)

    out_path = Path("data/processed/eval_dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
        if not isinstance(current, list):
            current = []
    except Exception:
        current = []

    next_case_id = 1 + max([int(r.get("case_id", 0)) for r in current if str(r.get("case_id", "0")).isdigit()] or [0])
    row = {
        "case_id": next_case_id,
        "question": query,
        "answer": answer,
        "contexts": [d["contents"] for d in docs],
        "retrieved_documents_info": [{"id": d["id"], "source": d["source"]} for d in docs],
        "constraints_mode": mode.value,
        "policy": args.policy,
        "family_id": fam.family_id,
        "structure_id": struct.get("_id"),
        "archetype_id": arch.get("_id"),
        "session_type_id": st.get("_id"),
        "enforcer_applied": enforcer_applied,
        "disallowed_found": disallowed,
        "violations_structure": v_struct,
        "violations_archetype": v_arch,
        "violations_session_type": v_type,
        "prompt_files": {
            "base": str(base_prompt_path),
            "structure_yaml": struct.get("_file"),
            "archetype_yaml": arch.get("_file"),
            "session_types_yaml": st.get("_file"),
        },
    }
    current.append(row)
    out_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 1 row added to {out_path} with Case ID: {next_case_id}")
