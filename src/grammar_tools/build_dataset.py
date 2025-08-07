from pathlib import Path
import json, random, uuid, tqdm

from .planner import build_session              # ← uses your YAML grammar
from rag.pipelines.generation.generator import (
    generate_session,                           # ← surface realisation LLM prompt
)
from rag.pipelines.generation.grammar_enforcer import enforce

OUT = Path("data/generated_sessions.jsonl")
PROMPTS = [
    "Intermediate club player, wants to improve length control (60 min)",
    "Junior elite, attacking skills focus (45 min)",
    "Adult beginner, front-court confidence (60 min)",
    "National squad, pro-level long drills (90 min)",
    # …add as many seed questions as you like
]

def main(n_per_prompt: int = 20, seed: int = 42):
    random.seed(seed)
    out = OUT.open("w")

    for q in tqdm.tqdm(PROMPTS, desc="prompts"):
        for _ in range(n_per_prompt):
            # Decide the requested duration from the question string
            minutes = 90 if "90" in q else 60 if "60" in q else 45
            warm, sched = build_session(minutes)

            # Produce the full text answer
            draft = generate_session(schedule=sched, warmup=warm, question=q)
            final = enforce(draft, valid_names={a.name for a in sched + [warm]})

            json.dump(
                {
                    "id": str(uuid.uuid4()),
                    "question": q,
                    "session_text": final,
                    "grammar_only": True,
                },
                out,
            )
            out.write("\n")

    out.close()
    print(f"✅  Saved {len(PROMPTS)*n_per_prompt} sessions to {OUT}")

if __name__ == "__main__":
    main()
