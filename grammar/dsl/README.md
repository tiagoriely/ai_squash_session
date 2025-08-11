
---

### `grammar/dsl/README.md`

```markdown
# Sequence DSL — How to use

This folder defines the **executable grammar** (`sequence.peg`) and the
**human-facing spec** (`spec.ebnf.md`) for the small language used in
`rules.sequence` inside your exercise YAMLs.

## What this is

- A compact language for linear shot **sequences** with:
  - **Actions** (e.g., `boast (A)`)
  - **Choices** `( X | Y )`
  - **Repeat** `( … )*`
  - **Optional** `optional: X` or `optional: ( X | Y )`
  - **Restart** `restart …`
  - **Arrows** `→` or `->` between steps

- Deterministic (PEG = prioritized choice), scannerless (no separate lexer).

## Where it’s used

- `src/grammar_tools/parser.py` loads **`sequence.peg`** and parses each
  `rules.sequence` **line** into an AST.
- `src/grammar_tools/semantic_checks.py` uses that AST for domain rules
  (e.g., “optional extra drive only once”, time budgets).
- `src/grammar_tools/features.py` converts ASTs to **features** for retrieval.
- `src/grammar_tools/constrained_decode.py` can enforce the grammar at generation
  time or run generate→parse→repair.

## Authoring rules in YAML

- Keep `rules.sequence` as either:
  - a single string (simple cases), or
  - a **list of strings** (recommended for complex cases), one “program” per line.
- Examples:
  ```yaml
  rules:
    sequence: "boast → drive → cross"
