from pathlib import Path, PurePath
import json, csv, pandas as pd, docx, sys
import re


LOADERS = {
    ".docx": lambda p: "\n".join(para.text for para in docx.Document(p).paragraphs),
    ".csv":  lambda p: Path(p).read_text(encoding="utf-8", errors="replace"),
    ".xlsx": lambda p: "\n".join(
        df.to_csv(sep="\t", index=False)
        for _, df in pd.read_excel(p, sheet_name=None).items()
    ),
    ".json": lambda p: json.dumps(json.load(open(p, encoding="utf-8")), ensure_ascii=False),
}

PATTERNS = {
    "drill":            re.compile(r"\bdrill\b", re.I),
    "conditioned_game": re.compile(r"\bconditi(?:on|oned)|\bgame\b", re.I),
    "solo":             re.compile(r"\bsolo\b", re.I),
    "ghosting":         re.compile(r"\bghosting\b", re.I),
}

def classify(text: str) -> str:
    hits = [t for t, pat in PATTERNS.items() if pat.search(text)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "mix"
    return "unknown"  # default fallback


def main(raw_dir="data/raw", out_path="data/my_kb.jsonl"):
    raw_dir = Path(raw_dir)
    with open(out_path, "w", encoding="utf-8") as out:
        for idx, f in enumerate(raw_dir.rglob("*")):
            if f.suffix.lower() not in LOADERS:
                continue
            text = LOADERS[f.suffix.lower()](f)
            item = {
                "id": idx,
                "type": classify(f.stem + " " + text[:400]),
                "source": str(PurePath(f).relative_to(raw_dir)),
                "contents": text,
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Wrote", out_path)

if __name__ == "__main__":
    main(*sys.argv[1:])   # allow custom paths

