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

# Define patterns for all fields

SESSION_TYPE_PATTERNS = {
    "drill":            re.compile(r"\bdrill\b", re.I),
    "conditioned_game": re.compile(r"\bconditi(?:on|oned)\b|\bgame\b", re.I),
    "solo":             re.compile(r"\bsolo\b", re.I),
    "ghosting":         re.compile(r"\bghosting\b", re.I),
}

PARTICIPANTS_PATTERNS = {
    "1": re.compile(r"\b1(?:\s*player)?s?\b|\bsolo\b|\bone\s*player\b", re.I),
    "2": re.compile(r"\b2(?:\s*player)?s?\b|\btwo\s*players\b", re.I),
    "3": re.compile(r"\b3(?:\s*player)?s?\b|\bthree\s*players\b", re.I),
    "4": re.compile(r"\b4(?:\s*player)?s?\b|\bfour\s*players\b", re.I),
}

SQUASH_LEVEL_PATTERNS = {
    "beginner":     re.compile(r"\bbeginner\b", re.I),
    "intermediate": re.compile(r"\bintermediate\b", re.I),
    "advanced":     re.compile(r"\badvanced\b", re.I),
    "professional": re.compile(r"\bprofessional\b", re.I),
}

FITNESS_PATTERNS = {
    "low":            re.compile(r"\blow\s*fitness\b|\blow\s*intensity\b", re.I),
    "medium":         re.compile(r"\bmedium\s*fitness\b|\bmedium\s*intensity\b", re.I),
    "high":           re.compile(r"\bhigh\s*fitness\b|\bhigh\s*intensity\b", re.I),
    "extremely_high": re.compile(r"\bextremely\s*high\s*fitness\b|\bextremely\s*high\s*intensity\b", re.I),
    "intermediate":   re.compile(r"\bintermediate\s*fitness\b|\bintermediate\s*intensity\b", re.I),
}

# New INTENSITY_PATTERNS dictionary
INTENSITY_PATTERNS = {
    "low":    re.compile(r"\blow\s*intensity\b", re.I),
    "medium": re.compile(r"\bmedium\s*intensity\b", re.I),
    "high":   re.compile(r"\bhigh\s*intensity\b", re.I),
}

SHOT_PATTERNS = {
    "drive": re.compile(r"\bdrive\b", re.I),
    "cross": re.compile(r"\bcross\b", re.I),
    "lob":   re.compile(r"\blob\b", re.I),
    "drop":  re.compile(r"\bdrop\b", re.I),
    "boast": re.compile(r"\bboast\b", re.I),
    "volley": re.compile(r"\bvolley\b", re.I),
    "serve": re.compile(r"\bserve\b", re.I),
}

SHOT_SIDE_PATTERNS = {
    "forehand": re.compile(r"\bforehand\b|\bFH\b", re.I),
    "backhand": re.compile(r"\bbackhand\b|\bBH\b", re.I),
}

DURATION_PATTERNS = {
    "10": re.compile(r"\b10\s*min(?:ute)?s?\b", re.I),
    "15": re.compile(r"\b15\s*min(?:ute)?s?\b", re.I),
    "30": re.compile(r"\b30\s*min(?:ute)?s?\b", re.I),
    "45": re.compile(r"\b45\s*min(?:ute)?s?\b", re.I),
    "60": re.compile(r"\b60\s*min(?:ute)?s?\b", re.I),
    "90": re.compile(r"\b90\s*min(?:ute)?s?\b", re.I),
}


# Generic extraction function
def extract_field_values(text: str, patterns: dict, allow_multiple: bool = False, return_first_match: bool = False):
    """
    Extracts values for a given field based on provided patterns.

    Args:
        text (str): The text to search within.
        patterns (dict): A dictionary where keys are field values (e.g., "advanced")
                         and values are compiled regex patterns.
        allow_multiple (bool): If True, returns a list of all matched values.
                               If False, returns a single value ("mix" if multiple, "unknown" if none).
        return_first_match (bool): If True and allow_multiple is False, returns the first
                                   matched value instead of "mix" for multiple matches.
                                   Useful for fields like 'duration' or 'participants' where
                                   only one value is expected.
    Returns:
        str or list: The extracted value(s), "mix", "unknown", or an empty list.
    """
    matched_values = []
    for value, pattern in patterns.items():
        if pattern.search(text):
            matched_values.append(value)

    if allow_multiple:
        return matched_values if matched_values else []
    else:
        if len(matched_values) == 1:
            return matched_values[0]
        if len(matched_values) > 1:
            if return_first_match:
                return matched_values[0]
            return "mix"
        return "unknown"


def main(raw_dir="data/raw", out_path="data/my_kb.jsonl"):
    raw_dir = Path(raw_dir)
    with open(out_path, "w", encoding="utf-8") as out:
        for idx, f in enumerate(raw_dir.rglob("*")):
            print(f"Processing file: {f}")
            if f.suffix.lower() not in LOADERS:
                continue

            text = ""
            try:
                text = LOADERS[f.suffix.lower()](f)
            except Exception as e:
                print(f"Error loading file {f}: {e}")
                continue

            session_data = {}
            lines = text.split('\n')

            # Updated key_value_pattern to include 'intensity'
            key_value_pattern = re.compile(r"^(type|participants|squashlevel|fitness|duration|shots|shotSide|intensity):\s*(.+)$",
                                           re.IGNORECASE)

            for line in lines:
                match = key_value_pattern.match(line.strip())
                if match:
                    key = match.group(1).lower()
                    value = match.group(2).strip()

                    if key == "shots" or key == "shotside":
                        session_data[key] = [v.strip() for v in re.split(r'\s+and/or\s+|\s+and\s+|,', value) if
                                             v.strip()]
                    else:
                        session_data[key] = value

            search_text = f.stem + " " + text

            item = {
                "id": idx,
                "source": str(PurePath(f).relative_to(raw_dir)),
                "type": session_data.get("type") or extract_field_values(search_text, SESSION_TYPE_PATTERNS),
                "participants": session_data.get("participants") or extract_field_values(search_text,
                                                                                         PARTICIPANTS_PATTERNS,
                                                                                         return_first_match=True),
                "duration": session_data.get("duration") or extract_field_values(search_text, DURATION_PATTERNS,
                                                                                 return_first_match=True),
                "squashLevel": session_data.get("squashlevel") or extract_field_values(search_text,
                                                                                       SQUASH_LEVEL_PATTERNS),
                "fitness": session_data.get("fitness") or extract_field_values(search_text, FITNESS_PATTERNS),
                # Added new 'intensity' field
                "intensity": session_data.get("intensity") or extract_field_values(search_text, INTENSITY_PATTERNS),
                "shots": session_data.get("shots", []) or extract_field_values(search_text, SHOT_PATTERNS,
                                                                               allow_multiple=True),
                "shotSide": session_data.get("shotside", []) or extract_field_values(search_text, SHOT_SIDE_PATTERNS,
                                                                                     allow_multiple=True),
                "contents": text,
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:])