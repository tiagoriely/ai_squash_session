#rag/parsers/user_query_parser.py
from __future__ import annotations

import re
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Optional, Set, Dict
from nltk.stem import PorterStemmer

# Load configuration from the central YAML file instead of a Python file.
_CONFIG_PATH = Path(__file__).parent.parent.parent / 'configs' / 'retrieval' / 'raw_squash_field_retrieval_config.yaml'

def _load_config_from_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

_config_data = _load_config_from_yaml(_CONFIG_PATH)

# Assign the loaded data to the global variables the rest of the script expects
SYNONYM_MAP = _config_data.get("SYNONYM_MAP", {})
SPECIFIC_MAPS = _config_data.get("SPECIFIC_MAPS", {})

# ---------- small helpers ----------
_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
}
# phrases that imply solo = 1 participant
_SOLO_PHRASES = {
    "solo", "by myself", "on my own", "alone on court", "no partner available",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _re_word_boundary(phrase: str) -> str:
    # add \b around words; keep spaces inside phrase as \s+
    p = re.escape(phrase.lower())
    p = p.replace(r"\ ", r"\s+")
    return rf"\b{p}\b"


def _to_number(token: str) -> Optional[int]:
    """Convert 'two' -> 2 or '2' -> 2; return None if not numeric/known word."""
    token = token.lower().strip()
    if token in _WORD_NUM:
        return _WORD_NUM[token]
    m = re.fullmatch(r"\d{1,3}", token)
    if m:
        return int(m.group(0))
    return None


# --- plural/gerund tolerant helpers ------------------------------------------
def _plural_variants(word: str) -> Set[str]:
    """Return singular/plural variants for the last token (simple English rules)."""
    v = {word}
    # plain 's'
    v.add(word + "s")
    # y -> ies (but not if vowel before y)
    if word.endswith("y") and not re.search(r"[aeiou]y$", word):
        v.add(word[:-1] + "ies")
    return v


def _plural_aware_pattern(phrase: str) -> re.Pattern:
    """
    Build a hyphen/space/slash tolerant regex for a phrase, allowing plural on the LAST token.
    Example: 'conditioned game' matches 'conditioned-games' and 'conditioned games' and 'conditioned game(s)'.
    """
    parts = phrase.lower().split()
    if not parts:
        return re.compile(r"(?!x)x")

    *before, last = parts
    sep = r"(?:[-/\s]+)"
    head = sep.join(map(re.escape, before)) if before else ""
    last_group = "|".join(map(re.escape, _plural_variants(last)))
    body = last_group if not head else rf"{head}{sep}(?:{last_group})"
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


def _normalize_words(text: str, replacements: list[tuple[str, str]]) -> str:
    """Generic normalizer: apply a list of (pattern, replacement) with IGNORECASE."""
    t = text
    for pat, repl in replacements:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    return t


# ---------- extracting user prompt ----------
def parse_user_prompt(text: str, allowed_durations: list[int] | None = None) -> dict:
    """
    High-level extractor that aggregates all user-desire fields from free text.
    Returns only fields that were confidently found.

    Keys match FIELD_SCORING_CONFIG:
      - "type": str  ("drill" | "conditioned game" | "solo practice" | "ghosting" | "mix")
      - "participants": int
      - "squashLevel": str ("beginner" | "intermediate" | "advanced")
      - "intensity": str ("low" | "medium" | "high")
      - "duration": int (minutes)
      - "shots": list[str]  (union of general + specific)
      - "shotSide": list[str] (subset of ["forehand","backhand","both"])
      - "movement": list[str]
    """
    out: dict = {}

    dur = parse_duration(text, allowed_durations=allowed_durations)
    if dur is not None:
        out["duration"] = dur

    part = parse_participants(text)
    if part is not None:
        out["participants"] = part

    stype = parse_type(text)
    if stype:
        out["type"] = stype

    level = parse_squash_level(text)
    if level:
        out["squashLevel"] = level

    inten = parse_intensity(text)
    if inten:
        out["intensity"] = inten

    shots = parse_shots(text)
    if shots:
        combo = set(shots.get("shots_general", set())) | set(shots.get("shots_specific", set()))
        if combo:
            out["shots"] = sorted(combo)

    sides = parse_shot_side(text)
    if sides:
        out["shotSide"] = sides

    moves = parse_movement(text)
    if moves:
        out["movement"] = moves

    return out


# ---------- duration ----------
def parse_duration(text: str, allowed_durations: list[int] | None = None) -> Optional[int]:
    """
    Returns minutes as int if a plausible total session duration is found, else None.

    Supports:
      - '60-min', '45-minute', '30min', '90 minutes', 'Need ~60 min total'
      - '1h30', '1 h 30', '1h'
      - 'an hour', 'about an hour', 'hour and a half', 'half an hour'
      - 'hour-long' (treated as 60)

    If multiple durations are present, returns the **largest** (assumed total session).

    If `allowed_durations` is provided, snaps the detected value to the nearest allowed
    duration (on ties, prefers the larger value).
    """
    t = _norm(text)
    minutes_found: list[int] = []

    # 1) h+m combos e.g. 1h30, 1 h 30
    for m in re.finditer(r"\b(\d{1,2})\s*h\s*(\d{1,2})\b", t):
        h, mm = int(m.group(1)), int(m.group(2))
        minutes_found.append(h * 60 + mm)
    for m in re.finditer(r"\b(\d{1,2})h(\d{1,2})\b", t):
        h, mm = int(m.group(1)), int(m.group(2))
        minutes_found.append(h * 60 + mm)

    # 2) explicit minutes, allowing hyphen/space/tilde before unit
    for m in re.finditer(r"\b(\d{1,3})\s*[-\s~]?(?:m|min|mins|minute|minutes)\b", t):
        minutes_found.append(int(m.group(1)))

    # 3) hours as integer/decimal (e.g., 1h, 1.5 hours)
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", t):
        hours = float(m.group(1))
        minutes_found.append(int(round(hours * 60)))

    # 3b) hour-long
    if re.search(r"\bhour[-\s]*long\b", t):
        minutes_found.append(60)

    # 4) common phrases
    if re.search(_re_word_boundary("hour and a half"), t) or re.search(_re_word_boundary("an hour and a half"), t):
        minutes_found.append(90)
    if re.search(_re_word_boundary("half an hour"), t):
        minutes_found.append(30)
    # Put this after "hour and a half" so it doesn't override with 60
    if re.search(_re_word_boundary("about an hour"), t) or re.search(_re_word_boundary("an hour"), t):
        minutes_found.append(60)

    if not minutes_found:
        return None

    raw_minutes = max(minutes_found)

    if not allowed_durations:
        return raw_minutes

    # Snap to nearest allowed (tie -> choose the larger)
    allowed = sorted(set(int(x) for x in allowed_durations))
    best = min(allowed, key=lambda a: (abs(a - raw_minutes), -a))
    return best


# ---------- participants ----------
def parse_participants(text: str) -> Optional[int]:
    """
    Returns number of players if found; prefers the largest unambiguous match tied to 'player(s)'.
    Maps solo-phrases → 1. Handles:
      - '2 players', '3-player', 'four-player'
      - 'for two players', 'best of five games, two players'
      - 'two or three players'
      - 'with a friend' / 'with another player' → 2
    """
    t = _norm(text)

    # check for just a plain number, for the dialogue manager
    if t.isdigit():
        num = int(t)
        if 1 <= num <= 4:  # Or whatever range is reasonable
            return num

    # Solo phrases
    for ph in _SOLO_PHRASES:
        if re.search(_re_word_boundary(ph), t):
            return 1

    candidates: Set[int] = set()
    words = "|".join(map(re.escape, _WORD_NUM.keys()))

    # Numeric before 'players', allowing for adjectives in between (e.g., "2 advanced players")
    for m in re.finditer(r"\b(\d{1,2})(?:\s+\w+)*?\s*players?\b", t):
        candidates.add(int(m.group(1)))

    # Word before 'players', allowing for adjectives (e.g., "two intermediate players")
    for m in re.finditer(rf"\b({words})(?:\s+\w+)*?\s*players?\b", t):
        candidates.add(_WORD_NUM[m.group(1)])

    # 'for two players' / 'for 2 advanced players'
    for m in re.finditer(rf"\bfor\s+({words}|\d{{1,2}})(?:\s+\w+)*?\s*players?\b", t):
        token = m.group(1)
        n = _to_number(token)
        if n is not None:
            candidates.add(n)

    # Hyphenated word: 'four-player'
    for m in re.finditer(rf"\b({words})\s*-\s*players?\b", t):
        candidates.add(_WORD_NUM[m.group(1)])

    # 'for two players' / 'for 2 players'
    for m in re.finditer(rf"\bfor\s+({words}|\d{{1,2}})\s*players?\b", t):
        token = m.group(1)
        n = _to_number(token)
        if n is not None:
            candidates.add(n)

    # Conjoined choices: 'two or three players', '2/3 players'
    for m in re.finditer(rf"\b({words}|\d{{1,2}})\s*(?:or|to|/)\s*({words}|\d{{1,2}})\s*players?\b", t):
        a, b = m.group(1), m.group(2)
        na, nb = _to_number(a), _to_number(b)
        if na is not None:
            candidates.add(na)
        if nb is not None:
            candidates.add(nb)

    # Partner phrasing → 2 (singular/plural 'friend(s)/player(s)')
    if re.search(r"\bwith\s+(?:a|another)?\s*(?:friend|friends|player|players)\b", t):
        candidates.add(2)

    return max(candidates) if candidates else None


# ---------- type ----------
def parse_type(text: str) -> Optional[str]:
    """
    Returns canonical session type based on SYNONYM_MAP['type'].
    If two or more distinct types appear, returns 'mix'.
    Plural tolerant: 'drills', 'games', 'practices'.
    """
    t = _norm(text)
    type_map = SYNONYM_MAP.get("type", {}) or {}

    # Sort synonyms by length (descending) to match specific phrases first.
    # e.g., 'conditioned game' will be checked before 'game'.
    sorted_synonyms = sorted(type_map.keys(), key=len, reverse=True)

    found_canonicals = set()
    for syn in sorted_synonyms:
        pat = _plural_aware_pattern(syn)
        if pat.search(t):
            # Once a synonym is found, add its canonical form.
            # To prevent sub-string matches (like 'game' in 'conditioned game'),
            # we can remove the matched text from the search string for subsequent iterations.
            canonical_form = type_map[syn]
            found_canonicals.add(canonical_form)
            t = pat.sub('', t)  # Remove the found phrase to avoid re-matching

    # If explicit 'mix' phrasing was in the original text OR multiple types present → 'mix'
    if re.search(r"\bmix\b", _norm(text)) or len(found_canonicals) >= 2:
        return "mix"

    if len(found_canonicals) == 1:
        return next(iter(found_canonicals))

    return None




# ---------- level ----------
def parse_squash_level(text: str) -> Optional[str]:
    """
    Extracts player's level using SYNONYM_MAP['squashLevel'].
    - Hyphen/space tolerant matching for multiword synonyms.
    - Plural tolerant: 'beginners', 'intermediates', 'professionals'.
    - Guards against ambiguous 'medium' firing on 'medium intensity' by requiring a nearby
      level context (level/player/skill/experience) when the token is 'medium'.
    - If multiple levels are mentioned (e.g., 'intermediate to advanced'), returns the highest:
      advanced > intermediate > beginner.
    """
    t = _norm(text)
    level_map = SYNONYM_MAP.get("squashLevel", {}) or {}

    def _present_plural_ok(token: str) -> bool:
        # treat spaces/hyphens interchangeably; plural on last token
        return _plural_aware_pattern(token).search(t) is not None

    found: Set[str] = set()
    for syn, canon in level_map.items():
        syn_l = syn.lower()

        # special guard for ambiguous 'medium'
        if syn_l == "medium":
            if not re.search(r"\bmedium\s*(?:level|player|players|skill|experience)\b", t):
                continue

        if _present_plural_ok(syn_l):
            found.add(canon)

    if not found:
        return None

    # Prioritize highest level if multiple are present
    priority = {"beginner": 0, "intermediate": 1, "advanced": 2}
    return max(found, key=lambda c: priority.get(c, -1))


# ---------- shots ----------

# --- Shots parsing ------------------------------------------------------------

# Canonical general shot families we care about
_GENERAL_SHOTS: Set[str] = {
    "drive", "cross", "drop", "boast", "lob", "volley", "kill", "serve", "nick", "flick"
}

# Heads that can appear as the final noun in specifics (singular form)
_HEAD_WORDS: Set[str] = {"drive", "drop", "lob", "boast", "kill", "nick", "serve"}

# Use your configured general->specific map
_SPEC_MAP: Dict[str, list] = SPECIFIC_MAPS.get("GENERAL_SHOT_TYPES", {}) or {}

def _norm_phrase(s: str) -> str:
    # lowercase, replace all hyphen variants with spaces, collapse whitespace
    s = s.lower().replace("—", "-").replace("–", "-")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _word_to_digit(tok: str) -> str:
    # map number-words we already support in _WORD_NUM
    t = tok.strip().lower()
    if t in _WORD_NUM:
        return str(_WORD_NUM[t])
    return t

def _make_fuzzy_phrase_regex(phrase_norm: str) -> re.Pattern:
    """
    Build a regex that treats spaces/hyphens interchangeably between tokens.
    Allow plural 's' on the last token when it's a head word.
    """
    toks = phrase_norm.split()
    if not toks:
        return re.compile(r"(?!x)x")
    parts = []
    for i, tok in enumerate(toks):
        esc = re.escape(tok)
        if i == len(toks) - 1 and tok in _HEAD_WORDS:
            # optional plural 's' for the head
            parts.append(fr"{esc}s?")
        else:
            parts.append(esc)
    body = r"(?:[-\s]+)".join(parts)
    return re.compile(rf"\b{body}\b", re.IGNORECASE)

# Precompute: phrase -> set(generals it belongs to), compiled patterns, and canonical form
_PHRASE_TO_GENERALS: Dict[str, Set[str]] = defaultdict(set)
_PHRASE_PATTERNS: Dict[str, re.Pattern] = {}
_PHRASE_CANONICAL: Dict[str, str] = {}  # normalized -> canonical (as in config)

for gen, phrases in _SPEC_MAP.items():
    for p in phrases or []:
        pn = _norm_phrase(p)
        if not pn:
            continue
        _PHRASE_TO_GENERALS[pn].add(gen)
        _PHRASE_CANONICAL.setdefault(pn, p)

for pn in list(_PHRASE_TO_GENERALS.keys()):
    # We do NOT include pure general words in "specifics"
    if pn in _GENERAL_SHOTS:
        continue
    _PHRASE_PATTERNS[pn] = _make_fuzzy_phrase_regex(pn)

def _general_plural_regex(word: str) -> re.Pattern:
    """
    Build a regex for a general word that accepts simple plurals:
    - always allow +s
    - allow +es for s/x/z/ch/sh endings (e.g., cross -> crosses)
    - allow y->ies when consonant+y (none of your generals need this now,
      but it’s safe to include).
    """
    forms = {word, word + "s"}
    if re.search(r"(?:s|x|z|ch|sh)$", word):
        forms.add(word + "es")
    if re.search(r"[^aeiou]y$", word):
        forms.add(word[:-1] + "ies")
    if word == "drive":
        forms.add("driving")
    body = "|".join(map(re.escape, sorted(forms, key=len, reverse=True)))
    return re.compile(rf"\b(?:{body})\b", re.IGNORECASE)

# General words (plural-aware)
_GENERAL_PATTERNS: Dict[str, re.Pattern] = {g: _general_plural_regex(g) for g in _GENERAL_SHOTS}

# A convenience set of allowed normalized specifics for validation
_ALLOWED_SPEC_NORMALS: Set[str] = set(_PHRASE_CANONICAL.keys())

def _split_mod_list(mods_text: str) -> list[str]:
    """
    Split a 'mods' chunk like 'deep, hard and straight' or 'cross- and straight'
    into individual modifier strings (not including the head).
    """
    # normalize separators to ' , ' and ' and ' and ' or ' and '/'
    t = re.sub(r"[,\u2013\u2014;]+", ",", mods_text)        # commas/semicolons/dashes → ','
    # space out slashes
    t = re.sub(r"\s*/\s*", " / ", t)
    # split by , and conjunctions
    parts = re.split(r"\s*(?:,|and|or|/)\s*", t.strip())
    parts = [p for p in (p.strip() for p in parts) if p]
    return parts

def _clean_modifier_piece(piece: str, head: str, suffix_hint: Optional[str]) -> str:
    p = piece.lower().strip()
    # 1) Drop filler words that precede real modifiers
    p = re.sub(r"\b(?:both|either|each|all)\b", "", p).strip()
    # 2) Remove head if embedded (e.g., 'straight-drops' -> 'straight')
    p = re.sub(rf"(?:[-\s]+)?{re.escape(head)}s?\b", "", p).strip()
    # 3) Tidy punctuation/hyphens and spacing
    p = re.sub(r"-$", "", p).strip()
    p = re.sub(r"\s+", " ", p)

    # 4) Number words -> digits
    p = " ".join(_word_to_digit(tok) for tok in p.split())

    # 5) If mods mention 'wall', standardize forms like '2 wall'/'two wall' -> '2-wall'
    if suffix_hint == "wall":
        # Handle both space and hyphen separated versions
        wall_pattern = rf"((?:{'|'.join(_WORD_NUM.keys())}|\d+))[-\s]wall"
        m = re.search(wall_pattern, p)
        if m:
            num = m.group(1)
            # Convert word numbers to digits if needed
            if num in _WORD_NUM:
                num = str(_WORD_NUM[num])
            p = f"{num}-wall"

    return p


def _expand_coordinated_shots(text: str) -> tuple[Set[str], Set[str]]:
    """
    Expand coordinated 'mods + head(s)' phrases into specific shot phrases,
    validate against SPECIFIC_MAP, and imply generals.
    Handles:
      - "two- and three-wall boasts" -> {"2-wall boast","3-wall boast"}
      - "cross and straight lobs" -> {"cross lob","straight lob"}
      - "cross- and straight-drops" -> {"cross drop","straight drop"}
      - "volley deep and hard drives" -> {"volley deep drive","volley hard drive"}
      - "volley cross- and straight-drops" -> {"volley cross drop","volley straight drop"}
    """
    gens: Set[str] = set()
    specs: Set[str] = set()

    # Normalize: lowercase and treat hyphens as spaces so suspended hyphens turn into tokens
    T = text.lower().replace("—", "-").replace("–", "-").replace("-", " ")
    T = re.sub(r"\s+", " ", T)

    # For each head (drive, drop, lob, boast, kill, nick, serve), find "mods ... head(s)"
    head_re = "|".join(map(re.escape, sorted(_HEAD_WORDS)))
    # Optional 'volley ' prefix (applies to all mods)
    # capture a small window before head to avoid over-greedy matches
    pattern = re.compile(
        rf"(?P<prefix>\bvolley\s+)?(?P<mods>(?:\w+(?:\s+\w+)*\s*(?:,|\band\b|\bor\b|/)\s*)+\w+(?:\s+\w+)*)\s+(?P<head>{head_re})s?\b"
    )

    for m in pattern.finditer(T):
        prefix = (m.group("prefix") or "").strip()
        head = m.group("head").strip()  # singular head
        mods_chunk = m.group("mods").strip()

        # If any token in mods_chunk contains 'wall', we can hint suffix attachment
        suffix_hint = "wall" if re.search(r"\bwall\b", mods_chunk) else None

        mods = _split_mod_list(mods_chunk)
        if not mods:
            continue

        # Clean each modifier
        cleaned_mods = []
        for piece in mods:
            cm = _clean_modifier_piece(piece, head=head, suffix_hint=suffix_hint)
            if cm:
                cleaned_mods.append(cm)

        # Compose candidates and validate against SPEC_MAP canonicals
        for cm in cleaned_mods:
            candidate = f"{prefix + ' ' if prefix else ''}{cm} {head}".strip()
            norm_cand = _norm_phrase(candidate)
            if norm_cand in _ALLOWED_SPEC_NORMALS:
                canonical = _PHRASE_CANONICAL[norm_cand]
                specs.add(canonical)
                # imply generals for this specific
                for g in _PHRASE_TO_GENERALS.get(norm_cand, ()):
                    gens.add(g)

        # If we saw a 'volley' prefix, make sure 'volley' is present as a general
        if prefix:
            gens.add("volley")

        # Always add the head's general
        # map: drive->drive, drop->drop, lob->lob, boast->boast, kill->kill, nick->nick, serve->serve
        head_general = head  # they align
        if head_general in _GENERAL_SHOTS:
            gens.add(head_general)

    return gens, specs

def parse_shots(text: str) -> dict:
    """
    Extracts general shots and specific phrases.
    Returns:
      {
        "shots_general": set[str],
        "shots_specific": set[str],
      }
    - Specifics are phrases from GENERAL_SHOT_TYPES (excluding bare general words).
    - Specifics imply their general families (many phrases map to multiple generals).
    - Handles coordination ellipsis (mods + head) and suspended hyphens.
    - Hyphen/space tolerant, boundary safe, deduplicated, case-insensitive.
    - Generals accept simple plurals (e.g., 'boasts' -> 'boast').
    """
    if not text:
        return {"shots_general": set(), "shots_specific": set()}

    # Setup the stemmer
    stemmer = PorterStemmer()
    # Create a mapping from a stemmed general shot to its original canonical form
    # e.g., {'boast': 'boast', 'drive': 'drive'}
    stemmed_general_map = {stemmer.stem(g): g for g in _GENERAL_SHOTS}

    # 0) Expand coordinated forms first (adds both specifics + implied generals)
    generals: Set[str] = set()
    specifics: Set[str] = set()
    g2, s2 = _expand_coordinated_shots(text)
    generals |= g2
    specifics |= s2

    # 1) Direct general mentions (allow simple plural 's')
    # Tokenize and stem the input text once
    words_in_text = re.findall(r'\b\w+\b', text.lower())
    stemmed_text_words = [stemmer.stem(word) for word in words_in_text]

    # Check for matches
    for stemmed_word in stemmed_text_words:
        if stemmed_word in stemmed_general_map:
            # If a stemmed word matches, add the original canonical word
            generals.add(stemmed_general_map[stemmed_word])

    # 2) Specific phrases (hyphen/space tolerant, allow plural head)
    for phrase_norm, pat in _PHRASE_PATTERNS.items():
        if pat.search(text):
            canonical = _PHRASE_CANONICAL.get(phrase_norm, phrase_norm)
            specifics.add(canonical)
            for g in _PHRASE_TO_GENERALS.get(phrase_norm, ()):
                generals.add(g)

    return {"shots_general": generals, "shots_specific": specifics}

# ---------- shotsides (FH/BH) ----------
def parse_shot_side(text: str) -> list[str]:
    """
    Extracts shot side preferences.
    Uses SYNONYM_MAP['shotSide'] to map synonyms → {'forehand','backhand','both'}.

    - Hyphen/space/slash tolerant for multiword phrases (e.g., "fh and bh", "forehand-and-backhand", "fh/bh").
    - Plural tolerant: 'forehands', 'backhands', 'FHs/BHs'.
    - Returns a list of unique canonicals. If both sides are present or 'both' is mentioned,
      it includes 'both' and the individual sides for maximal overlap downstream.
    """
    t = _norm(text)

    # normalize common plural/abbr variants to canonical tokens before matching
    t = _normalize_words(
        t,
        [
            (r"\bforehands\b", "forehand"),
            (r"\bbackhands\b", "backhand"),
            (r"\bfhs\b", "fh"),
            (r"\bbhs\b", "bh"),
            (r"\bboth\s+(?:sides|hands)\b", "both"),
        ],
    )

    side_map = SYNONYM_MAP.get("shotSide", {}) or {}

    def _present_plural_ok(token: str) -> bool:
        # tolerate spaces/hyphens/slashes and plural on last token
        return _plural_aware_pattern(token).search(t) is not None

    found: set[str] = set()

    for syn, canon in side_map.items():
        if _present_plural_ok(syn):
            if canon == "both":
                # expand to cover matching with docs that list either style
                found.update(["forehand", "backhand", "both"])
            else:
                found.add(canon)

    # If both present, also add 'both' to maximize overlap with docs that use the single token.
    if "forehand" in found and "backhand" in found:
        found.add("both")

    return sorted(found)


# ---------- movement ----------
def parse_movement(text: str) -> list[str]:
    """
    Extracts movement intents (front, middle, back, sideways, diagonal, multi-directional)
    using SYNONYM_MAP['movement'].

    - Hyphen/space/slash tolerant (e.g., 'side-to-side', 'side to side', 'side-to/side').
    - Boundary-safe.
    - Ignores the common non-squash phrase 'back-to-back' to avoid false positives on 'back'.
    - Plural/gerund tolerant for 'diagonal(s)/diagonally', plus 'fronts/middles/backs' singularization.
    - Returns a sorted list of unique canonical values.
    """
    t = _norm(text)
    # normalize some plural/gerund forms to canonical movement words
    t = _normalize_words(
        t,
        [
            (r"\bdiagonals?\b", "diagonal"),
            (r"\bdiagonally\b", "diagonal"),
            (r"\bfronts\b", "front"),
            (r"\bmiddles\b", "middle"),
            (r"\bbacks\b", "back"),
            (r"\bmulti[-\s]*directionals?\b", "multi-directional"),
        ],
    )

    move_map = SYNONYM_MAP.get("movement", {}) or {}

    # Hard ignore "back-to-back" (very common non-squash phrase)
    if re.search(r"\bback[-\s]+to[-\s]+back\b", t):
        t_masked = re.sub(r"\bback[-\s]+to[-\s]+back\b", " __BTB__ ", t)
    else:
        t_masked = t

    found: set[str] = set()
    for syn, canon in move_map.items():
        # plural on last token is okay (e.g., "sideways" effectively unchanged, but safe)
        if _plural_aware_pattern(syn).search(t_masked):
            found.add(canon)

    return sorted(found)


# ---------- intensity ----------
def parse_intensity(text: str) -> Optional[str]:
    """
    Extracts session intensity using SYNONYM_MAP['intensity'].
    - Hyphen/space tolerant for multiword synonyms (e.g., 'extremely-high').
    - If multiple intensities appear (e.g., 'low/medium', 'medium to high'), returns the highest:
      high > medium > low.
    """
    t = _norm(text)
    inten_map = SYNONYM_MAP.get("intensity", {}) or {}

    def _fuzzy_present(token: str) -> bool:
        # treat spaces/hyphens interchangeably
        tokens = token.lower().split()
        body = r"(?:[-\s]+)".join(map(re.escape, tokens))
        return re.search(rf"\b{body}\b", t) is not None

    found: Set[str] = set()
    for syn, canon in inten_map.items():
        if _fuzzy_present(syn):
            found.add(canon)

    if not found:
        return None

    priority = {"low": 0, "medium": 1, "high": 2}
    return max(found, key=lambda c: priority.get(c, -1))
