# tests/test_user_desires_shots.py
import pytest

from rag.parsers.user_query_parser import parse_shots

# Helper to normalize the function output into two sets
def _extract_sets(out):
    assert isinstance(out, dict), "parse_shots must return a dict"
    gens = set(out.get("shots_general", []))
    specs = set(out.get("shots_specific", []))
    return gens, specs


@pytest.mark.parametrize(
    "text, exp_gen, exp_spec_subset",
    [
        # 1) General-only mentions
        ("work on drive and cross", {"drive", "cross"}, set()),

        # 2) Specific implies general(s)
        ("practice deep drive", {"drive"}, {"deep drive"}),
        ("use a straight drop", {"drop"}, {"straight drop"}),
        ("add a cross lob", {"cross", "lob"}, {"cross lob"}),

        # 3) Volley-prefixed specifics imply volley + base families
        ("volley deep drive and volley cross lob",
         {"volley", "drive", "cross", "lob"},
         {"volley deep drive", "volley cross lob"}),

        # 4) Boast variants (singulars)
        ("2-wall boast; 3-wall boast; reverse boast; trickle boast",
         {"boast"},
         {"2-wall boast", "3-wall boast", "reverse boast", "trickle boast"}),

        # 4b) Boast variants (plurals) – plural tolerant
        ("2-wall boasts; 3-wall boasts; reverse boasts; trickle boasts",
         {"boast"},
         {"2-wall boast", "3-wall boast", "reverse boast", "trickle boast"}),

        # 5) Kill / serve / nick / flick (general)
        ("serve and flick to the nick",
         {"serve", "flick", "nick"},
         set()),

        # 6) Mixed, multi-phrase example
        ("boast → cross → drive; optional counter drop; finish with volley straight kill",
         {"boast", "cross", "drive", "drop", "volley", "kill"},
         {"counter drop", "volley straight kill"}),

        # plural
        ("2-wall boasts; 3-wall boasts; reverse boasts; trickle boasts",
        {"boast"},
        {"2-wall boast", "3-wall boast", "reverse boast", "trickle boast"},)
    ],
)
def test_parse_shots_general_and_specific(text, exp_gen, exp_spec_subset):
    gens, specs = _extract_sets(parse_shots(text))
    assert exp_gen.issubset(gens), f"missing generals {exp_gen - gens} in {gens}"
    assert exp_spec_subset.issubset(specs), f"missing specifics {exp_spec_subset - specs} in {specs}"


def test_parse_shots_case_and_dedup():
    text = "Boast, BOAST, boast; Cross Lob, cross lob; VOLLEY Straight DROP"
    gens, specs = _extract_sets(parse_shots(text))
    # generals present irrespective of case, deduped
    assert "boast" in gens
    assert {"cross", "lob", "volley", "drop"}.issubset(gens)
    # specifics deduped, normalized spacing/hyphens
    assert "cross lob" in specs
    assert "volley straight drop" in specs


@pytest.mark.parametrize(
    "text",
    [
        # boundaries: should NOT fire on these
        "we were boasting about wins in the lobby",
        "the cafeteria lobby is open",
        "a backhanded compliment is not backhand practice",
    ],
)
def test_parse_shots_negative_boundaries(text):
    gens, specs = _extract_sets(parse_shots(text))
    assert gens == set(), f"expected no generals, got {gens}"
    assert specs == set(), f"expected no specifics, got {specs}"


def test_parse_shots_hyphen_variants():
    text = "reverse-boast and cross-lob with hard-drive follow-up"
    gens, specs = _extract_sets(parse_shots(text))
    # hyphens should be treated like spaces
    assert {"boast", "cross", "lob", "drive"}.issubset(gens)
    # longest specific phrases recognized where applicable
    assert "reverse boast" in specs
    assert "cross lob" in specs
    # "hard drive" may also be recognized as a specific
    assert any(s in specs for s in ["hard drive", "deep drive", "straight drive", "volley hard drive"])


def test_parse_shots_no_overlap_general_in_specifics():
    text = "drive, cross, drop, boast, lob, volley, kill, serve, nick, flick"
    gens, specs = _extract_sets(parse_shots(text))
    # all 10 generals should be present
    assert gens == {"drive","cross","drop","boast","lob","volley","kill","serve","nick","flick"}
    # specifics should be empty (pure general words must not appear in specifics)
    assert specs == set()

def test_parse_shots_plurals_general_words():
    gens, specs = _extract_sets(parse_shots("work on boasts and crosses"))
    assert {"boast", "cross"}.issubset(gens)
    assert specs == set()


def _extract_sets(out):
    assert isinstance(out, dict)
    return set(out.get("shots_general", [])), set(out.get("shots_specific", []))

@pytest.mark.parametrize(
    "text, exp_gens, exp_specs",
    [
        # numeric + '-wall' with plural head
        #("two- and three-wall boasts", {"boast"}, {"2-wall boast", "3-wall boast"}),

        # numbers in digits too
        ("2-wall and 3-wall boasts", {"boast"}, {"2-wall boast", "3-wall boast"}),

        # classic mods + head (plural)
        ("cross and straight lobs", {"cross", "lob"}, {"cross lob", "straight lob"}),

        # suspended hyphen + plural head
        ("cross- and straight-drops", {"cross", "drop"}, {"cross drop", "straight drop"}),

        # with 'volley' prefix applied to all
        ("volley deep and hard drives",
         {"volley", "drive"},
         {"volley deep drive", "volley hard drive"}),

        # mixed punctuation and ‘or’
        ("reverse, trickle or 3-wall boasts",
         {"boast"},
         {"reverse boast", "trickle boast", "3-wall boast"}),

        # suspended hyphen + prefix
        ("volley cross- and straight-drops",
         {"volley", "cross", "drop"},
         {"volley cross drop", "volley straight drop"}),

        # ensure we don’t accept combos not in the map
        ("deep and cross lobs (should only keep cross lob if deep lob is not allowed)",
         {"lob", "cross"},  # general for ‘lob’ and ‘cross’ still appear if present elsewhere
         {"cross lob"}),     # deep lob is not in your config, so it must NOT appear

        # kept failing in run_field_retrieval.py
        ("Create a training routine that helps with 'Strategic Application of Both 2-Wall and 3-Wall Boasts within a Driving Game",
         {"boast", "drive"},
         {"2-wall boast", "3-wall boast"}),
    ],
)
def test_shots_coordination_expansion(text, exp_gens, exp_specs):
    gens, specs = _extract_sets(parse_shots(text))
    # generals must include all expected (there can be more due to other matches)
    assert exp_gens.issubset(gens), f"missing generals {exp_gens - gens} in {gens}"
    # specifics must include all expected expansions
    assert exp_specs.issubset(specs), f"missing specifics {exp_specs - specs} in {specs}"

def test_plural_generals_map_to_singular():
    gens, specs = _extract_sets(parse_shots("boasts, drops and lobs"))
    # plural generals should map to canonical singulars
    assert {"boast", "drop", "lob"}.issubset(gens)
    assert specs == set()


