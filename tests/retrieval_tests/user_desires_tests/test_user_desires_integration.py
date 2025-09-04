import pytest
from rag.parsers.user_query_parser import parse_user_prompt

def _subset(container_list, subset_list):
    c = set(container_list or [])
    s = set(subset_list or [])
    return s.issubset(c)

@pytest.mark.parametrize(
    "text,allowed,expects",
    [
        # 1) All fields simple
        (
            "60-min conditioned game for two players, intermediate level, medium intensity; "
            "practice deep drive and cross-court nick; backhand only; mostly front and diagonal movement.",
            None,
            {
                "duration": 60,
                "participants": 2,
                "type": "conditioned game",
                "squashLevel": "intermediate",
                "intensity": "medium",
                "shotSide_exact": ["backhand"],  # parse_shot_side returns sorted list
                "movement_contains": ["front", "diagonal"],
                "shots_contains": ["drive", "deep drive", "cross", "nick", "cross-court nick"],
            },
        ),

        # 2) Mix type, multiple specifics, FH/BH expands to ['forehand','backhand','both']
        (
            "Mix: drill then a conditioned games for 4 players; 90 minutes; advanced; high intensity; "
            "use 2-wall boast, counter drop, finish with volley straight kill; FH and BH; side-to-side and back court movement.",
            None,
            {
                "duration": 90,
                "participants": 4,
                "type": "mix",
                "squashLevel": "advanced",
                "intensity": "high",
                "shotSide_exact": ["backhand", "both", "forehand"],
                "movement_contains": ["sideways", "back"],
                "shots_contains": [
                    "boast", "2-wall boast",
                    "drop", "counter drop",
                    "volley", "kill", "volley straight kill",
                ],
            },
        ),

        # 3) Duration snapping, solo detection, fh/bh slash, multi-directional, basic shots
        (
            "58 min solo practice, beginner level, low intensity, serve and flick; multi-directional; fh/bh.",
            [45, 60, 90],
            {
                "duration": 60,  # snapped to nearest allowed
                "participants": 1,
                "type": "solo practice",
                "squashLevel": "beginner",
                "intensity": "low",
                "shotSide_exact": ["backhand", "both", "forehand"],
                "movement_contains": ["multi-directional"],
                "shots_contains": ["serve", "flick"],
            },
        ),

        # 4) Guard: 'medium intensity' should NOT set level, but explicit beginner level should.
        # Also ensure 'back-to-back' does NOT trigger 'back' movement.
        (
            "45 minutes drill, medium intensity, beginner level, back-to-back matches.",
            None,
            {
                "duration": 45,
                "type": "drill",
                "intensity": "medium",
                "squashLevel": "beginner",
                "movement_not_contains": ["back"],
            },
        ),

        # 5) Hyphen/space tolerant shot specifics
        (
            "3-wall boast and cross court nick and volley straight drop; 30min; two players.",
            None,
            {
                "duration": 30,
                "participants": 2,
                "shots_contains": [
                    "3-wall boast", "boast",
                    "cross", "nick", "cross-court nick",
                    "volley", "drop", "volley straight drop",
                ],
            },
        ),
    ],
)
def test_parse_user_prompt_integration(text, allowed, expects):
    out = parse_user_prompt(text, allowed_durations=allowed)

    # Simple equality fields (only assert if provided in expects)
    for k in ["duration", "participants", "type", "squashLevel", "intensity"]:
        if k in expects:
            assert out.get(k) == expects[k], f"{k}: expected {expects[k]}, got {out.get(k)}"

    # Shot sides exact equality (sorted)
    if "shotSide_exact" in expects:
        assert out.get("shotSide") == sorted(expects["shotSide_exact"]), f"shotSide mismatch: {out.get('shotSide')}"

    # Movement subset
    if "movement_contains" in expects:
        assert _subset(out.get("movement", []), expects["movement_contains"]), \
            f"movement missing {set(expects['movement_contains']) - set(out.get('movement', []))}"

    # Movement negatives
    if "movement_not_contains" in expects:
        for mv in expects["movement_not_contains"]:
            assert mv not in (out.get("movement") or []), f"movement unexpectedly contains {mv}"

    # Shots subset (combined list)
    if "shots_contains" in expects:
        assert _subset(out.get("shots", []), expects["shots_contains"]), \
            f"shots missing {set(expects['shots_contains']) - set(out.get('shots', []))}"
