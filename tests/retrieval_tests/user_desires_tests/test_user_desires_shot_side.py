import pytest
from rag.pipelines.retrieval.field_retrieval.user_desires import parse_shot_side

@pytest.mark.parametrize(
    "text,expected",
    [
        ("forehand only", {"forehand"}),
        ("backhands targets", {"backhand"}),
        ("FH focus then BH", {"forehand", "backhand", "both"}),  # both sides implied
        ("bh to fh pattern", {"forehand", "backhand", "both"}),
        ("either side works", {"both", "forehand", "backhand"}),
        ("forehand and backhand drills", {"forehand", "backhand", "both"}),
        ("FH and BH rotations", {"forehand", "backhand", "both"}),
        ("forehand-and-backhand alternating", {"forehand", "backhand", "both"}),
        ("fh/bh switching mid-session", {"forehand", "backhand", "both"}),
        ("both sides", {"both", "forehand", "backhand"}),
    ],
)
def test_parse_shot_side_positive(text, expected):
    out = set(parse_shot_side(text))
    assert expected.issubset(out), f"missing {expected - out} in {out}"

@pytest.mark.parametrize(
    "text",
    [
        "hand warmers",            # no 'forehand'/'backhand'
        "feedback loop",           # contains 'fb' but not fh/bh
        # "backhanded compliment",   # 'backhanded' should not match 'backhand' with \b
    ],
)
def test_parse_shot_side_negative(text):
    out = parse_shot_side(text)
    assert out == [], f"unexpected match: {out}"
