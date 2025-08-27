import pytest
from rag_old.pipelines.retrieval.field_retrieval.user_desires import parse_participants

@pytest.mark.parametrize(
    "text,expected",
    [
        ("solo session", 1),
        ("a 60-min solo practice", 1),
        ("by myself please", 1),
        ("on my own today", 1),
        ("no partner available", 1),

        ("1 player routine", 1),
        ("one player only", 1),

        ("2 players drill", 2),
        ("two players conditioning", 2),
        ("a drill for 2 players", 2),
        ("a drill for with a friend", 2),
        ("a drill for with another player", 2),

        ("3-player game", 3),
        ("three players on court", 3),

        ("4 players rotation", 4),
        ("four-player conditioned game", 4),

        # Prefer the largest number clearly tied to players
        ("I might have two or three players", 3),

        # Should ignore unrelated numbers and pick the one tied to players
        ("45-minute drill for two players", 2),
        ("3 sets of 7 points to 11, for 2 players", 2),
        ("best of five games, two players", 2),
    ],
)
def test_parse_participants_happy_path(text, expected):
    assert parse_participants(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        ("quick drill please"),
        ("I’m free tonight"),
        ("progressive family session"),
    ],
)
def test_parse_participants_none(text):
    assert parse_participants(text) is None
