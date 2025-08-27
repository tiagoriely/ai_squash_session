import pytest
from rag_old.pipelines.retrieval.field_retrieval.user_desires import parse_type

@pytest.mark.parametrize(
    "text,expected",
    [
        ("I want a drills on boasts", "drill"),
        ("conditioned game focusing on crosses", "conditioned game"),
        ("open game to 11", "conditioned game"),      # "game" → conditioned game (per synonyms)
        ("solo practice for 30 min", "solo practice"),
        ("by myself, 45 minutes", "solo practice"),
        ("no partner available today", "solo practice"),
        ("ghosting session 20min", "ghosting"),
    ],
)
def test_parse_type_basic(text, expected):
    assert parse_type(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("mix of drill and ghosting", "mix"),
        ("a drill then conditioned game", "mix"),
        ("ghosting and solo practice combo", "mix"),
    ],
)
def test_parse_type_mix(text, expected):
    assert parse_type(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        ("just a session please"),
        ("I need something for movement"),
    ],
)
def test_parse_type_none(text):
    assert parse_type(text) is None
