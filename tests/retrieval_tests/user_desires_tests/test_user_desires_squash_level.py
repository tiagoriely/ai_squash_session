# tests/test_user_desires_level.py
import pytest
from rag.parsers.user_query_parser import parse_squash_level


@pytest.mark.parametrize(
    "text,expected",
    [
        # Beginner & synonyms
        ("beginner session", "beginner"),
        ("novice drill", "beginner"),

        # Intermediate & synonyms
        ("intermediate players", "intermediate"),
        ("medium level drill", "intermediate"),
        ("Players are medium level", "intermediate"),
        ("INTERMEDIATE", "intermediate"),

        # Advanced & synonyms
        ("advanced session", "advanced"),
        ("expert players only", "advanced"),
        ("pro level match prep", "advanced"),
        ("professional-level players", "advanced"),
        ("expert-level ghosting", "advanced"),
        ("PRO-LEVEL routine", "advanced"),

        # Mixed — pick highest
        ("intermediate to advanced", "advanced"),
        ("beginner or intermediate", "intermediate"),
        ("beginner and advanced", "advanced"),
        ("novice moving towards intermediate", "intermediate"),
    ],
)
def test_parse_squash_level_positive(text, expected):
    assert parse_squash_level(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # Guard against 'medium' without level context
        "medium intensity drill",
        "keep intensity medium please",
        # No level present
        "looking for a 60-min conditioned game",
        # Word-boundary safety (should not match 'pro' inside other words)
    ],
)
def test_parse_squash_level_negative(text):
    assert parse_squash_level(text) is None
