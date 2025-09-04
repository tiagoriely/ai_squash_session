# tests/test_user_desires_duration.py
import pytest
from rag.parsers.user_query_parser import parse_duration

@pytest.mark.parametrize(
    "text,expected",
    [
        ("60-min conditioned game focusing on boasts", 60),
        ("a 45 minutes drill", 45),
        ("an hour solo practice", 60),
        ("about an hour, ghosting", 60),
        ("hour and a half with boasts", 90),
        ("1h30 drill on crosses", 90),
        ("1 h 30 session", 90),
        ("90 minutes alley game", 90),
        ("around 75 minutes of mixed games", 75),
        ("30min warmup and 60min session", 60),   # prefer largest single-session duration
        ("45-minute progressive family", 45),
        ("Need ~60 min total", 60),
    ],
)
def test_parse_duration_basic(text, expected):
    assert parse_duration(text) == expected

@pytest.mark.parametrize(
    "text,allowed,expected",
    [
        ("58 min", [45, 60, 90], 60),   # nearest allowed
        ("62 minutes", [45, 60, 90], 60),
        ("80 minutes", [45, 60, 90], 90),
        ("47 minutes", [45, 60, 90], 45),
        ("1.5 hours", [45, 60, 90], 90),
    ],
)
def test_parse_duration_with_allowed_nearest(text, allowed, expected):
    assert parse_duration(text, allowed_durations=allowed) == expected

@pytest.mark.parametrize(
    "text",
    [
        ("no time specified"),
        ("quick drill please"),
        ("as long as you want"),
    ],
)
def test_parse_duration_none(text):
    assert parse_duration(text) is None
