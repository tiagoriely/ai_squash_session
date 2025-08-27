import pytest
from rag_old.pipelines.retrieval.field_retrieval.user_desires import parse_intensity

@pytest.mark.parametrize(
    "text,expected",
    [
        ("keep it low intensity", "low"),
        ("please a medium intensity drill", "medium"),
        ("moderate pace today", "medium"),
        ("hard session please", "high"),
        ("very intense workout", "high"),
        ("extremely high effort today", "high"),
        ("high-intensity intervals", "high"),
        ("medium to high intensity", "high"),
        ("low/medium pace", "medium"),
        ("no intensity mentioned here", None),
    ],
)
def test_parse_intensity(text, expected):
    assert parse_intensity(text) == expected
