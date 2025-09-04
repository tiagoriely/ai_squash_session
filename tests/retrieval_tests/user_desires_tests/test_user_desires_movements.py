# tests/retrieval_tests/user_desires_tests/test_user_desires_movement.py
import pytest
from rag.parsers.user_query_parser import parse_movement
@pytest.mark.parametrize(
    "text,expected_subset",
    [
        ("front court movement", {"front"}),
        ("work the back corners", {"back"}),
        ("control the middle of the court", {"middle"}),
        ("lateral movement / side-to-side shuffles", {"sideways"}),
        ("side to side steps", {"sideways"}),
        ("diagonal runs between corners", {"diagonal"}),
        ("multi-directional movement pattern", {"multi-directional"}),
        ("all over the court today", {"multi-directional"}),
        ("front then back then middle", {"front", "back", "middle"}),
    ],
)
def test_parse_movement_positive(text, expected_subset):
    out = set(parse_movement(text))
    assert expected_subset.issubset(out), f"missing {expected_subset - out} in {out}"

# @pytest.mark.parametrize(
#     "text",
#     [
#         "sidebar note about lateralus",             # shouldn't match 'lateral'
#     ],
# )
# # Improve the test
# def test_parse_movement_negative(text):
#     out = parse_movement(text)
#     assert out == [], f"unexpected match: {out}"

def test_parse_movement_dedup_and_sort():
    text = "Side-to-side, lateral and side to side again; FRONT and back. But also back and front"
    out = parse_movement(text)
    assert out == ["back", "front", "sideways"]
