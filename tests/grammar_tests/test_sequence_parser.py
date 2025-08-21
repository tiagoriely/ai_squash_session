import pytest
from src.grammar_tools.dsl_tools.parser import parse_rules_sequence

GOOD = [
    'boast (A) → cross (B) → drive (A) → restart pattern at next boast',
    '( straight drop (B) | straight drop (A) )*',
    'optional: ( extra drive (A) | extra drive (B) )',
    'drive (opponent) # comment'
]

BAD = [
    'IF opponent plays drop: cross',              # not in DSL
    'boast (C)',                                  # unknown actor
    '( straight drop (A) | cross (B) '            # missing ')'
]

@pytest.mark.parametrize("s", GOOD)
def test_good_sequences(s):
    parse_rules_sequence(s)  # no exception

@pytest.mark.parametrize("s", BAD)
def test_bad_sequences(s):
    with pytest.raises(Exception):
        parse_rules_sequence(s)
