from __future__ import annotations

import google_tokenization as google
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from minibert.tokenizer.text import is_cjk, is_control, is_punctuation, split_words


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("Hello, world!", ["hello", ",", "world", "!"]),
        ("don't stop", ["don", "'", "t", "stop"]),
        ("$4.99 (tax)", ["$", "4", ".", "99", "(", "tax", ")"]),
        ("Héllo Ça Ñandú", ["hello", "ca", "nandu"]),
        ("東京は大きい", ["東", "京", "は", "大", "きい"]),  # each kanji alone; hiragana runs stay together
        ("tab\there\nnewline\rreturn", ["tab", "here", "newline", "return"]),
        ("no\u00a0break\u2003em\u3000ideographic", ["no", "break", "em", "ideographic"]),
        ("line\u2028separator", ["line", "separator"]),
        ("zero\u200bwidth", ["zerowidth"]),  # a format character is dropped, not a space
        ("bell\x07here", ["bellhere"]),
        ("vertical\x0btab", ["verticaltab"]),  # a control character, not whitespace, in BERT
        ("nul\x00and\ufffdreplacement", ["nulandreplacement"]),
        ("«quotes» — dash…", ["«", "quotes", "»", "—", "dash", "…"]),
        ("", []),
        ("   \t ", []),
    ],
)
def test_split_words(text, words):
    assert split_words(text) == words


def test_cased_keeps_capitals_and_accents():
    assert split_words("Héllo, World", lowercase=False) == ["Héllo", ",", "World"]
    assert split_words("Hello, World", lowercase=False) == ["Hello", ",", "World"]


def test_cjk_splitting_can_be_turned_off():
    assert split_words("東京", split_cjk=False) == ["東京"]


@pytest.mark.parametrize(
    ("char", "control", "punctuation", "cjk"),
    [
        ("\t", False, False, False),
        ("\x07", True, False, False),
        ("\u200b", True, False, False),
        ("$", False, True, False),
        ("^", False, True, False),
        ("¿", False, True, False),
        ("a", False, False, False),
        ("東", False, False, True),
        ("\U00020000", False, False, True),
        ("は", False, False, False),
    ],
)
def test_character_classes(char, control, punctuation, cjk):
    assert (is_control(char), is_punctuation(char), is_cjk(char)) == (control, punctuation, cjk)


@pytest.mark.parametrize("lowercase", [True, False])
@settings(max_examples=3000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=st.text(max_size=80))
def test_splits_words_exactly_like_googles_code(lowercase, text):
    assert split_words(text, lowercase) == google.BasicTokenizer(do_lower_case=lowercase).tokenize(text)


@settings(max_examples=1000, deadline=None)
@given(text=st.text(alphabet=st.characters(max_codepoint=127), max_size=80))
def test_the_ascii_fast_path_matches_googles_code(text):
    assert split_words(text) == google.BasicTokenizer(do_lower_case=True).tokenize(text)
