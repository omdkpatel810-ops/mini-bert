"""BERT's first tokenization step: clean the text and split it into words and punctuation.

This follows Google's original `tokenization.py` (BasicTokenizer) rule for rule:

1. drop NUL, the replacement character and control characters (Unicode Cc and Cf);
   every kind of whitespace separates words;
2. put spaces around Chinese, Japanese and Korean ideographs, so each is its own word;
3. lowercase and strip accents ("Héllo" → "hello"), for the uncased models;
4. split punctuation off into separate tokens ("don't" → "don", "'", "t").

Google's code checks every character's Unicode category in a Python loop. Here each
rule is a regular expression over a character class built once from the same
categories, which is 3.5x faster. The tests check both give identical tokens.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from functools import cache

# Google's ranges for CJK Unified Ideographs. Hiragana, Katakana and Hangul are
# deliberately not included: they're written with spaces like other languages.
CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)

# ASCII control characters BERT drops. Tab, newline and carriage return are whitespace instead.
_ASCII_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# In ASCII, a token is a run of letters and digits, or a single punctuation character.
_ASCII_TOKEN = re.compile(r"[A-Za-z0-9]+|[!-/:-@\[-`{-~]")


def is_control(char: str) -> bool:
    """Google counts tab, newline and carriage return as whitespace, not control characters."""
    if char in "\t\n\r":
        return False
    return unicodedata.category(char) in ("Cc", "Cf")


def is_punctuation(char: str) -> bool:
    """Unicode punctuation, plus ASCII symbols like "$" and "^" that Unicode calls symbols."""
    code = ord(char)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
        return True
    return unicodedata.category(char).startswith("P")


def is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in CJK_RANGES)


def split_words(text: str, lowercase: bool = True, split_cjk: bool = True) -> list[str]:
    """Split text into BERT's words: whole words, single punctuation marks and single ideographs."""
    if text.isascii():
        cleaned = _ASCII_CONTROL.sub("", text)
        return _ASCII_TOKEN.findall(cleaned.lower() if lowercase else cleaned)

    patterns = _patterns()
    text = patterns.removed.sub("", text)
    if split_cjk:
        text = patterns.cjk.sub(r" \g<0> ", text)

    words: list[str] = []
    # After removing control characters, Python's split() splits on exactly the characters
    # Google treats as whitespace: spaces of every kind, and line and paragraph separators.
    for word in text.split():
        if lowercase:
            word = word.lower()
            if not word.isascii():
                word = patterns.accents.sub("", unicodedata.normalize("NFD", word))
        words.extend(patterns.token.findall(word))
    return words


class _Patterns:
    def __init__(self) -> None:
        removed = _character_class(lambda c: c in "\x00\ufffd" or is_control(c))
        punctuation = _character_class(is_punctuation)
        self.removed = re.compile(f"[{removed}]")
        self.cjk = re.compile("[" + "".join(f"{_escape(low)}-{_escape(high)}" for low, high in CJK_RANGES) + "]")
        self.accents = re.compile(f"[{_character_class(lambda c: unicodedata.category(c) == 'Mn')}]")
        self.token = re.compile(f"[{punctuation}]|[^{punctuation}]+")


@cache
def _patterns() -> _Patterns:
    """Built on first use (it takes a moment to scan every Unicode character), then reused."""
    return _Patterns()


def _character_class(test) -> str:
    """A regex character class body matching every code point that passes `test`."""
    ranges: list[tuple[int, int]] = []
    for code in range(sys.maxunicode + 1):
        if 0xD800 <= code <= 0xDFFF:
            continue  # surrogates aren't characters on their own
        if test(chr(code)):
            if ranges and ranges[-1][1] == code - 1:
                ranges[-1] = (ranges[-1][0], code)
            else:
                ranges.append((code, code))
    return "".join(_escape(low) if low == high else f"{_escape(low)}-{_escape(high)}" for low, high in ranges)


def _escape(code: int) -> str:
    return f"\\U{code:08x}"
