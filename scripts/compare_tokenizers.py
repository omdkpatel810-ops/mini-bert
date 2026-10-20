"""Check our tokenizer against Google's original code and Hugging Face's, token for token.

    python scripts/get_wikitext.py
    python scripts/compare_tokenizers.py

Uses WikiText-103's test and validation text and Google's bert-base-uncased vocabulary.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from tokenizers import BertWordPieceTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "reference"))
import google_tokenization  # noqa: E402  (Google's code, kept in the tests)

from minibert.tokenizer import Tokenizer  # noqa: E402

VOCAB = ROOT / "minibert" / "data" / "bert-base-uncased-vocab.txt"


def timed(name: str, tokenize, lines: list[str]) -> list[list[str]]:
    started = time.perf_counter()
    result = [tokenize(line) for line in lines]
    print(f"  {name:<14} {time.perf_counter() - started:5.1f} s")
    return result


def main() -> None:
    folder = ROOT / "data" / "wikitext-103"
    lines = [
        line for name in ("test.txt", "valid.txt") for line in (folder / name).read_text(encoding="utf-8").splitlines()
    ]

    ours = Tokenizer.load(VOCAB, special_in_text=False)
    google = google_tokenization.FullTokenizer(str(VOCAB), do_lower_case=True)
    hugging_face = BertWordPieceTokenizer(str(VOCAB), lowercase=True)

    print(f"Tokenizing {len(lines):,} lines of WikiText-103 (test and validation):")
    expected = timed("Google", google.tokenize, lines)
    got = timed("ours", ours.tokenize, lines)
    theirs = timed("Hugging Face", lambda line: hugging_face.encode(line, add_special_tokens=False).tokens, lines)

    tokens = sum(map(len, expected))
    for name, result in (("ours", got), ("Hugging Face", theirs)):
        same = sum(a == b for a, b in zip(expected, result, strict=True))
        print(f"{name}: {same:,} of {len(lines):,} lines identical to Google's ({tokens:,} tokens)")


if __name__ == "__main__":
    main()
