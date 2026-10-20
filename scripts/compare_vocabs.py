"""Compare vocabularies on held-out text: how many tokens a word takes, and how often words become [UNK].

python scripts/compare_vocabs.py data/wikitext-103/test.txt ours=vocab.txt google
"""

from __future__ import annotations

import sys
from pathlib import Path

from minibert.tokenizer import Tokenizer
from minibert.tokenizer.text import split_words


def main(text_file: str, specs: list[str]) -> None:
    words = [word for line in Path(text_file).read_text(encoding="utf-8").splitlines() for word in split_words(line)]
    print(f"{len(words):,} words in {text_file}")
    vocabs = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        tokenizer = Tokenizer.bert_base_uncased() if spec == "google" else Tokenizer.load(path)
        vocabs[name] = set(tokenizer.vocab)
        pieces = [tokenizer.split_word(word) for word in words]
        per_word = sum(map(len, pieces)) / len(words)
        split = sum(len(p) > 1 for p in pieces) / len(words)
        unknown = sum(p == ["[UNK]"] for p in pieces) / len(words)
        print(f"  {name:<12} {per_word:.3f} tokens per word, {split:.1%} of words split, {unknown:.3%} [UNK]")
    names = list(vocabs)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = len(vocabs[first] & vocabs[second])
            print(f"  {first} and {second} share {shared:,} tokens")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
