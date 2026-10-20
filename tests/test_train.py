from __future__ import annotations

from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minibert.tokenizer import SPECIAL_TOKENS, Tokenizer
from minibert.tokenizer.train import available_cpus, count_words, train_vocab

# The example from Hugging Face's course on WordPiece and BPE.
WORDS = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}
ALPHABET = ["b", "##b", "g", "##g", "h", "##h", "n", "##n", "p", "##p", "s", "##s", "u", "##u"]


def test_merges_the_most_frequent_pair_first():
    result = train_vocab(WORDS, vocab_size=100, min_frequency=1)

    # "##u" + "##g" appears 20 times (hug, pug, hugs), "##u" + "##n" 16 times, and so on.
    # Ties go to the alphabetically first pair: "hug" + "##s" (5) before "p" + "##ug" (5).
    assert result.vocab == [*SPECIAL_TOKENS, *ALPHABET, "##ug", "##un", "hug", "pun", "hugs", "pug", "bun"]
    assert (result.merges, result.alphabet, result.skipped_words) == (7, 7, 0)
    assert (result.words, result.distinct_words) == (36, 5)
    assert result.seconds >= 0


def test_stops_at_the_vocabulary_size():
    assert train_vocab(WORDS, vocab_size=21, min_frequency=1).vocab[-2:] == ["##ug", "##un"]


def test_a_tiny_vocabulary_keeps_the_special_tokens_first():
    assert train_vocab(WORDS, vocab_size=7).vocab == [*SPECIAL_TOKENS, "b", "##b"]


def test_never_merges_rare_pairs():
    result = train_vocab(WORDS, vocab_size=100, min_frequency=13)

    assert result.vocab[len(SPECIAL_TOKENS) + len(ALPHABET) :] == ["##ug", "##un", "hug"]


def test_rare_characters_are_left_out():
    result = train_vocab({"aaa": 10, "ab": 9, "xa": 1}, vocab_size=100, alphabet_size=2, min_frequency=1)

    assert "x" not in result.vocab
    assert result.skipped_words == 1


def test_vocab_size_must_fit_the_special_tokens():
    with pytest.raises(ValueError, match="vocab_size must be at least 5"):
        train_vocab(WORDS, vocab_size=4)


def test_a_trained_vocabulary_works_in_the_tokenizer():
    tokenizer = Tokenizer(train_vocab(WORDS, vocab_size=100, min_frequency=1).vocab)

    assert tokenizer.tokenize("hugs bun pugs") == ["hugs", "bun", "pug", "##s"]


@settings(max_examples=200, deadline=None)
@given(
    words=st.dictionaries(st.text(alphabet="abcde", min_size=1, max_size=8), st.integers(1, 50), min_size=1),
    vocab_size=st.integers(15, 80),
)
def test_every_training_word_can_be_tokenized_without_unknowns(words, vocab_size):
    result = train_vocab(words, vocab_size=vocab_size, min_frequency=1)
    tokenizer = Tokenizer(result.vocab)

    assert len(result.vocab) == len(set(result.vocab)) <= vocab_size
    for word in words:
        pieces = tokenizer.split_word(word)
        assert "[UNK]" not in pieces
        assert "".join(piece.removeprefix("##") for piece in pieces) == word


def test_count_words():
    lines = ["Hello, hello world!", "Héllo again."]

    assert count_words(lines) == Counter({"hello": 3, ",": 1, "world": 1, "!": 1, "again": 1, ".": 1})
    assert count_words(lines, lowercase=False)["Hello"] == 1


def test_counting_in_parallel_gives_the_same_counts():
    lines = [f"line {n} has words, like line {n % 7}." for n in range(50_000)]

    assert count_words(lines, workers=2) == count_words(lines)


def test_counts_available_cpus(monkeypatch):
    from minibert.tokenizer import train

    assert available_cpus() >= 1
    monkeypatch.delattr(train.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(train.os, "cpu_count", lambda: None)
    assert available_cpus() == 1
