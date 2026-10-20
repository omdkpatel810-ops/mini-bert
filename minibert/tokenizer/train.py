"""Learn a WordPiece vocabulary from text.

Google never published how it built BERT's vocabulary. This uses the method Hugging
Face's WordPiece trainer uses:

1. split the text into words exactly as the tokenizer will, and count them;
2. start the vocabulary with single characters, each in two forms: "a" to start a
   word and "##a" to continue one;
3. write every word as those pieces ("hug" → "h", "##u", "##g"), then repeatedly glue
   the most frequent neighbouring pair into a new piece ("##u" + "##g" → "##ug")
   until the vocabulary is full.

Counting every pair again after each merge would take hours, so the counts are
updated only for the words a merge touches, and a heap finds the next most
frequent pair.
"""

from __future__ import annotations

import heapq
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import islice, pairwise

from minibert.tokenizer.text import split_words
from minibert.tokenizer.wordpiece import CONTINUATION, SPECIAL_TOKENS

Pair = tuple[str, str]


@dataclass(frozen=True)
class TrainedVocab:
    vocab: list[str]
    words: int  # words in the training text
    distinct_words: int
    skipped_words: int  # distinct words with a character outside the alphabet
    alphabet: int
    merges: int
    seconds: float


def count_words(
    lines: Iterable[str], *, lowercase: bool = True, split_cjk: bool = True, workers: int = 1
) -> Counter[str]:
    """How often each word appears, split exactly as the tokenizer splits text."""
    if workers <= 1:
        return _count_batch(list(lines), lowercase, split_cjk)
    counts: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for partial in pool.map(_count_batch, _batches(lines, 20_000), _repeat(lowercase), _repeat(split_cjk)):
            counts.update(partial)
    return counts


def _batches(lines: Iterable[str], size: int) -> Iterator[list[str]]:
    iterator = iter(lines)
    while batch := list(islice(iterator, size)):
        yield batch


def _count_batch(lines: list[str], lowercase: bool, split_cjk: bool) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in lines:
        counts.update(split_words(line, lowercase, split_cjk))
    return counts


def _repeat(value):
    while True:
        yield value


def train_vocab(
    word_counts: Mapping[str, int],
    *,
    vocab_size: int = 30_522,
    min_frequency: int = 2,
    alphabet_size: int = 1_000,
    special_tokens: Sequence[str] = SPECIAL_TOKENS,
) -> TrainedVocab:
    """Learn up to `vocab_size` tokens. Pairs seen fewer than `min_frequency` times are never merged.

    Only the `alphabet_size` most frequent characters get tokens; words with other
    characters are left out of training and will become [UNK], as in BERT.
    """
    started = time.perf_counter()
    if vocab_size < len(special_tokens):
        raise ValueError(f"vocab_size must be at least {len(special_tokens)}, room for the special tokens.")

    alphabet = _alphabet(word_counts, alphabet_size)
    vocab = list(special_tokens)
    for char in sorted(alphabet):
        vocab += [char, CONTINUATION + char]
    vocab = vocab[:vocab_size]
    known = set(vocab)

    words: list[list[str]] = []
    counts: list[int] = []
    skipped = 0
    for word, count in word_counts.items():
        if not word or any(char not in alphabet for char in word):
            skipped += 1
            continue
        words.append([word[0], *(CONTINUATION + char for char in word[1:])])
        counts.append(count)

    pair_counts: defaultdict[Pair, int] = defaultdict(int)
    pair_words: defaultdict[Pair, set[int]] = defaultdict(set)
    for index, (pieces, count) in enumerate(zip(words, counts, strict=True)):
        for pair in pairwise(pieces):
            pair_counts[pair] += count
            pair_words[pair].add(index)
    heap = [(-count, first, second) for (first, second), count in pair_counts.items()]
    heapq.heapify(heap)

    merges = 0
    while len(vocab) < vocab_size and heap:
        negative, first, second = heapq.heappop(heap)
        count = pair_counts.get((first, second), 0)
        if count != -negative:
            continue  # out of date: the pair's current count was pushed when it changed
        if count < min_frequency:
            break
        merged = first + second[len(CONTINUATION) :]
        # Two different pairs gluing into the same piece never happened in 20,000 random
        # corpora, and seems impossible with left-to-right merging, but a duplicate token
        # would break the vocabulary, so it's guarded anyway.
        if merged not in known:  # pragma: no branch
            vocab.append(merged)
            known.add(merged)
        merges += 1

        changed: set[Pair] = set()
        for index in pair_words.pop((first, second)):
            pieces, word_count = words[index], counts[index]
            for pair in pairwise(pieces):
                pair_counts[pair] -= word_count
                changed.add(pair)
            pieces = _merge(pieces, first, second, merged)
            words[index] = pieces
            for pair in pairwise(pieces):
                pair_counts[pair] += word_count
                pair_words[pair].add(index)
                changed.add(pair)
        for pair in changed:
            current = pair_counts[pair]
            if current > 0:
                heapq.heappush(heap, (-current, *pair))
            else:
                del pair_counts[pair]
                pair_words.pop(pair, None)

    return TrainedVocab(
        vocab=vocab,
        words=sum(word_counts.values()),
        distinct_words=len(word_counts),
        skipped_words=skipped,
        alphabet=len(alphabet),
        merges=merges,
        seconds=time.perf_counter() - started,
    )


def _alphabet(word_counts: Mapping[str, int], size: int) -> set[str]:
    chars: Counter[str] = Counter()
    for word, count in word_counts.items():
        for char in word:
            chars[char] += count
    ranked = sorted(chars.items(), key=lambda item: (-item[1], item[0]))
    return {char for char, _ in ranked[:size]}


def _merge(pieces: list[str], first: str, second: str, merged: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(pieces):
        if i + 1 < len(pieces) and pieces[i] == first and pieces[i + 1] == second:
            out.append(merged)
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return out


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS and Windows
        return os.cpu_count() or 1
