"""Training data for masked-word pre-training.

Text is tokenized once into a flat file of token ids (2 bytes each), with [SEP]
between articles. Training then reads random 126-token windows from it, adds
[CLS] and [SEP], and hides 15% of the words afresh each time a window is used
(dynamic masking, from RoBERTa), so the model never sees the same puzzle twice.
"""

from __future__ import annotations

import json
import multiprocessing
import re
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from minibert.tokenizer import Tokenizer

IGNORE = -100  # label for positions the loss skips
_WIKITEXT_JOIN = re.compile(r" @(.)@ ")  # WikiText writes "2 @,@ 500" and "guest @-@ starring"
_ARTICLE_TITLE = re.compile(r"^= [^=].* =$")  # "= Robert Boulter =", but not "= = Career = ="


def clean_wikitext(line: str) -> str:
    return _WIKITEXT_JOIN.sub(r"\1", line.strip())


def is_article_title(line: str) -> bool:
    return bool(_ARTICLE_TITLE.match(line.strip()))


# Tokenizing a corpus ------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """Token ids on disk, read without loading the whole file into memory."""

    tokens: np.ndarray  # uint16 memory map
    vocab_size: int

    @classmethod
    def open(cls, path: str | Path) -> Corpus:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(np.memmap(path, dtype=np.uint16, mode="r"), meta["vocab_size"])

    def __len__(self) -> int:
        return len(self.tokens)


def prepare_corpus(
    lines: Iterable[str], tokenizer: Tokenizer, out: str | Path, *, wikitext: bool = False, workers: int = 1
) -> dict:
    """Tokenize text into `out` (token ids) and `out` with .json (counts). Returns the counts.

    Each line is a paragraph. With `wikitext`, its formatting is undone and [SEP]
    goes between articles; otherwise each line is treated as its own document.
    """
    if len(tokenizer) > 65_536:
        raise ValueError("Vocabularies over 65,536 tokens don't fit in 2 bytes per token.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"vocab_size": len(tokenizer), "tokens": 0, "documents": 0, "lines": 0}
    with out.open("wb") as handle:
        for ids, documents, lines_done in _tokenized_batches(lines, tokenizer, wikitext, workers):
            np.asarray(ids, dtype=np.uint16).tofile(handle)
            counts["tokens"] += len(ids)
            counts["documents"] += documents
            counts["lines"] += lines_done
    out.with_suffix(".json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    return counts


def _tokenized_batches(lines: Iterable[str], tokenizer: Tokenizer, wikitext: bool, workers: int):
    batches = _batches(lines, 5_000)
    if workers <= 1:
        for batch in batches:
            yield _tokenize_batch(batch, tokenizer.vocab, tokenizer.lowercase, wikitext)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        # map keeps the batches in order, so articles stay in sequence.
        yield from pool.map(
            _tokenize_batch, batches, _repeat(tokenizer.vocab), _repeat(tokenizer.lowercase), _repeat(wikitext)
        )


def _tokenize_batch(lines: list[str], vocab: list[str], lowercase: bool, wikitext: bool) -> tuple[list[int], int, int]:
    tokenizer = _worker_tokenizer(tuple(vocab), lowercase)
    ids: list[int] = []
    documents = 0
    for line in lines:
        if not line.strip():
            continue
        if wikitext:
            if is_article_title(line):
                if ids:
                    ids.append(tokenizer.sep_id)
                documents += 1
            line = clean_wikitext(line)
        pieces = tokenizer.tokenize(line)
        ids.extend(tokenizer.token_to_id.get(piece, tokenizer.unk_id) for piece in pieces)
        if not wikitext:
            ids.append(tokenizer.sep_id)
            documents += 1
    return ids, documents, len(lines)


_WORKER_TOKENIZERS: dict[tuple, Tokenizer] = {}


def _worker_tokenizer(vocab: tuple[str, ...], lowercase: bool) -> Tokenizer:
    key = (hash(vocab), lowercase)
    if key not in _WORKER_TOKENIZERS:
        _WORKER_TOKENIZERS[key] = Tokenizer(list(vocab), lowercase=lowercase, special_in_text=False)
    return _WORKER_TOKENIZERS[key]


def _batches(lines: Iterable[str], size: int) -> Iterator[list[str]]:
    iterator = iter(lines)
    while batch := list(islice(iterator, size)):
        yield batch


def _repeat(value):
    while True:
        yield value


# Batches for training -------------------------------------------------------------------


def random_windows(
    corpus: Corpus, batch_size: int, length: int, cls_id: int, sep_id: int, generator: torch.Generator
) -> Tensor:
    """`batch_size` random stretches of the corpus, each [CLS] + `length` - 2 tokens + [SEP]."""
    inner = length - 2
    if len(corpus) < inner:
        raise ValueError(f"The corpus has {len(corpus)} tokens, fewer than one window of {inner}.")
    starts = torch.randint(0, len(corpus) - inner + 1, (batch_size,), generator=generator).numpy()
    windows = np.stack([corpus.tokens[start : start + inner] for start in starts]).astype(np.int64)
    batch = torch.from_numpy(windows)
    cls = torch.full((batch_size, 1), cls_id, dtype=torch.long)
    sep = torch.full((batch_size, 1), sep_id, dtype=torch.long)
    return torch.cat([cls, batch, sep], dim=1)


def fixed_windows(corpus: Corpus, count: int, length: int, cls_id: int, sep_id: int) -> Tensor:
    """The first `count` back-to-back windows, for evaluating on the same text every time."""
    inner = length - 2
    available = len(corpus) // inner
    if available < 1:
        raise ValueError(f"The corpus has {len(corpus)} tokens, fewer than one window of {inner}.")
    count = min(count, available)
    windows = torch.from_numpy(np.asarray(corpus.tokens[: count * inner], dtype=np.int64)).view(count, inner)
    cls = torch.full((count, 1), cls_id, dtype=torch.long)
    sep = torch.full((count, 1), sep_id, dtype=torch.long)
    return torch.cat([cls, windows, sep], dim=1)


def mask_tokens(
    ids: Tensor,
    *,
    mask_id: int,
    vocab_size: int,
    special_ids: Iterable[int],
    rate: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """BERT's masking: choose `rate` of the ordinary tokens for the model to guess.

    Of the chosen tokens, 80% become [MASK], 10% become a random token and 10% stay
    as they are, so the model can't learn to only think hard at [MASK]. Returns the
    model's input and the labels: the original token where chosen, IGNORE elsewhere.
    """
    special = torch.tensor(sorted(set(special_ids)), dtype=ids.dtype, device=ids.device)
    candidates = ~torch.isin(ids, special)
    roll = torch.rand(ids.shape, generator=generator, device=ids.device)
    chosen = (roll < rate) & candidates
    labels = torch.where(chosen, ids, torch.full_like(ids, IGNORE))

    action = torch.rand(ids.shape, generator=generator, device=ids.device)
    inputs = ids.clone()
    inputs[chosen & (action < 0.8)] = mask_id
    replace = chosen & (action >= 0.8) & (action < 0.9)
    random_ids = torch.randint(0, vocab_size, ids.shape, generator=generator, device=ids.device)
    inputs[replace] = random_ids[replace]
    return inputs, labels
