from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from minibert.data import (
    IGNORE,
    Corpus,
    clean_wikitext,
    fixed_windows,
    is_article_title,
    mask_tokens,
    prepare_corpus,
    random_windows,
)
from minibert.tokenizer import SPECIAL_TOKENS, Tokenizer

WORDS = ["the", "cat", "sat", "on", "mat", "dog", "ran", ".", ",", "2", "500", "-", "guest", "starring", "="]
TOKENIZER = Tokenizer([*SPECIAL_TOKENS, *WORDS], special_in_text=False)
ID = TOKENIZER.token_to_id

WIKITEXT = [
    "= The Cat =",
    "The cat sat on the mat .",
    "= = Later = =",
    "The 2 @,@ 500 guest @-@ starring dog ran .",
    "",
    "= The Dog =",
    "The dog ran .",
]


def test_clean_wikitext():
    assert clean_wikitext(" The 2 @,@ 500 guest @-@ starring role , 1 @.@ 5 hours ") == (
        "The 2,500 guest-starring role , 1.5 hours"
    )


@pytest.mark.parametrize(
    ("line", "title"),
    [("= Robert Boulter =", True), (" = Du Fu = ", True), ("= = Career = =", False), ("Plain text", False)],
)
def test_is_article_title(line, title):
    assert is_article_title(line) is title


def test_prepares_wikitext_with_sep_between_articles(tmp_path):
    counts = prepare_corpus(WIKITEXT, TOKENIZER, tmp_path / "train.bin", wikitext=True)

    corpus = Corpus.open(tmp_path / "train.bin")
    tokens = [TOKENIZER.vocab[i] for i in corpus.tokens]
    assert tokens == [
        *"= the cat =".split(),
        *"the cat sat on the mat .".split(),
        *"= = [UNK] = =".split(),
        *"the 2 , 500 guest - starring dog ran .".split(),
        "[SEP]",
        *"= the dog =".split(),
        *"the dog ran .".split(),
    ]
    assert counts == {"vocab_size": len(TOKENIZER), "tokens": len(tokens), "documents": 2, "lines": 7}
    assert json.loads((tmp_path / "train.json").read_text()) == counts
    assert corpus.vocab_size == len(TOKENIZER)
    assert len(corpus) == len(tokens)


def test_plain_text_has_sep_after_every_line(tmp_path):
    prepare_corpus(["the cat sat", "the dog ran"], TOKENIZER, tmp_path / "plain.bin")

    tokens = [TOKENIZER.vocab[i] for i in Corpus.open(tmp_path / "plain.bin").tokens]
    assert tokens == ["the", "cat", "sat", "[SEP]", "the", "dog", "ran", "[SEP]"]


def test_tokenizing_in_parallel_gives_the_same_file(tmp_path):
    lines = WIKITEXT * 3_000  # several batches, so order across workers matters

    prepare_corpus(lines, TOKENIZER, tmp_path / "one.bin", wikitext=True)
    prepare_corpus(lines, TOKENIZER, tmp_path / "two.bin", wikitext=True, workers=2)

    assert (tmp_path / "one.bin").read_bytes() == (tmp_path / "two.bin").read_bytes()


def test_vocabularies_must_fit_in_two_bytes(tmp_path):
    big = Tokenizer([*SPECIAL_TOKENS, *(f"w{n}" for n in range(70_000))])
    with pytest.raises(ValueError, match="don't fit in 2 bytes"):
        prepare_corpus(["w1"], big, tmp_path / "big.bin")


@pytest.fixture
def corpus(tmp_path) -> Corpus:
    path = tmp_path / "numbers.bin"
    np.arange(1_000, 1_500, dtype=np.uint16).tofile(path)
    path.with_suffix(".json").write_text(json.dumps({"vocab_size": 2_000}))
    return Corpus.open(path)


def test_random_windows_are_real_stretches_of_the_corpus(corpus):
    batch = random_windows(corpus, 8, 10, cls_id=2, sep_id=3, generator=torch.Generator().manual_seed(0))

    assert batch.shape == (8, 10)
    assert (batch[:, 0] == 2).all() and (batch[:, -1] == 3).all()
    for row in batch[:, 1:-1]:
        assert torch.equal(row, torch.arange(int(row[0]), int(row[0]) + 8))  # consecutive corpus tokens


def test_random_windows_depend_only_on_the_generator(corpus):
    first = random_windows(corpus, 4, 10, 2, 3, torch.Generator().manual_seed(7))
    second = random_windows(corpus, 4, 10, 2, 3, torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


def test_fixed_windows_are_back_to_back(corpus):
    batch = fixed_windows(corpus, 3, 12, cls_id=2, sep_id=3)

    assert torch.equal(batch[:, 1:-1].flatten(), torch.arange(1_000, 1_030))
    assert len(fixed_windows(corpus, 1_000, 12, 2, 3)) == 50  # only as many as the corpus holds


@pytest.mark.parametrize("make", [random_windows, fixed_windows])
def test_windows_need_enough_text(tmp_path, make):
    path = tmp_path / "tiny.bin"
    np.arange(5, dtype=np.uint16).tofile(path)
    path.with_suffix(".json").write_text(json.dumps({"vocab_size": 10}))
    extra = (torch.Generator(),) if make is random_windows else ()

    with pytest.raises(ValueError, match="fewer than one window of 8"):
        make(Corpus.open(path), 1, 10, 2, 3, *extra)


def test_masking_follows_berts_recipe():
    ids = torch.randint(5, 1_000, (200, 128), generator=torch.Generator().manual_seed(0))
    ids[:, 0], ids[:, -1] = 2, 3  # [CLS] and [SEP]

    inputs, labels = mask_tokens(
        ids, mask_id=4, vocab_size=1_000, special_ids=[0, 1, 2, 3, 4], generator=torch.Generator().manual_seed(1)
    )

    chosen = labels != IGNORE
    assert not chosen[:, 0].any() and not chosen[:, -1].any()  # special tokens are never hidden
    assert torch.equal(labels[chosen], ids[chosen])
    assert torch.equal(inputs[~chosen], ids[~chosen])
    share = chosen.float().mean().item()
    assert 0.14 < share < 0.16
    masked = (inputs[chosen] == 4).float().mean().item()
    kept = (inputs[chosen] == ids[chosen]).float().mean().item()
    assert 0.77 < masked < 0.83
    assert 0.08 < kept < 0.12  # 10% kept, plus the odd random token that happens to match
    assert 0.08 < 1 - masked - kept < 0.12  # replaced by a random token


def test_masking_is_reproducible_and_can_be_switched_off():
    ids = torch.randint(5, 100, (4, 20))
    first = mask_tokens(ids, mask_id=4, vocab_size=100, special_ids=[0], generator=torch.Generator().manual_seed(3))
    second = mask_tokens(ids, mask_id=4, vocab_size=100, special_ids=[0], generator=torch.Generator().manual_seed(3))
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))

    inputs, labels = mask_tokens(ids, mask_id=4, vocab_size=100, special_ids=[0], rate=0.0)
    assert torch.equal(inputs, ids)
    assert (labels == IGNORE).all()
