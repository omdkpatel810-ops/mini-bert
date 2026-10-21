from __future__ import annotations

import io

import pytest

from minibert import __version__
from minibert.cli import main


def run(*args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(args), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"minibert {__version__}"


def test_tokenize_uses_googles_vocabulary_by_default():
    code, out, err = run("tokenize", "Unaffable [MASK]!")

    assert (code, err) == (0, "")
    assert out.splitlines() == [
        "[CLS]      101",
        "una      14477",
        "##ffa    20961",
        "##ble     3468",
        "[MASK]     103",
        "!          999",
        "[SEP]      102",
        "7 tokens",
    ]


def test_tokenize_a_pair_shows_the_sentences():
    _, out, _ = run("tokenize", "Where?", "--pair", "Here.", "--max-length", "6")

    assert out.splitlines()[:3] == [
        "[CLS]     101  sentence A",
        "where    2073  sentence A",
        "?        1029  sentence A",
    ]
    assert out.splitlines()[4] == "here     2182  sentence B"
    assert out.splitlines()[-1] == "6 tokens"


def test_tokenize_cased_with_another_vocabulary(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nHi\n", encoding="utf-8")

    _, out, _ = run("tokenize", "Hi hi", "--vocab", str(vocab), "--cased")

    assert out.splitlines()[1:3] == ["Hi          5", "[UNK]       1"]


def test_train_tokenizer(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hug pug pun bun hugs\n" * 20 + "hug pun x☃\n", encoding="utf-8")
    vocab = tmp_path / "vocab.txt"

    code, out, err = run(
        "train-tokenizer", str(corpus), "--out", str(vocab), "--vocab-size", "40", "--alphabet", "7", "--workers", "1"
    )

    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0].startswith("Counted 103 words (6 different) in ")
    assert lines[1].startswith("Learned 26 tokens (7 merges) in ")
    assert lines[2] == "  Stopped early: no pair left that appears 2+ times."
    assert lines[3] == "  Left out 1 word with rare characters outside the 7-character alphabet; it will become [UNK]."
    assert vocab.read_text(encoding="utf-8").splitlines()[:5] == ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def test_train_tokenizer_with_the_default_workers(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a b a b ab\n", encoding="utf-8")

    code, out, _ = run("train-tokenizer", str(corpus), "--out", str(tmp_path / "v.txt"), "--vocab-size", "1000")

    assert code == 0
    assert "Learned" in out


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["train-tokenizer", "missing.txt", "--out", "v.txt"], "error: No such file: missing.txt"),
        (["tokenize", "hi", "--max-length", "1"], "error: max_length must be at least 2"),
        (["tokenize", "hi", "--vocab", "missing-vocab.txt"], "error: [Errno 2] No such file or directory"),
    ],
)
def test_errors(args, message):
    code, out, err = run(*args)

    assert code == 1
    assert out == ""
    assert err.startswith(message)


def test_training_on_an_empty_file(tmp_path):
    corpus = tmp_path / "empty.txt"
    corpus.write_text("   \n", encoding="utf-8")

    code, _, err = run("train-tokenizer", str(corpus), "--out", str(tmp_path / "v.txt"), "--workers", "1")

    assert code == 1
    assert err.strip() == "error: The files have no words to learn from."


# fill ------------------------------------------------------------------------------


@pytest.fixture
def tiny_model(tmp_path):
    import torch

    from minibert.checkpoint import save
    from minibert.model import BertConfig, MaskedLanguageModel
    from minibert.tokenizer import SPECIAL_TOKENS, Tokenizer

    vocab = [*SPECIAL_TOKENS, "the", "cat", "sat", "mat", "."]
    torch.manual_seed(0)
    model = MaskedLanguageModel(
        BertConfig(vocab_size=len(vocab), hidden_size=16, layers=1, heads=2, intermediate_size=32, max_positions=16)
    )
    return save(model, Tokenizer(vocab), tmp_path / "tiny")


def test_fill_one_blank(tiny_model):
    code, out, err = run("fill", "the [MASK] sat .", "--model", str(tiny_model), "--top", "3")

    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert len(lines) == 3
    assert all(line.startswith("  ") and line.endswith("%") for line in lines)


def test_fill_several_blanks(tiny_model):
    _, out, _ = run("fill", "the [MASK] sat on the [MASK] .", "--model", str(tiny_model), "--top", "2")

    lines = out.splitlines()
    assert lines[0] == "[MASK] 1:"
    assert lines[3] == "[MASK] 2:"
    assert len(lines) == 6


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--top", "0"], "error: --top must be at least 1."),
        ([], "error: Put [MASK] where the missing word goes"),
    ],
)
def test_fill_errors(tiny_model, args, message):
    text = "the cat sat" if not args else "the [MASK] sat"
    code, _, err = run("fill", text, "--model", str(tiny_model), *args)

    assert code == 1
    assert err.startswith(message)


def test_fill_with_a_broken_model(tiny_model):
    from safetensors.torch import load_file, save_file

    weights = tiny_model / "model.safetensors"
    state = load_file(weights)
    del state["bias"]
    save_file(state, weights)

    code, _, err = run("fill", "the [MASK] sat", "--model", str(tiny_model))

    assert code == 1
    assert err.strip() == f"error: {tiny_model} is missing weights: bias"
