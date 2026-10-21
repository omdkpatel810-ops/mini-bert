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


# prepare-data, pretrain, evaluate -------------------------------------------------


@pytest.fixture
def small_vocab(tmp_path):
    from minibert.tokenizer import SPECIAL_TOKENS

    path = tmp_path / "vocab.txt"
    path.write_text("\n".join([*SPECIAL_TOKENS, "=", "the", "cat", "sat", "dog", "ran", "."]) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def prepared(tmp_path, small_vocab):
    text = tmp_path / "wiki.txt"
    text.write_text(
        "= The Cat =\n" + "the cat sat . the dog ran .\n" * 200 + "= The Dog =\nthe dog ran .\n", encoding="utf-8"
    )
    code, out, err = run(
        "prepare-data",
        str(text),
        "--out",
        str(tmp_path / "train.bin"),
        "--wikitext",
        "--vocab",
        str(small_vocab),
        "--workers",
        "1",
    )
    assert (code, err) == (0, "")
    return tmp_path / "train.bin", out


def test_prepare_data(prepared):
    path, out = prepared

    assert out.startswith(f"Wrote 1,613 tokens from 203 lines (2 documents) to {path} in ")
    assert path.exists() and path.with_suffix(".json").exists()


def test_prepare_data_with_the_default_vocabulary_and_workers(tmp_path):
    text = tmp_path / "plain.txt"
    text.write_text("Hello world.\nSecond line.\n", encoding="utf-8")

    code, out, _ = run("prepare-data", str(text), "--out", str(tmp_path / "plain.bin"))

    assert code == 0
    assert out.startswith("Wrote 8 tokens from 2 lines (2 documents)")


def test_pretrain_then_evaluate(tmp_path, prepared, small_vocab):
    path, _ = prepared
    run_dir = tmp_path / "run"
    args = ["--train", str(path), "--valid", str(path), "--run", str(run_dir), "--vocab", str(small_vocab)]
    shape = ["--layers", "1", "--hidden", "16", "--heads", "2", "--max-positions", "16", "--sequence-length", "16"]
    schedule = ["--steps", "20", "--batch-size", "4", "--warmup", "2", "--eval-every", "10", "--save-every", "10"]

    code, out, err = run("pretrain", *args, *shape, *schedule, "--device", "cpu")

    assert (code, err) == (0, "")
    assert "held-out text: loss" in out
    assert f"Finished 20 steps. Model saved to {run_dir / 'model'}." in out
    assert out.strip().splitlines()[-1].startswith("Held-out text: guesses ")

    code, out, _ = run("evaluate", str(run_dir / "model"), "--valid", str(path), "--sequences", "8")
    assert code == 0
    assert out.startswith(f"{run_dir / 'model'}: guesses ")
    assert "hidden words right (loss " in out


def test_pretrain_sequence_must_fit_the_positions(tmp_path, prepared, small_vocab):
    path, _ = prepared

    code, _, err = run(
        "pretrain",
        "--train",
        str(path),
        "--valid",
        str(path),
        "--run",
        str(tmp_path / "r"),
        "--vocab",
        str(small_vocab),
        "--max-positions",
        "64",
        "--sequence-length",
        "128",
    )

    assert code == 1
    assert err.strip() == "error: --sequence-length can't be longer than --max-positions."


def test_evaluate_needs_the_same_vocabulary(tmp_path, prepared, tiny_model):
    path, _ = prepared

    code, _, err = run("evaluate", str(tiny_model), "--valid", str(path))

    assert code == 1
    assert "was prepared with a different vocabulary" in err


def test_evaluate_a_broken_model(tmp_path, prepared, tiny_model):
    from safetensors.torch import load_file, save_file

    weights = tiny_model / "model.safetensors"
    state = load_file(weights)
    del state["bias"]
    save_file(state, weights)

    code, _, err = run("evaluate", str(tiny_model), "--valid", str(prepared[0]))

    assert code == 1
    assert "is missing weights: bias" in err


def test_prepare_data_needs_a_file(tmp_path):
    code, _, err = run("prepare-data", str(tmp_path / "missing.txt"), "--out", str(tmp_path / "x.bin"))

    assert code == 1
    assert err.strip() == f"error: No such file: {tmp_path / 'missing.txt'}"


# finetune and classify ------------------------------------------------------------


@pytest.fixture
def reviews(tmp_path):
    rows = ["sentence\tlabel"] + [f"the cat sat {'.' if i % 2 else 'mat'}\t{i % 2}" for i in range(40)]
    train = tmp_path / "train.tsv"
    train.write_text("\n".join(rows) + "\n", encoding="utf-8")
    valid = tmp_path / "valid.tsv"
    valid.write_text("sentence\tlabel\nthe cat sat .\t1\nthe mat\t0\n", encoding="utf-8")
    return train, valid


def test_finetune_then_classify(tmp_path, tiny_model, reviews):
    train, valid = reviews
    code, out, err = run(
        "finetune", "--model", str(tiny_model), "--train", str(train), "--valid", str(valid),
        "--out", str(tmp_path / "clf"), "--epochs", "2", "--batch-size", "8", "--labels", "no, yes", "--device", "cpu",
    )  # fmt: skip

    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0] == f"Fine-tuning {tiny_model} on 40 examples, checking on 2."
    assert lines[1].startswith("epoch 1/2  training loss ")
    assert lines[-1].startswith("Validation: ") and lines[-1].endswith(f"Saved to {tmp_path / 'clf'}.")

    code, out, err = run("classify", "the cat sat .", "the mat", "--model", str(tmp_path / "clf"))

    assert (code, err) == (0, "")
    labels = [line.split(" (")[0] for line in out.splitlines()]
    assert len(labels) == 2 and set(labels) <= {"no", "yes"}
    assert out.splitlines()[0].endswith("%)  the cat sat .")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--train", "train.tsv"], "error: Give both --train and --valid, or neither to use SST-2."),
        (["--train", "nowhere.tsv", "--valid", "nowhere.tsv"], "error: No such file: nowhere.tsv"),
    ],
)
def test_finetune_errors(tmp_path, tiny_model, reviews, monkeypatch, args, message):
    monkeypatch.chdir(tmp_path)
    code, _, err = run("finetune", "--model", str(tiny_model), "--out", "clf", *args)

    assert code == 1
    assert err.strip() == message


def test_finetune_with_a_broken_model(tmp_path, tiny_model, reviews):
    from safetensors.torch import load_file, save_file

    weights = tiny_model / "model.safetensors"
    state = load_file(weights)
    del state["bias"]
    save_file(state, weights)
    train, valid = reviews

    code, _, err = run(
        "finetune", "--model", str(tiny_model), "--train", str(train), "--valid", str(valid), "--out", "x"
    )

    assert code == 1
    assert err.strip() == f"error: {tiny_model} is missing weights: bias"


def test_classify_needs_a_classifier(tmp_path, tiny_model):
    assert run("classify", "hi", "--model", str(tmp_path / "missing"))[2].strip().startswith("error: No such folder")
    code, _, err = run("classify", "hi", "--model", str(tiny_model))
    assert code == 1
    assert "isn't a fine-tuned classifier" in err
