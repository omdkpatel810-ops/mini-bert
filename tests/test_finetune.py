from __future__ import annotations

import random

import pytest
import torch

from minibert import checkpoint
from minibert.finetune import (
    SST2_LABELS,
    Example,
    FinetuneSettings,
    SentenceClassifier,
    classify,
    evaluate_classifier,
    finetune,
    learning_rate_at,
    load_classifier,
    read_examples,
    save_classifier,
    sst2,
)
from minibert.model import BertConfig, MaskedLanguageModel
from minibert.tokenizer import SPECIAL_TOKENS, Tokenizer

FILLER = ["the", "film", "was", "very", "quite", "story"]
VOCAB = [*SPECIAL_TOKENS, *FILLER, "good", "bad"]
TOKENIZER = Tokenizer(VOCAB, special_in_text=False)
CONFIG = BertConfig(vocab_size=len(VOCAB), hidden_size=32, layers=2, heads=2, intermediate_size=64, max_positions=16)


def reviews(count: int, seed: int) -> list[Example]:
    """Short made-up reviews: positive exactly when "good" appears, somewhere among filler words."""
    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        words = rng.choices(FILLER, k=rng.randint(2, 8))
        label = rng.randint(0, 1)
        words.insert(rng.randint(0, len(words)), "good" if label else "bad")
        examples.append(Example(" ".join(words), label))
    return examples


def pretrained(seed: int = 0) -> MaskedLanguageModel:
    torch.manual_seed(seed)
    return MaskedLanguageModel(CONFIG)


def quiet(_: str) -> None:
    pass


SETTINGS = FinetuneSettings(epochs=3, batch_size=16, learning_rate=3e-3, max_length=16)


# The classifier -----------------------------------------------------------------------


def test_one_score_per_class():
    model = SentenceClassifier(CONFIG, ["a", "b", "c"])
    ids = torch.randint(5, len(VOCAB), (4, 7))

    assert model(ids).shape == (4, 3)


def test_needs_at_least_two_labels():
    with pytest.raises(ValueError, match="at least two labels"):
        SentenceClassifier(CONFIG, ["only"])


def test_starts_from_a_copy_of_the_pretrained_encoder():
    mlm = pretrained()
    model = SentenceClassifier.from_pretrained(mlm, SST2_LABELS)

    for name, tensor in mlm.bert.state_dict().items():
        assert torch.equal(model.bert.state_dict()[name], tensor), name
    with torch.no_grad():
        model.bert.embeddings.words.weight.add_(1.0)
    assert not torch.equal(model.bert.embeddings.words.weight, mlm.bert.embeddings.words.weight)


def test_the_new_head_depends_only_on_the_seed():
    mlm = pretrained()
    state = torch.random.get_rng_state()

    first = SentenceClassifier.from_pretrained(mlm, SST2_LABELS, seed=3)
    again = SentenceClassifier.from_pretrained(mlm, SST2_LABELS, seed=3)
    other = SentenceClassifier.from_pretrained(mlm, SST2_LABELS, seed=4)

    assert torch.equal(first.dense.weight, again.dense.weight)
    assert not torch.equal(first.dense.weight, other.dense.weight)
    assert torch.equal(torch.random.get_rng_state(), state)  # making a head doesn't disturb other randomness


def test_padding_doesnt_change_the_answer():
    model = SentenceClassifier.from_pretrained(pretrained(), SST2_LABELS, seed=0)
    short = "the film was good"

    alone = classify(model, TOKENIZER, [short])[0]
    padded = classify(model, TOKENIZER, [short, "the story was quite very bad film the film"])[0]

    assert alone.label == padded.label
    assert alone.probability == pytest.approx(padded.probability, abs=1e-5)


# Training -----------------------------------------------------------------------------


def test_learning_rate_warms_up_then_falls_to_zero():
    assert learning_rate_at(0, 100, 10, 1.0) == pytest.approx(0.1)
    assert learning_rate_at(9, 100, 10, 1.0) == pytest.approx(1.0)
    assert learning_rate_at(55, 100, 10, 1.0) == pytest.approx(0.5)
    assert learning_rate_at(100, 100, 10, 1.0) == 0.0
    assert learning_rate_at(0, 100, 0, 1.0) == pytest.approx(1.0)  # no warm-up


def test_learns_to_tell_good_from_bad():
    model = SentenceClassifier.from_pretrained(pretrained(), SST2_LABELS, seed=0)
    train, valid = reviews(400, seed=1), reviews(100, seed=2)
    before = evaluate_classifier(model, TOKENIZER, valid, max_length=16)

    result = finetune(model, TOKENIZER, train, valid, SETTINGS, device="cpu", report=quiet)

    assert before["accuracy"] < 0.7
    assert result["accuracy"] == 1.0
    assert result["accuracy"] == result["history"][-1]["valid_accuracy"]  # the last epoch, not the best
    assert [r["epoch"] for r in result["history"]] == [1, 2, 3]
    assert result["history"][-1]["train_loss"] < result["history"][0]["train_loss"]
    assert not model.training
    predictions = classify(model, TOKENIZER, ["the film was good", "the story was bad", "bad very bad"])
    assert [p.label for p in predictions] == ["positive", "negative", "negative"]
    assert all(p.probability > 0.5 for p in predictions)


def test_the_same_seed_gives_the_same_model():
    train, valid = reviews(64, seed=1), reviews(16, seed=2)
    settings = FinetuneSettings(epochs=1, batch_size=16, learning_rate=3e-3, max_length=16, seed=5)
    runs = []
    for _ in range(2):
        model = SentenceClassifier.from_pretrained(pretrained(), SST2_LABELS, seed=5)
        result = finetune(model, TOKENIZER, train, valid, settings, device="cpu", report=quiet)
        runs.append((model.state_dict(), result["history"]))

    (first, first_history), (second, second_history) = runs
    assert first_history == second_history
    for name in first:
        assert torch.equal(first[name], second[name]), name


def test_reports_each_epoch():
    lines: list[str] = []
    model = SentenceClassifier.from_pretrained(pretrained(), SST2_LABELS, seed=0)
    settings = FinetuneSettings(epochs=2, batch_size=16, max_length=16)

    finetune(model, TOKENIZER, reviews(32, 1), reviews(8, 2), settings, device="cpu", report=lines.append)

    assert len(lines) == 2 and lines[0].startswith("epoch 1/2  training loss ")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"epochs": 0}, "epochs and batch_size must be at least 1"),
        ({"warmup_fraction": 1.0}, "warmup_fraction must be at least 0"),
        ({"max_length": 2}, "max_length must be at least 3"),
    ],
)
def test_settings_are_checked(overrides, message):
    with pytest.raises(ValueError, match=message):
        FinetuneSettings(**overrides)


@pytest.mark.parametrize(
    ("tokenizer", "train", "valid", "message"),
    [
        (Tokenizer([*SPECIAL_TOKENS, "x"]), reviews(4, 1), reviews(4, 2), "same vocabulary"),
        (TOKENIZER, [Example("good", 2)], reviews(4, 2), "Label 2 isn't one of the model's 2 classes"),
        (TOKENIZER, [], reviews(4, 2), "needs training and validation examples"),
    ],
)
def test_bad_inputs_are_refused(tokenizer, train, valid, message):
    model = SentenceClassifier(CONFIG, SST2_LABELS)
    with pytest.raises(ValueError, match=message):
        finetune(model, tokenizer, train, valid, SETTINGS, device="cpu", report=quiet)


def test_classifying_nothing():
    assert classify(SentenceClassifier(CONFIG, SST2_LABELS), TOKENIZER, []) == []


# Saving -------------------------------------------------------------------------------


def test_a_saved_classifier_loads_back_the_same(tmp_path):
    model = SentenceClassifier.from_pretrained(pretrained(), ["no", "yes", "maybe"], seed=0).eval()
    ids = torch.randint(5, len(VOCAB), (2, 6))

    save_classifier(model, TOKENIZER, tmp_path / "clf")
    loaded, tokenizer = load_classifier(tmp_path / "clf")

    assert loaded.labels == ("no", "yes", "maybe")
    assert tokenizer.vocab == VOCAB
    torch.testing.assert_close(loaded(ids), model(ids))


def test_a_pretrained_model_is_not_a_classifier(tmp_path):
    checkpoint.save(pretrained(), TOKENIZER, tmp_path / "plain")

    with pytest.raises(ValueError, match="isn't a fine-tuned classifier"):
        load_classifier(tmp_path / "plain")


# Data ---------------------------------------------------------------------------------


def test_reads_glue_style_tsv(tmp_path):
    path = tmp_path / "train.tsv"
    path.write_text('sentence\tlabel\nit \'s a "charming" film \t1\na mess\t0\n', encoding="utf-8")

    assert read_examples(path) == [Example('it \'s a "charming" film', 1), Example("a mess", 0)]


def test_reads_a_text_column_too(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("label\ttext\n2\tsomething\n", encoding="utf-8")

    assert read_examples(path) == [Example("something", 2)]


@pytest.mark.parametrize(
    ("content", "message"),
    [("words\tscore\nhi\t1\n", "needs a header row"), ("sentence\tlabel\n", "has no examples")],
)
def test_unreadable_files(tmp_path, content, message):
    path = tmp_path / "bad.tsv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_examples(path)


def test_reads_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "sst2.parquet"
    pq.write_table(pa.table({"sentence": ["fine ", "awful"], "label": [1, 0], "idx": [0, 1]}), path)

    assert read_examples(path) == [Example("fine", 1), Example("awful", 0)]


def test_sst2_test_labels_are_secret():
    with pytest.raises(ValueError, match="kept secret"):
        sst2("test")


@pytest.mark.network
def test_downloads_sst2_validation():
    pytest.importorskip("pyarrow")
    examples = sst2("validation")

    assert len(examples) == 872
    assert {e.label for e in examples} == {0, 1}


def test_fused_attention_gives_the_same_scores():
    model = SentenceClassifier.from_pretrained(pretrained(), SST2_LABELS, seed=0).eval()
    ids = torch.randint(5, len(VOCAB), (2, 9))
    mask = torch.ones_like(ids)
    mask[1, 6:] = 0

    with torch.no_grad():
        written_out = model(ids, attention_mask=mask)
        model.use_fused_attention()
        fused = model(ids, attention_mask=mask)

    torch.testing.assert_close(fused, written_out, atol=1e-5, rtol=1e-5)
