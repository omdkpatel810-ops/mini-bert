from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from minibert.data import IGNORE, Corpus
from minibert.model import BertConfig, MaskedLanguageModel, count_parameters
from minibert.pretrain import (
    LOG_FILE,
    TrainSettings,
    evaluate,
    evaluation_set,
    learning_rate_at,
    make_optimizer,
    mini_config,
    pretrain,
    read_log,
)
from minibert.tokenizer import SPECIAL_TOKENS, Tokenizer

VOCAB = [*SPECIAL_TOKENS, *(f"w{n}" for n in range(20))]
TOKENIZER = Tokenizer(VOCAB)
CONFIG = BertConfig(vocab_size=len(VOCAB), hidden_size=32, layers=2, heads=2, intermediate_size=64, max_positions=16)


def cycle_corpus(tmp_path, name: str, length: int = 4_000) -> Corpus:
    """w0 w1 … w19 w0 w1 …: every hidden word can be worked out from its neighbours."""
    path = tmp_path / f"{name}.bin"
    (np.arange(length) % 20 + len(SPECIAL_TOKENS)).astype(np.uint16).tofile(path)
    path.with_suffix(".json").write_text(json.dumps({"vocab_size": len(VOCAB)}))
    return Corpus.open(path)


def runs_corpus(tmp_path, name: str) -> Corpus:
    """Each word repeated 40 times: a hidden word can be copied from its neighbours."""
    path = tmp_path / f"{name}.bin"
    (np.repeat(np.arange(400) % 20, 40) + len(SPECIAL_TOKENS)).astype(np.uint16).tofile(path)
    path.with_suffix(".json").write_text(json.dumps({"vocab_size": len(VOCAB)}))
    return Corpus.open(path)


def settings(**overrides) -> TrainSettings:
    values = dict(
        steps=200,
        batch_size=16,
        sequence_length=16,
        learning_rate=3e-3,
        warmup_steps=20,
        eval_every=100,
        eval_sequences=32,
        save_every=50,
        log_every=50,
    )
    return TrainSettings(**{**values, **overrides})


def quiet(_: str) -> None:
    pass


# The schedule and optimizer ---------------------------------------------------------


def test_learning_rate_warms_up_then_falls_to_zero():
    s = TrainSettings(steps=100, warmup_steps=10, learning_rate=1.0)

    assert learning_rate_at(0, s) == pytest.approx(0.1)
    assert learning_rate_at(9, s) == pytest.approx(1.0)
    assert learning_rate_at(10, s) == pytest.approx(1.0)
    assert learning_rate_at(55, s) == pytest.approx(0.5)
    assert learning_rate_at(100, s) == 0.0


def test_no_weight_decay_on_biases_or_layernorm():
    model = MaskedLanguageModel(CONFIG)
    decay, no_decay = make_optimizer(model, TrainSettings()).param_groups

    assert decay["weight_decay"] == 0.01 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim == 2 for p in decay["params"])
    assert all(p.ndim == 1 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"steps": 0}, "steps and batch_size must be at least 1"),
        ({"steps": 10, "warmup_steps": 11}, "warmup_steps must be between 0 and steps"),
        ({"sequence_length": 2}, "sequence_length must be at least 3"),
    ],
)
def test_settings_are_checked(overrides, message):
    with pytest.raises(ValueError, match=message):
        TrainSettings(**overrides)


def test_mini_is_the_size_of_googles_bert_mini():
    model = MaskedLanguageModel(mini_config(30_522))
    assert (model.config.layers, model.config.hidden_size, model.config.heads) == (4, 256, 4)
    assert 11_000_000 < count_parameters(model) < 11_500_000


# The loss ---------------------------------------------------------------------------


def test_masked_word_loss_matches_the_full_scores():
    torch.manual_seed(0)
    model = MaskedLanguageModel(CONFIG).eval()
    ids = torch.randint(5, len(VOCAB), (3, 10))
    labels = torch.full_like(ids, IGNORE)
    labels[0, 2], labels[1, 5], labels[2, 7] = 7, 9, 11

    result = model.masked_word_loss(ids, labels)
    logits, _ = model(ids)
    chosen = labels != IGNORE

    torch.testing.assert_close(result.loss, F.cross_entropy(logits[chosen], labels[chosen]))
    assert torch.equal(result.predictions, logits[chosen].argmax(-1))
    assert result.correct == int((result.predictions == result.targets).sum())


def test_nothing_hidden_means_zero_loss():
    model = MaskedLanguageModel(CONFIG)
    ids = torch.randint(5, len(VOCAB), (2, 6))

    result = model.masked_word_loss(ids, torch.full_like(ids, IGNORE))
    result.loss.backward()

    assert result.loss.item() == 0.0
    assert result.correct == 0


def test_fused_attention_gives_the_same_results():
    torch.manual_seed(0)
    model = MaskedLanguageModel(CONFIG).eval()
    ids = torch.randint(5, len(VOCAB), (2, 12))
    mask = torch.ones_like(ids)
    mask[1, 8:] = 0

    with torch.no_grad():
        manual, _ = model(ids, attention_mask=mask)
        model.use_fused_attention()
        fused, output = model(ids, attention_mask=mask)
        _, with_weights = model(ids, attention_mask=mask, return_attentions=True)

    real = mask.bool()
    torch.testing.assert_close(fused[real], manual[real], atol=1e-5, rtol=1e-5)
    assert output.attentions is None
    assert with_weights.attentions[0].shape == (2, 2, 12, 12)  # asking for weights uses the written-out path


# Training ---------------------------------------------------------------------------


def test_learns_a_pattern_it_can_work_out(tmp_path):
    torch.manual_seed(0)
    model = MaskedLanguageModel(CONFIG)
    train, valid = runs_corpus(tmp_path, "train"), runs_corpus(tmp_path, "valid")
    before = evaluate(model, *evaluation_set(valid, TOKENIZER, 32, 16))

    result = pretrain(model, TOKENIZER, train, valid, tmp_path / "run", settings(), device="cpu", report=quiet)

    assert before["accuracy"] < 0.2
    assert result["finished"] and result["step"] == 200
    # Well short of 100% after 200 tiny steps, but from chance to most words right.
    assert result["accuracy"] > 0.6
    assert result["loss"] < before["loss"] / 2
    assert (tmp_path / "run" / "model" / "model.safetensors").exists()
    log = read_log(tmp_path / "run")
    assert [r["step"] for r in log if "loss" in r] == [50, 100, 150, 200]
    assert [r["step"] for r in log if "eval_accuracy" in r] == [100, 200]


def test_a_resumed_run_ends_exactly_where_an_uninterrupted_one_does(tmp_path):
    train, valid = cycle_corpus(tmp_path, "train"), cycle_corpus(tmp_path, "valid", 1_000)
    s = settings(steps=6, warmup_steps=2, save_every=2, eval_every=3, log_every=1)

    torch.manual_seed(0)
    straight = MaskedLanguageModel(CONFIG)
    pretrain(straight, TOKENIZER, train, valid, tmp_path / "straight", s, device="cpu", report=quiet)

    messages: list[str] = []
    stops = 0
    while True:
        torch.manual_seed(0)
        resumed = MaskedLanguageModel(CONFIG)
        result = pretrain(
            resumed, TOKENIZER, train, valid, tmp_path / "bumpy", s, device="cpu", max_minutes=0, report=messages.append
        )
        if result["finished"]:
            break
        stops += 1

    assert stops == 5  # one step per run, as if Colab kept disconnecting
    assert "Resuming from step 1." in messages
    assert any(m.startswith("Stopped at step 1 after 0 minutes") for m in messages)
    for (name, a), (_, b) in zip(straight.state_dict().items(), resumed.state_dict().items(), strict=True):
        torch.testing.assert_close(a, b, msg=name)
    assert read_log(tmp_path / "bumpy") == [
        {k: v for k, v in r.items() if k != "tokens_per_second"} | {"tokens_per_second": b["tokens_per_second"]}
        for r, b in zip(read_log(tmp_path / "straight"), read_log(tmp_path / "bumpy"), strict=True)
    ]


def test_resuming_drops_log_lines_after_the_checkpoint(tmp_path):
    train, valid = cycle_corpus(tmp_path, "train"), cycle_corpus(tmp_path, "valid", 1_000)
    s = settings(steps=4, warmup_steps=1, save_every=2, eval_every=10, log_every=1)
    run = tmp_path / "run"

    pretrain(MaskedLanguageModel(CONFIG), TOKENIZER, train, valid, run, s, device="cpu", max_minutes=0, report=quiet)
    with (run / LOG_FILE).open("a") as log:  # as if Colab died after logging step 2, before saving it
        log.write(json.dumps({"step": 2, "loss": 1.0}) + "\n")

    pretrain(MaskedLanguageModel(CONFIG), TOKENIZER, train, valid, run, s, device="cpu", report=quiet)

    assert [r["step"] for r in read_log(run)] == [1, 2, 3, 4]


def test_resuming_with_different_settings_is_refused(tmp_path):
    train, valid = cycle_corpus(tmp_path, "train"), cycle_corpus(tmp_path, "valid", 1_000)
    run = tmp_path / "run"
    pretrain(
        MaskedLanguageModel(CONFIG),
        TOKENIZER,
        train,
        valid,
        run,
        settings(steps=4, warmup_steps=1),
        device="cpu",
        max_minutes=0,
        report=quiet,
    )

    with pytest.raises(ValueError, match="was made with different settings"):
        pretrain(
            MaskedLanguageModel(CONFIG),
            TOKENIZER,
            train,
            valid,
            run,
            settings(steps=8, warmup_steps=1),
            device="cpu",
            report=quiet,
        )


def test_the_vocabularies_must_match(tmp_path):
    train = cycle_corpus(tmp_path, "train")
    other = Tokenizer([*SPECIAL_TOKENS, "x"])

    with pytest.raises(ValueError, match="same vocabulary"):
        pretrain(MaskedLanguageModel(CONFIG), other, train, train, tmp_path / "run", settings(), report=quiet)


def test_a_run_with_no_log_yet():
    assert read_log("/nowhere/at/all") == []
