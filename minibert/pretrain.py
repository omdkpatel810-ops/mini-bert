"""Pre-train BERT from scratch by guessing hidden words (masked language modelling).

Built for a free Colab GPU, which can disconnect at any time: progress is saved to
a checkpoint every `save_every` steps and running again picks up where it stopped.
Each step's batch and masks come from a random generator seeded by the step number,
so a resumed run sees exactly the data an uninterrupted one would have.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from minibert import checkpoint
from minibert.data import IGNORE, Corpus, fixed_windows, mask_tokens, random_windows
from minibert.model import BertConfig, MaskedLanguageModel
from minibert.tokenizer import Tokenizer

CHECKPOINT_FILE = "checkpoint.pt"
LOG_FILE = "log.jsonl"
EVAL_SEED = 12_345  # the same hidden words every evaluation, so scores are comparable


@dataclass(frozen=True)
class TrainSettings:
    steps: int = 60_000
    batch_size: int = 128
    sequence_length: int = 128
    learning_rate: float = 5e-4
    warmup_steps: int = 3_000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mask_rate: float = 0.15
    eval_every: int = 1_000
    eval_sequences: int = 512
    save_every: int = 1_000
    log_every: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be at least 1.")
        if not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be between 0 and steps.")
        if self.sequence_length < 3:
            raise ValueError("sequence_length must be at least 3, for [CLS], a word and [SEP].")


def learning_rate_at(step: int, settings: TrainSettings) -> float:
    """Warm up linearly to the peak, then fall linearly to zero at the last step, as BERT does."""
    if step < settings.warmup_steps:
        return settings.learning_rate * (step + 1) / settings.warmup_steps
    remaining = settings.steps - step
    return settings.learning_rate * max(0.0, remaining / max(1, settings.steps - settings.warmup_steps))


def make_optimizer(model: torch.nn.Module, settings: TrainSettings) -> torch.optim.AdamW:
    """AdamW with weight decay on weight matrices only, not on biases or LayerNorm, as in BERT."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": settings.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    # beta2 = 0.98 and eps = 1e-6 (RoBERTa's settings) keep half-precision training stable.
    return torch.optim.AdamW(groups, lr=settings.learning_rate, betas=(0.9, 0.98), eps=1e-6)


def special_ids(tokenizer: Tokenizer) -> list[int]:
    return [tokenizer.pad_id, tokenizer.unk_id, tokenizer.cls_id, tokenizer.sep_id, tokenizer.mask_id]


def evaluation_set(corpus: Corpus, tokenizer: Tokenizer, sequences: int, length: int) -> tuple[Tensor, Tensor]:
    """Fixed held-out text with fixed hidden words, for comparing models fairly."""
    ids = fixed_windows(corpus, sequences, length, tokenizer.cls_id, tokenizer.sep_id)
    generator = torch.Generator().manual_seed(EVAL_SEED)
    return mask_tokens(
        ids,
        mask_id=tokenizer.mask_id,
        vocab_size=len(tokenizer),
        special_ids=special_ids(tokenizer),
        generator=generator,
    )


@torch.no_grad()
def evaluate(model: MaskedLanguageModel, inputs: Tensor, labels: Tensor, batch_size: int = 64) -> dict:
    """Average loss over the hidden words, and how often the best guess is the right word."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for start in range(0, len(inputs), batch_size):
        with _autocast(device):
            result = model.masked_word_loss(
                inputs[start : start + batch_size].to(device), labels[start : start + batch_size].to(device)
            )
        n = result.targets.numel()
        total_loss += result.loss.item() * n
        correct += result.correct
        count += n
    model.train(was_training)
    loss = total_loss / max(count, 1)
    return {"loss": loss, "perplexity": math.exp(min(loss, 50)), "accuracy": correct / max(count, 1), "words": count}


def pretrain(
    model: MaskedLanguageModel,
    tokenizer: Tokenizer,
    train: Corpus,
    valid: Corpus,
    run_dir: str | Path,
    settings: TrainSettings,
    *,
    device: torch.device | str | None = None,
    max_minutes: float | None = None,
    report: Callable[[str], None] = print,
) -> dict:
    """Train until `settings.steps`, or until `max_minutes` pass (then run again to continue)."""
    if len(tokenizer) != model.config.vocab_size or train.vocab_size != len(tokenizer):
        raise ValueError("The model, tokenizer and training data must use the same vocabulary.")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).train()
    if device.type == "cuda":
        model.use_fused_attention()
    optimizer = make_optimizer(model, settings)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    step = _resume(run_dir, model, optimizer, scaler, settings, report)
    eval_inputs, eval_labels = evaluation_set(valid, tokenizer, settings.eval_sequences, settings.sequence_length)
    specials = special_ids(tokenizer)
    started = time.monotonic()
    window_started, window_loss, window_steps = time.monotonic(), 0.0, 0

    while step < settings.steps:
        seed = settings.seed * 1_000_003 + step
        torch.manual_seed(seed)  # dropout too, so a resumed run matches an uninterrupted one exactly
        generator = torch.Generator().manual_seed(seed)
        ids = random_windows(
            train, settings.batch_size, settings.sequence_length, tokenizer.cls_id, tokenizer.sep_id, generator
        )
        inputs, labels = mask_tokens(
            ids,
            mask_id=tokenizer.mask_id,
            vocab_size=len(tokenizer),
            special_ids=specials,
            rate=settings.mask_rate,
            generator=generator,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, settings)

        with _autocast(device):
            loss = model.masked_word_loss(inputs.to(device), labels.to(device)).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        window_loss += loss.item()
        window_steps += 1

        record: dict = {}
        if step % settings.log_every == 0 or step == settings.steps:
            elapsed = time.monotonic() - window_started
            tokens_per_second = window_steps * settings.batch_size * settings.sequence_length / max(elapsed, 1e-9)
            seconds_per_step = elapsed / window_steps
            record.update(
                step=step,
                loss=window_loss / window_steps,
                learning_rate=learning_rate_at(step - 1, settings),
                tokens_per_second=tokens_per_second,
            )
            eta = (settings.steps - step) * seconds_per_step / 60
            report(
                f"step {step:>6,}/{settings.steps:,}  loss {record['loss']:.3f}  "
                f"lr {record['learning_rate']:.2e}  {tokens_per_second:,.0f} tokens/s  ~{eta:,.0f} min left"
            )
            window_started, window_loss, window_steps = time.monotonic(), 0.0, 0
        if step % settings.eval_every == 0 or step == settings.steps:
            scores = evaluate(model, eval_inputs, eval_labels)
            record.update(step=step, eval_loss=scores["loss"], eval_accuracy=scores["accuracy"])
            report(
                f"  held-out text: loss {scores['loss']:.3f}, guesses {scores['accuracy']:.1%} of hidden words right"
            )
        if record:
            _log(run_dir, record)
        out_of_time = max_minutes is not None and (time.monotonic() - started) / 60 >= max_minutes
        if step % settings.save_every == 0 or step == settings.steps or out_of_time:
            _save_checkpoint(run_dir, model, optimizer, scaler, settings, step)
        if out_of_time and step < settings.steps:
            report(f"Stopped at step {step:,} after {max_minutes:g} minutes. Run again to continue.")
            break

    finished = step >= settings.steps
    if finished:
        checkpoint.save(model.cpu(), tokenizer, run_dir / "model")
        model.to(device)
        report(f"Finished {settings.steps:,} steps. Model saved to {run_dir / 'model'}.")
    return {"step": step, "finished": finished, **evaluate(model, eval_inputs, eval_labels)}


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _resume(run_dir: Path, model, optimizer, scaler, settings: TrainSettings, report) -> int:
    path = run_dir / CHECKPOINT_FILE
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["settings"] != asdict(settings) or state["model_config"] != model.config.to_dict():
        raise ValueError(
            f"{path} was made with different settings. Use a new run folder, or the same settings to continue."
        )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    # Drop log lines written after the checkpoint: those steps are about to be redone.
    log = [record for record in read_log(run_dir) if record["step"] <= state["step"]]
    (run_dir / LOG_FILE).write_text("".join(json.dumps(record) + "\n" for record in log), encoding="utf-8")
    report(f"Resuming from step {state['step']:,}.")
    return state["step"]


def _save_checkpoint(run_dir: Path, model, optimizer, scaler, settings: TrainSettings, step: int) -> None:
    state = {
        "step": step,
        "settings": asdict(settings),
        "model_config": model.config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    temporary = run_dir / (CHECKPOINT_FILE + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, run_dir / CHECKPOINT_FILE)  # all or nothing, even if Colab cuts out mid-save


def _log(run_dir: Path, record: dict) -> None:
    with (run_dir / LOG_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def read_log(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / LOG_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mini_config(vocab_size: int, max_positions: int = 128) -> BertConfig:
    """BERT-Mini's shape (4 layers, 256 wide, 4 heads), Google's smallest-but-one published BERT."""
    return BertConfig(
        vocab_size=vocab_size, hidden_size=256, layers=4, heads=4, intermediate_size=1_024, max_positions=max_positions
    )


__all__ = [
    "IGNORE",
    "TrainSettings",
    "evaluate",
    "evaluation_set",
    "learning_rate_at",
    "make_optimizer",
    "mini_config",
    "pretrain",
    "read_log",
]
