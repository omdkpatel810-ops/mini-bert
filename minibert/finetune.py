"""Fine-tune BERT to sort sentences into classes, such as positive or negative movie reviews (SST-2).

A small head goes on top of the [CLS] token's vector, and the whole model trains on
labelled sentences for a few epochs. The head is new for every model, so two encoders
can be compared fairly: BERT's pooler is learned by the next-sentence task, which this
project skips (as RoBERTa does), so the classifier doesn't use it.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F

from minibert.checkpoint import CONFIG_FILE, VOCAB_FILE, WEIGHTS_FILE
from minibert.model import Bert, BertConfig, MaskedLanguageModel, _init_weights
from minibert.tokenizer import Tokenizer

SST2_LABELS = ("negative", "positive")
SST2_REPO = "nyu-mll/glue"


@dataclass(frozen=True)
class Example:
    text: str
    label: int


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float


@dataclass(frozen=True)
class FinetuneSettings:
    epochs: int = 4  # Google's recipe for its small BERTs
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_fraction: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be at least 1.")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be at least 0 and less than 1.")
        if self.max_length < 3:
            raise ValueError("max_length must be at least 3, for [CLS], a word and [SEP].")


class SentenceClassifier(nn.Module):
    """BERT plus a small head that turns the [CLS] vector into one score per class."""

    def __init__(self, config: BertConfig, labels: Sequence[str], bert: Bert | None = None) -> None:
        super().__init__()
        if len(labels) < 2:
            raise ValueError("A classifier needs at least two labels.")
        self.config = config
        self.labels = tuple(labels)
        self.bert = bert if bert is not None else Bert(config)
        self.dropout = nn.Dropout(config.dropout)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, len(self.labels))
        _init_weights(self.dense)
        _init_weights(self.out)

    @classmethod
    def from_pretrained(
        cls, model: MaskedLanguageModel, labels: Sequence[str], seed: int | None = None
    ) -> SentenceClassifier:
        """A classifier on a copy of a pre-trained encoder, with a new head (made from `seed`, if given)."""
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            return cls(model.config, labels, copy.deepcopy(model.bert))

    def forward(self, ids: Tensor, segment_ids: Tensor | None = None, attention_mask: Tensor | None = None) -> Tensor:
        """One score per class for each input: (batch, labels)."""
        first = self.bert(ids, segment_ids, attention_mask).hidden[:, 0]  # the [CLS] token's vector
        x = torch.tanh(self.dense(self.dropout(first)))
        return self.out(self.dropout(x))

    def use_fused_attention(self, enabled: bool = True) -> None:
        self.bert.use_fused_attention(enabled)


# Data ---------------------------------------------------------------------------------


def read_examples(path: str | Path) -> list[Example]:
    """Labelled sentences from a .tsv with a header row (sentence and label columns, as GLUE ships SST-2) or a .parquet."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        text_column = "sentence" if "sentence" in table.column_names else "text"
        rows = zip(table.column(text_column).to_pylist(), table.column("label").to_pylist(), strict=True)
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
        columns = records[0].keys() if records else set()
        if records and ("label" not in columns or not {"sentence", "text"} & columns):
            raise ValueError(f"{path} needs a header row with 'sentence' (or 'text') and 'label' columns.")
        text_column = "sentence" if records and "sentence" in records[0] else "text"
        rows = ((record.get(text_column) or "", record["label"]) for record in records)
    examples = [Example(text.strip(), int(label)) for text, label in rows]
    if not examples:
        raise ValueError(f"{path} has no examples.")
    return examples


def sst2(split: str) -> list[Example]:
    """SST-2 from GLUE, downloaded from the Hugging Face Hub: "train" (67,349) or "validation" (872)."""
    from huggingface_hub import hf_hub_download

    if split not in ("train", "validation"):
        raise ValueError('Use split "train" or "validation"; SST-2\'s test labels are kept secret by GLUE.')
    return read_examples(hf_hub_download(SST2_REPO, f"sst2/{split}-00000-of-00001.parquet", repo_type="dataset"))


def _encode(tokenizer: Tokenizer, examples: Sequence[Example], max_length: int) -> list[tuple[list[int], int]]:
    return [(tokenizer.encode(example.text, max_length=max_length).ids, example.label) for example in examples]


def _batch(items: Sequence[tuple[list[int], int]], pad_id: int) -> tuple[Tensor, Tensor, Tensor]:
    """Pad to the longest input in the batch: (ids, attention mask, labels)."""
    longest = max(len(ids) for ids, _ in items)
    ids = torch.full((len(items), longest), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, (tokens, _) in enumerate(items):
        ids[row, : len(tokens)] = torch.tensor(tokens)
        mask[row, : len(tokens)] = 1
    return ids, mask, torch.tensor([label for _, label in items])


# Training -----------------------------------------------------------------------------


def learning_rate_at(step: int, total: int, warmup: int, peak: float) -> float:
    """Warm up linearly to the peak, then fall linearly to zero at the last step."""
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * max(0.0, (total - step) / max(1, total - warmup))


def finetune(
    model: SentenceClassifier,
    tokenizer: Tokenizer,
    train: Sequence[Example],
    valid: Sequence[Example],
    settings: FinetuneSettings,
    *,
    device: torch.device | str | None = None,
    report: Callable[[str], None] = print,
) -> dict:
    """Train the whole model on `train`, scoring it on `valid` after each epoch.

    The returned accuracy is the last epoch's, not the best one's, so `valid` isn't
    used to pick a result.
    """
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("The model and the tokenizer must use the same vocabulary.")
    for example in (*train, *valid):
        if not 0 <= example.label < len(model.labels):
            raise ValueError(f"Label {example.label} isn't one of the model's {len(model.labels)} classes.")
    if not train or not valid:
        raise ValueError("Fine-tuning needs training and validation examples.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).train()
    if device.type == "cuda":
        model.use_fused_attention()

    train_items = _encode(tokenizer, train, settings.max_length)
    valid_items = _encode(tokenizer, valid, settings.max_length)
    total = math.ceil(len(train_items) / settings.batch_size) * settings.epochs
    warmup = round(total * settings.warmup_fraction)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": settings.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=settings.learning_rate,
        eps=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    shuffler = torch.Generator().manual_seed(settings.seed)
    torch.manual_seed(settings.seed)  # dropout

    history, step, started = [], 0, time.monotonic()
    for epoch in range(1, settings.epochs + 1):
        order = torch.randperm(len(train_items), generator=shuffler).tolist()
        running = torch.zeros((), device=device)
        for start in range(0, len(order), settings.batch_size):
            ids, mask, labels = _batch(
                [train_items[i] for i in order[start : start + settings.batch_size]], tokenizer.pad_id
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(step, total, warmup, settings.learning_rate)
            with _autocast(device):
                logits = model(ids.to(device), attention_mask=mask.to(device))
            loss = F.cross_entropy(logits.float(), labels.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            running += loss.detach() * len(labels)
            step += 1
        scores = _evaluate(model, valid_items, tokenizer.pad_id)
        model.train()
        history.append(
            {
                "epoch": epoch,
                "train_loss": running.item() / len(train_items),
                "valid_loss": scores["loss"],
                "valid_accuracy": scores["accuracy"],
            }
        )
        report(
            f"epoch {epoch}/{settings.epochs}  training loss {history[-1]['train_loss']:.3f}  "
            f"validation: loss {scores['loss']:.3f}, {scores['accuracy']:.1%} right"
        )
    model.eval()
    return {
        "accuracy": history[-1]["valid_accuracy"],
        "loss": history[-1]["valid_loss"],
        "history": history,
        "seconds": time.monotonic() - started,
        "settings": asdict(settings),
    }


def evaluate_classifier(
    model: SentenceClassifier, tokenizer: Tokenizer, examples: Sequence[Example], max_length: int = 128
) -> dict:
    """How often the model picks the right class, and its average loss."""
    return _evaluate(model, _encode(tokenizer, examples, max_length), tokenizer.pad_id)


@torch.no_grad()
def _evaluate(
    model: SentenceClassifier, items: Sequence[tuple[list[int], int]], pad_id: int, batch_size: int = 128
) -> dict:
    device = next(model.parameters()).device
    model.eval()
    total_loss, correct = 0.0, 0
    for start in range(0, len(items), batch_size):
        ids, mask, labels = _batch(items[start : start + batch_size], pad_id)
        with _autocast(device):
            logits = model(ids.to(device), attention_mask=mask.to(device)).float()
        labels = labels.to(device)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += int((logits.argmax(dim=-1) == labels).sum())
    return {"loss": total_loss / len(items), "accuracy": correct / len(items), "examples": len(items)}


@torch.no_grad()
def classify(model: SentenceClassifier, tokenizer: Tokenizer, texts: Sequence[str]) -> list[Prediction]:
    """The most likely class for each text, and how sure the model is."""
    if not texts:
        return []
    items = _encode(tokenizer, [Example(text, 0) for text in texts], model.config.max_positions)
    model.eval()
    device = next(model.parameters()).device
    ids, mask, _ = _batch(items, tokenizer.pad_id)
    probabilities = model(ids.to(device), attention_mask=mask.to(device)).float().softmax(dim=-1)
    best = probabilities.max(dim=-1)
    return [Prediction(model.labels[i], p) for p, i in zip(best.values.tolist(), best.indices.tolist(), strict=True)]


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# Saving -------------------------------------------------------------------------------


def save_classifier(model: SentenceClassifier, tokenizer: Tokenizer, folder: str | Path) -> Path:
    """Save a fine-tuned classifier: config.json (with its labels), model.safetensors and vocab.txt."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    values = {**model.config.to_dict(), "lowercase": tokenizer.lowercase, "labels": list(model.labels)}
    (folder / CONFIG_FILE).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    save_file({name: tensor.contiguous() for name, tensor in model.state_dict().items()}, folder / WEIGHTS_FILE)
    tokenizer.save(folder / VOCAB_FILE)
    return folder


def load_classifier(folder: str | Path) -> tuple[SentenceClassifier, Tokenizer]:
    folder = Path(folder)
    values = json.loads((folder / CONFIG_FILE).read_text(encoding="utf-8"))
    if "labels" not in values:
        raise ValueError(f"{folder} isn't a fine-tuned classifier (its config.json has no labels).")
    model = SentenceClassifier(BertConfig.from_dict(values), values["labels"])
    model.load_state_dict(load_file(folder / WEIGHTS_FILE))
    tokenizer = Tokenizer.load(folder / VOCAB_FILE, lowercase=values.get("lowercase", True), special_in_text=False)
    return model.eval(), tokenizer


__all__ = [
    "SST2_LABELS",
    "Example",
    "FinetuneSettings",
    "Prediction",
    "SentenceClassifier",
    "classify",
    "evaluate_classifier",
    "finetune",
    "learning_rate_at",
    "load_classifier",
    "read_examples",
    "save_classifier",
    "sst2",
]
