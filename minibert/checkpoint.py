"""Save and load models, including Google's pre-trained BERT from the Hugging Face Hub.

Google's checkpoints use the names from the original TensorFlow code
("bert.encoder.layer.0.attention.self.query.weight"). This model uses shorter names
("bert.layers.0.attention.query.weight"), so loading translates each name, and
refuses a checkpoint with anything it doesn't recognise rather than silently
leaving weights random.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from minibert.model import BertConfig, MaskedLanguageModel
from minibert.tokenizer import Tokenizer

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
VOCAB_FILE = "vocab.txt"

# Hugging Face config.json field → our BertConfig field.
_CONFIG_FIELDS = {
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_size",
    "num_hidden_layers": "layers",
    "num_attention_heads": "heads",
    "intermediate_size": "intermediate_size",
    "max_position_embeddings": "max_positions",
    "type_vocab_size": "segments",
    "hidden_dropout_prob": "dropout",
    "attention_probs_dropout_prob": "attention_dropout",
    "layer_norm_eps": "layer_norm_eps",
    "pad_token_id": "pad_id",
}

# Google's names → ours, applied in order.
_RENAMES = [
    (r"^bert\.embeddings\.word_embeddings\.", "bert.embeddings.words."),
    (r"^bert\.embeddings\.position_embeddings\.", "bert.embeddings.positions."),
    (r"^bert\.embeddings\.token_type_embeddings\.", "bert.embeddings.segments."),
    (r"^bert\.embeddings\.LayerNorm\.", "bert.embeddings.norm."),
    (r"^bert\.encoder\.layer\.(\d+)\.attention\.self\.(query|key|value)\.", r"bert.layers.\1.attention.\2."),
    (r"^bert\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"bert.layers.\1.attention.out."),
    (r"^bert\.encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.", r"bert.layers.\1.attention.norm."),
    (r"^bert\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"bert.layers.\1.feed_forward.up."),
    (r"^bert\.encoder\.layer\.(\d+)\.output\.dense\.", r"bert.layers.\1.feed_forward.down."),
    (r"^bert\.encoder\.layer\.(\d+)\.output\.LayerNorm\.", r"bert.layers.\1.feed_forward.norm."),
    (r"^bert\.pooler\.dense\.", "bert.pooler."),
    (r"^cls\.predictions\.transform\.dense\.", "transform."),
    (r"^cls\.predictions\.transform\.LayerNorm\.", "norm."),
    (r"^cls\.predictions\.bias$", "bias"),
    (r"\.gamma$", ".weight"),  # the TensorFlow-era names for LayerNorm's scale and shift
    (r"\.beta$", ".bias"),
]
# In Google's checkpoints but not needed: the next-sentence head (this model skips that
# task, as RoBERTa showed it doesn't help), copies of tied weights, and a constant buffer.
_SKIPPED = re.compile(
    r"^(cls\.seq_relationship\.|cls\.predictions\.decoder\.(weight|bias)$|bert\.embeddings\.position_ids$)"
)


class CheckpointError(Exception):
    pass


def config_from_hugging_face(values: Mapping) -> BertConfig:
    if values.get("hidden_act", "gelu") != "gelu":
        raise CheckpointError(f"Only GELU models are supported, not {values['hidden_act']}.")
    if values.get("position_embedding_type", "absolute") != "absolute":
        raise CheckpointError("Only absolute position embeddings are supported.")
    return BertConfig(**{ours: values[theirs] for theirs, ours in _CONFIG_FIELDS.items() if theirs in values})


def convert_hugging_face_weights(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Translate a Google/Hugging Face BERT state dict to this model's names."""
    converted: dict[str, Tensor] = {}
    for name, tensor in state.items():
        if not name.startswith(("bert.", "cls.")):
            name = "bert." + name  # a bare BertModel checkpoint, without the "bert." prefix
        if _SKIPPED.match(name):
            continue
        ours = name
        for pattern, replacement in _RENAMES:
            ours = re.sub(pattern, replacement, ours)
        converted[ours] = tensor
    return converted


def load_weights(model: MaskedLanguageModel, state: Mapping[str, Tensor]) -> list[str]:
    """Load weights by name and return the masked-LM head weights the checkpoint didn't have.

    The encoder must be complete, except the pooler: checkpoints trained only on masked
    words often leave it out, and it's only used, and trained, when classifying.
    """
    result = model.load_state_dict(dict(state), strict=False)
    if result.unexpected_keys:
        raise CheckpointError(f"The checkpoint has weights this model doesn't: {', '.join(result.unexpected_keys[:5])}")
    encoder_missing = [
        name for name in result.missing_keys if name.startswith("bert.") and not name.startswith("bert.pooler.")
    ]
    if encoder_missing:
        raise CheckpointError(f"The checkpoint is missing encoder weights: {', '.join(encoder_missing[:5])}")
    return [name for name in result.missing_keys if not name.startswith("bert.")]


def load_hugging_face(source: str | Path) -> tuple[MaskedLanguageModel, Tokenizer]:
    """Load Google's BERT from a Hugging Face Hub name ("bert-base-uncased") or a downloaded folder."""
    folder = _download(source) if not Path(source).is_dir() else Path(source)
    values = json.loads((folder / CONFIG_FILE).read_text(encoding="utf-8"))
    model = MaskedLanguageModel(config_from_hugging_face(values))
    if (folder / WEIGHTS_FILE).exists():
        state = load_file(folder / WEIGHTS_FILE)
    else:
        state = torch.load(folder / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing = load_weights(model, convert_hugging_face_weights(state))
    if missing:
        raise CheckpointError(f"{source} has no masked-language-model head ({', '.join(missing)}).")
    lowercase = values.get("do_lower_case", "uncased" in str(source).lower())
    return model.eval(), Tokenizer.load(folder / VOCAB_FILE, lowercase=lowercase)


def _download(name: str | Path) -> Path:
    """The folder with a Hub model's files, downloading them only if they aren't cached yet."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.errors import LocalEntryNotFoundError

    name = str(name)
    for weights in (WEIGHTS_FILE, "pytorch_model.bin"):
        try:
            paths = [hf_hub_download(name, file, local_files_only=True) for file in (CONFIG_FILE, VOCAB_FILE, weights)]
            return Path(paths[0]).parent
        except LocalEntryNotFoundError:
            continue
    files = set(list_repo_files(name))
    weights = WEIGHTS_FILE if WEIGHTS_FILE in files else "pytorch_model.bin"
    paths = [hf_hub_download(name, file) for file in (CONFIG_FILE, VOCAB_FILE, weights)]
    return Path(paths[0]).parent


def save(model: MaskedLanguageModel, tokenizer: Tokenizer, folder: str | Path) -> Path:
    """Save in this project's own format: config.json, model.safetensors and vocab.txt."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    values = {**model.config.to_dict(), "lowercase": tokenizer.lowercase}
    (folder / CONFIG_FILE).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    save_file({name: tensor.contiguous() for name, tensor in model.state_dict().items()}, folder / WEIGHTS_FILE)
    tokenizer.save(folder / VOCAB_FILE)
    return folder


def load(source: str | Path) -> tuple[MaskedLanguageModel, Tokenizer]:
    """Load a model saved with `save`, or Google's BERT by its Hugging Face name or folder."""
    folder = Path(source)
    if folder.is_dir() and (folder / CONFIG_FILE).exists():
        values = json.loads((folder / CONFIG_FILE).read_text(encoding="utf-8"))
        if "layers" in values:  # our own format
            model = MaskedLanguageModel(BertConfig.from_dict(values))
            missing = load_weights(model, load_file(folder / WEIGHTS_FILE))
            if missing:
                raise CheckpointError(f"{folder} is missing weights: {', '.join(missing)}")
            tokenizer = Tokenizer.load(folder / VOCAB_FILE, lowercase=values.get("lowercase", True))
            return model.eval(), tokenizer
    return load_hugging_face(source)
