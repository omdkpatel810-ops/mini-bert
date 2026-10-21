from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import BertConfig as HFConfig
from transformers import BertForMaskedLM, BertModel

from minibert import checkpoint
from minibert.checkpoint import (
    CheckpointError,
    config_from_hugging_face,
    convert_hugging_face_weights,
    load,
    load_weights,
    save,
)
from minibert.model import BertConfig, MaskedLanguageModel
from minibert.predict import fill_mask
from minibert.tokenizer import SPECIAL_TOKENS, Tokenizer

WORDS = ["the", "cat", "sat", "on", "mat", "dog", "ran", "."]
VOCAB = [*SPECIAL_TOKENS, *WORDS]
CONFIG = BertConfig(vocab_size=len(VOCAB), hidden_size=16, layers=2, heads=2, intermediate_size=32, max_positions=16)


def hf_config(**overrides) -> HFConfig:
    values = dict(
        vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32
    )
    return HFConfig(**{**values, "max_position_embeddings": 16, **overrides})


def outputs(model: MaskedLanguageModel) -> torch.Tensor:
    ids = torch.tensor([[2, 5, 6, 7, 3]])
    with torch.no_grad():
        return model(ids)[0]


def hugging_face_folder(tmp_path, model, rename=None, weights="model.safetensors"):
    folder = tmp_path / "hf"
    folder.mkdir()
    state = {name: tensor.clone().contiguous() for name, tensor in model.state_dict().items()}
    if rename:
        state = {rename(name): tensor for name, tensor in state.items()}
    if weights == "model.safetensors":
        save_file(state, folder / weights)
    else:
        torch.save(state, folder / weights)
    (folder / "config.json").write_text(model.config.to_json_string(), encoding="utf-8")
    (folder / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    return folder


# Our own format ---------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    model = MaskedLanguageModel(CONFIG).eval()
    tokenizer = Tokenizer(VOCAB, lowercase=False)

    folder = save(model, tokenizer, tmp_path / "model")
    loaded, loaded_tokenizer = load(folder)

    assert loaded.config == CONFIG
    assert loaded_tokenizer.vocab == VOCAB
    assert loaded_tokenizer.lowercase is False
    torch.testing.assert_close(outputs(loaded), outputs(model))
    assert not loaded.training
    assert json.loads((folder / "config.json").read_text())["layers"] == 2


def test_our_format_must_be_complete(tmp_path):
    folder = save(MaskedLanguageModel(CONFIG), Tokenizer(VOCAB), tmp_path / "model")
    state = MaskedLanguageModel(CONFIG).state_dict()
    del state["transform.weight"]
    save_file(state, folder / "model.safetensors")

    with pytest.raises(CheckpointError, match="is missing weights: transform.weight"):
        load(folder)


# Hugging Face and Google checkpoints ---------------------------------------------


def test_loads_a_hugging_face_checkpoint(tmp_path):
    torch.manual_seed(0)
    theirs = BertForMaskedLM(hf_config()).eval()

    ours, tokenizer = load(hugging_face_folder(tmp_path, theirs))

    expected = theirs(input_ids=torch.tensor([[2, 5, 6, 7, 3]])).logits.detach()
    torch.testing.assert_close(outputs(ours), expected, atol=1e-5, rtol=1e-4)
    assert tokenizer.vocab == VOCAB


def test_loads_old_pytorch_files_with_tensorflow_era_names(tmp_path):
    torch.manual_seed(0)
    theirs = BertForMaskedLM(hf_config()).eval()

    def tensorflow_names(name: str) -> str:
        if "LayerNorm" in name:
            return name.replace(".weight", ".gamma").replace(".bias", ".beta")
        return name

    folder = hugging_face_folder(tmp_path, theirs, rename=tensorflow_names, weights="pytorch_model.bin")
    ours, _ = load(folder)

    expected = theirs(input_ids=torch.tensor([[2, 5, 6, 7, 3]])).logits.detach()
    torch.testing.assert_close(outputs(ours), expected, atol=1e-5, rtol=1e-4)


def test_an_encoder_without_a_language_model_head_is_refused(tmp_path):
    folder = hugging_face_folder(tmp_path, BertModel(hf_config()), rename=lambda name: name)  # names lack "bert."

    with pytest.raises(CheckpointError, match="has no masked-language-model head"):
        load(folder)


def test_bare_encoder_names_get_the_bert_prefix():
    converted = convert_hugging_face_weights({"encoder.layer.3.attention.self.query.weight": torch.zeros(1)})
    assert list(converted) == ["bert.layers.3.attention.query.weight"]


def test_skips_the_next_sentence_head_and_tied_copies():
    state = {
        "cls.seq_relationship.weight": torch.zeros(1),
        "cls.predictions.decoder.weight": torch.zeros(1),
        "cls.predictions.decoder.bias": torch.zeros(1),
        "bert.embeddings.position_ids": torch.zeros(1),
        "cls.predictions.bias": torch.zeros(1),
    }
    assert list(convert_hugging_face_weights(state)) == ["bias"]


def test_a_checkpoint_without_a_pooler_is_fine():
    model = MaskedLanguageModel(CONFIG)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("bert.pooler.")}

    assert load_weights(MaskedLanguageModel(CONFIG), state) == []


def test_unknown_weights_are_refused():
    state = {**MaskedLanguageModel(CONFIG).state_dict(), "bert.adapter.weight": torch.zeros(1)}

    with pytest.raises(CheckpointError, match="weights this model doesn't: bert.adapter.weight"):
        load_weights(MaskedLanguageModel(CONFIG), state)


def test_missing_encoder_weights_are_refused():
    state = MaskedLanguageModel(CONFIG).state_dict()
    del state["bert.layers.1.feed_forward.up.weight"]

    with pytest.raises(CheckpointError, match="missing encoder weights: bert.layers.1.feed_forward.up.weight"):
        load_weights(MaskedLanguageModel(CONFIG), state)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_act": "relu"}, "Only GELU models are supported, not relu"),
        ({"position_embedding_type": "relative_key"}, "Only absolute position embeddings"),
    ],
)
def test_unsupported_configs(overrides, message):
    with pytest.raises(CheckpointError, match=message):
        config_from_hugging_face({**hf_config().to_dict(), **overrides})


def test_config_from_hugging_face():
    values = hf_config(hidden_dropout_prob=0.2, layer_norm_eps=1e-6).to_dict()
    assert config_from_hugging_face(values) == BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        layers=2,
        heads=2,
        intermediate_size=32,
        max_positions=16,
        dropout=0.2,
        layer_norm_eps=1e-6,
    )


# Downloading -----------------------------------------------------------------------


class FakeHub:
    """Stands in for huggingface_hub: a cache that may or may not have the files yet."""

    def __init__(self, folder, cached: bool, files=("config.json", "vocab.txt", "model.safetensors")):
        self.folder, self.cached, self.files, self.downloads = folder, cached, set(files), []

    def hf_hub_download(self, name, file, local_files_only=False):
        from huggingface_hub.errors import LocalEntryNotFoundError

        if local_files_only and (not self.cached or file not in self.files):
            raise LocalEntryNotFoundError(file)
        if not local_files_only:
            self.downloads.append(file)
        return str(self.folder / file)

    def list_repo_files(self, name):
        return sorted(self.files)


@pytest.mark.parametrize("cached", [True, False])
def test_downloads_only_what_is_not_cached(tmp_path, monkeypatch, cached):
    import huggingface_hub

    hub = FakeHub(tmp_path, cached)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", hub.list_repo_files)

    assert checkpoint._download("someone/bert") == tmp_path
    assert hub.downloads == ([] if cached else ["config.json", "vocab.txt", "model.safetensors"])


def test_downloads_old_pytorch_weights_when_there_are_no_safetensors(tmp_path, monkeypatch):
    import huggingface_hub

    hub = FakeHub(tmp_path, cached=False, files=("config.json", "vocab.txt", "pytorch_model.bin"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub.hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", hub.list_repo_files)

    checkpoint._download("someone/bert")

    assert hub.downloads[-1] == "pytorch_model.bin"


@pytest.mark.network
def test_googles_real_bert_tiny_matches_hugging_face():
    name = "google/bert_uncased_L-2_H-128_A-2"
    ours, tokenizer = load(name)
    theirs = BertForMaskedLM.from_pretrained(name).eval()
    encoding = tokenizer.encode("Paris is the [MASK] of France.")
    ids = torch.tensor([encoding.ids])

    with torch.no_grad():
        torch.testing.assert_close(ours(ids)[0], theirs(input_ids=ids).logits, atol=1e-4, rtol=1e-4)
    assert len(tokenizer) == 30_522


# Filling in blanks ---------------------------------------------------------------------


def test_fill_mask_gives_the_top_guesses_for_each_blank():
    model = MaskedLanguageModel(CONFIG)
    tokenizer = Tokenizer(VOCAB)

    guesses = fill_mask(model, tokenizer, "the [MASK] sat on the [MASK] .", top=3)

    assert [len(options) for options in guesses] == [3, 3]
    for options in guesses:
        probabilities = [guess.probability for guess in options]
        assert probabilities == sorted(probabilities, reverse=True)
        assert 0 < sum(probabilities) <= 1


def test_fill_mask_never_offers_more_guesses_than_words():
    guesses = fill_mask(MaskedLanguageModel(CONFIG), Tokenizer(VOCAB), "[MASK]", top=100)
    assert len(guesses[0]) == len(VOCAB)


def test_fill_mask_needs_a_mask():
    with pytest.raises(ValueError, match=r"Put \[MASK\] where the missing word goes"):
        fill_mask(MaskedLanguageModel(CONFIG), Tokenizer(VOCAB), "the cat sat")
