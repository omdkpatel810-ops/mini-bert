from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers import BertConfig as HFConfig
from transformers import BertForPreTraining

from minibert.checkpoint import config_from_hugging_face, convert_hugging_face_weights, load_weights
from minibert.model import Bert, BertConfig, MaskedLanguageModel, count_parameters

SMALL = BertConfig(vocab_size=100, hidden_size=32, layers=2, heads=4, intermediate_size=64, max_positions=40)


def hugging_face_twin(config: BertConfig, seed: int = 0) -> tuple[MaskedLanguageModel, BertForPreTraining]:
    """A randomly initialised Hugging Face BERT, and ours with the same weights copied in."""
    torch.manual_seed(seed)
    theirs = BertForPreTraining(
        HFConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.layers,
            num_attention_heads=config.heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_positions,
            attn_implementation="eager",
        )
    ).eval()
    # Hugging Face starts biases and LayerNorms at fixed values; randomise them too, so a
    # mix-up between, say, two biases can't hide.
    with torch.no_grad():
        for parameter in theirs.parameters():
            parameter.normal_(0, 0.5)
    ours = MaskedLanguageModel(config_from_hugging_face(theirs.config.to_dict())).eval()
    assert load_weights(ours, convert_hugging_face_weights(theirs.state_dict())) == []
    return ours, theirs


def random_batch(config: BertConfig, batch: int = 3, tokens: int = 12, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, config.vocab_size, (batch, tokens), generator=generator)
    segments = (torch.arange(tokens) >= tokens // 2).long().expand(batch, tokens)
    mask = torch.ones(batch, tokens, dtype=torch.long)
    mask[1, tokens - 4 :] = 0  # some padding
    return ids, segments, mask


# Against Hugging Face's implementation ----------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        SMALL,
        BertConfig(vocab_size=50, hidden_size=24, layers=1, heads=1, intermediate_size=24, max_positions=20),
        BertConfig(vocab_size=200, hidden_size=48, layers=3, heads=6, intermediate_size=192, max_positions=64),
    ],
    ids=["2 layers", "1 head", "6 heads"],
)
def test_matches_hugging_face_bert_with_the_same_weights(config):
    ours, theirs = hugging_face_twin(config)
    ids, segments, mask = random_batch(config)

    with torch.no_grad():
        logits, output = ours(ids, segments, mask)
        reference = theirs(input_ids=ids, token_type_ids=segments, attention_mask=mask, output_hidden_states=True)
        pooled = theirs.bert(input_ids=ids, token_type_ids=segments, attention_mask=mask).pooler_output

    real = mask.bool()
    torch.testing.assert_close(output.hidden[real], reference.hidden_states[-1][real], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(logits[real], reference.prediction_logits[real], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(output.pooled, pooled, atol=1e-5, rtol=1e-4)


def test_same_parameter_count_as_bert_base():
    with torch.device("meta"):  # count without allocating 440 MB
        encoder = Bert(BertConfig())
        full = MaskedLanguageModel(BertConfig())
    assert count_parameters(encoder) == 109_482_240  # BERT-Base, as published
    assert count_parameters(full) == 109_482_240 + 768 * 768 + 768 + 2 * 768 + 30_522


# Behaviour ----------------------------------------------------------------------------


def test_output_shapes():
    model = MaskedLanguageModel(SMALL).eval()
    ids, segments, mask = random_batch(SMALL)

    logits, output = model(ids, segments, mask, return_attentions=True)

    assert logits.shape == (3, 12, SMALL.vocab_size)
    assert output.hidden.shape == (3, 12, SMALL.hidden_size)
    assert output.pooled.shape == (3, SMALL.hidden_size)
    assert len(output.attentions) == SMALL.layers
    assert output.attentions[0].shape == (3, SMALL.heads, 12, 12)


def test_attention_weights_sum_to_one_and_ignore_padding():
    model = MaskedLanguageModel(SMALL).eval()
    ids, segments, mask = random_batch(SMALL)

    _, output = model(ids, segments, mask, return_attentions=True)

    for weights in output.attentions:
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        assert weights[1, :, :, -4:].abs().max() == 0  # nobody attends to sample 1's padding


@settings(max_examples=40, deadline=None)
@given(length=st.integers(1, 20), padding=st.integers(1, 15), seed=st.integers(0, 1000))
def test_padding_never_changes_the_real_tokens(length, padding, seed):
    model = Bert(SMALL).eval()
    ids = torch.randint(1, SMALL.vocab_size, (1, length), generator=torch.Generator().manual_seed(seed))
    padded = torch.cat([ids, torch.zeros(1, padding, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, length), torch.zeros(1, padding)], dim=1).long()

    with torch.no_grad():
        alone = model(ids).hidden
        with_padding = model(padded, attention_mask=mask).hidden[:, :length]

    torch.testing.assert_close(with_padding, alone, atol=1e-5, rtol=1e-5)


def test_defaults_are_one_sentence_and_no_padding():
    model = Bert(SMALL).eval()
    ids = torch.randint(1, SMALL.vocab_size, (2, 7))

    with torch.no_grad():
        default = model(ids).hidden
        explicit = model(ids, torch.zeros_like(ids), torch.ones_like(ids)).hidden

    torch.testing.assert_close(default, explicit)


def test_dropout_only_while_training():
    model = MaskedLanguageModel(SMALL)
    ids, segments, mask = random_batch(SMALL)

    model.train()
    first, _ = model(ids, segments, mask)
    second, _ = model(ids, segments, mask)
    assert not torch.equal(first, second)

    model.eval()
    first, _ = model(ids, segments, mask)
    second, _ = model(ids, segments, mask)
    assert torch.equal(first, second)


def test_the_output_layer_shares_the_word_embeddings():
    model = MaskedLanguageModel(SMALL)
    ids, segments, mask = random_batch(SMALL)

    logits, _ = model(ids, segments, mask)
    logits.sum().backward()

    # One weight matrix, trained from both ends: input embeddings and output scores.
    assert model.bert.embeddings.words.weight.grad is not None
    assert not any(
        p.shape == (SMALL.vocab_size, SMALL.hidden_size) for n, p in model.named_parameters() if "words" not in n
    )


def test_padding_embedding_starts_at_zero_and_learns_nothing_from_padding():
    model = Bert(SMALL)
    assert model.embeddings.words.weight[SMALL.pad_id].abs().max() == 0

    model(torch.tensor([[5, 6, 0, 0]])).hidden.sum().backward()

    # Reading [PAD] tokens sends no gradient to [PAD]'s row. (In the masked-LM model it still
    # trains a little from the output side, where [PAD] is one of the words it could predict.)
    assert model.embeddings.words.weight.grad[SMALL.pad_id].abs().max() == 0
    assert model.embeddings.words.weight.grad[5].abs().max() > 0


def test_too_many_tokens():
    with pytest.raises(ValueError, match="at most 40 tokens"):
        Bert(SMALL)(torch.ones(1, 41, dtype=torch.long))


def test_heads_must_divide_the_hidden_size():
    with pytest.raises(ValueError, match="must divide evenly into 5 heads"):
        BertConfig(hidden_size=32, heads=5)


def test_config_round_trip():
    assert BertConfig.from_dict({**SMALL.to_dict(), "something_else": 1}) == SMALL
    assert SMALL.head_size == 8
