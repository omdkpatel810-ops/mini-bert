"""BERT's model: a stack of Transformer encoder layers, written from scratch in PyTorch.

Only PyTorch's basic building blocks are used (Linear, Embedding, LayerNorm, Dropout);
attention is written out by hand rather than using nn.MultiheadAttention, so every
step of the paper is visible here.

    ids ─► word + position + segment embeddings ─► LayerNorm
        ─► N × [ self-attention ─► add & LayerNorm ─► feed-forward ─► add & LayerNorm ]
        ─► one vector per token (and a pooled vector for [CLS])
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BertConfig:
    vocab_size: int = 30_522
    hidden_size: int = 768
    layers: int = 12
    heads: int = 12
    intermediate_size: int = 3_072  # the feed-forward layer's width, 4x the hidden size in BERT
    max_positions: int = 512
    segments: int = 2  # sentence A and sentence B
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    pad_id: int = 0  # [PAD]'s embedding starts at zero, and padding sends it no gradient

    def __post_init__(self) -> None:
        if self.hidden_size % self.heads:
            raise ValueError(f"hidden_size ({self.hidden_size}) must divide evenly into {self.heads} heads.")

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> BertConfig:
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})


@dataclass
class MaskedWordResult:
    loss: Tensor  # cross-entropy over the hidden words
    predictions: Tensor  # the model's best guess for each hidden word
    targets: Tensor  # the words that were really there

    @property
    def correct(self) -> int:
        return int((self.predictions == self.targets).sum())


@dataclass
class BertOutput:
    hidden: Tensor  # (batch, tokens, hidden_size): one vector per token
    pooled: Tensor  # (batch, hidden_size): the [CLS] vector after the pooler, for classifying the whole input
    attentions: list[Tensor] | None = None  # per layer, (batch, heads, tokens, tokens), if asked for


class Embeddings(nn.Module):
    """A token's starting vector: what the word is, where it is, and which sentence it's in."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.words = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_id)
        self.positions = nn.Embedding(config.max_positions, config.hidden_size)
        self.segments = nn.Embedding(config.segments, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, ids: Tensor, segment_ids: Tensor) -> Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.words(ids) + self.positions(positions) + self.segments(segment_ids)
        return self.dropout(self.norm(x))


class SelfAttention(nn.Module):
    """Every token looks at every other token and takes a weighted mix of what it finds.

    Each of the `heads` heads does this separately in its own slice of the vector,
    so different heads can learn to follow different relationships.
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_size = config.head_size
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fused = False

    def forward(self, x: Tensor, mask_bias: Tensor, keep: Tensor, need_weights: bool) -> tuple[Tensor, Tensor | None]:
        batch, tokens, _ = x.shape

        def split_heads(t: Tensor) -> Tensor:  # (batch, tokens, hidden) → (batch, heads, tokens, head_size)
            return t.view(batch, tokens, self.heads, self.head_size).transpose(1, 2)

        q, k, v = split_heads(self.query(x)), split_heads(self.key(x)), split_heads(self.value(x))
        weights = None
        if self.fused and not need_weights:
            # PyTorch's fused kernel does the same sums in one step: faster, and lighter on GPU memory.
            dropout = self.attention_dropout.p if self.training else 0.0
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, dropout_p=dropout)
        else:
            # How well each token's query matches every token's key, scaled so the softmax doesn't saturate.
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size) + mask_bias
            weights = scores.softmax(dim=-1)
            context = self.attention_dropout(weights) @ v
        context = context.transpose(1, 2).reshape(batch, tokens, self.heads * self.head_size)
        return self.norm(x + self.dropout(self.out(context))), weights


class FeedForward(nn.Module):
    """The same small two-layer network applied to each token on its own."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.dropout(self.down(F.gelu(self.up(x)))))


class EncoderLayer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = SelfAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x: Tensor, mask_bias: Tensor, keep: Tensor, need_weights: bool) -> tuple[Tensor, Tensor | None]:
        x, weights = self.attention(x, mask_bias, keep, need_weights)
        return self.feed_forward(x), weights


class Bert(nn.Module):
    """The encoder: token ids in, one context-aware vector per token out."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.layers))
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.apply(_init_weights)

    def forward(
        self,
        ids: Tensor,
        segment_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        return_attentions: bool = False,
    ) -> BertOutput:
        """`ids` is (batch, tokens). `attention_mask` has 1 for real tokens and 0 for padding."""
        if ids.shape[1] > self.config.max_positions:
            raise ValueError(f"BERT reads at most {self.config.max_positions} tokens at once; got {ids.shape[1]}.")
        if segment_ids is None:
            segment_ids = torch.zeros_like(ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(ids)

        x = self.embeddings(ids, segment_ids)
        # Padding gets a huge negative score, so after the softmax no token attends to it.
        keep = attention_mask[:, None, None, :].bool()
        mask_bias = (~keep).to(x.dtype) * torch.finfo(x.dtype).min
        attentions = []
        for layer in self.layers:
            x, weights = layer(x, mask_bias, keep, return_attentions)
            if return_attentions:
                attentions.append(weights)
        pooled = torch.tanh(self.pooler(x[:, 0]))
        return BertOutput(hidden=x, pooled=pooled, attentions=attentions if return_attentions else None)

    def use_fused_attention(self, enabled: bool = True) -> None:
        """Switch attention to PyTorch's fused kernel (same results, faster), except when weights are asked for."""
        for layer in self.layers:
            layer.attention.fused = enabled


class MaskedLanguageModel(nn.Module):
    """BERT plus the head that guesses hidden words, used for pre-training and fill-in-the-blank.

    The head's output layer shares its weights with the word embeddings: the same
    vector stands for a word coming in and for predicting it going out.
    """

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = Bert(config)
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        _init_weights(self.transform)
        _init_weights(self.norm)

    def forward(
        self,
        ids: Tensor,
        segment_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        return_attentions: bool = False,
    ) -> tuple[Tensor, BertOutput]:
        """Scores for every vocabulary word at every position: (batch, tokens, vocab_size)."""
        output = self.bert(ids, segment_ids, attention_mask, return_attentions)
        return self.predict(output.hidden), output

    def predict(self, hidden: Tensor) -> Tensor:
        """Token vectors to scores for every word in the vocabulary."""
        h = self.norm(F.gelu(self.transform(hidden)))
        return h @ self.bert.embeddings.words.weight.T + self.bias

    def masked_word_loss(
        self,
        ids: Tensor,
        labels: Tensor,
        segment_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        ignore: int = -100,
    ) -> MaskedWordResult:
        """The pre-training loss: how badly the model guesses the hidden words.

        `labels` holds the real word where one was hidden and `ignore` elsewhere. Scores
        are only worked out at the hidden positions: scoring all 30,522 words at every
        position would be about 7x the work for the same loss.
        """
        hidden = self.bert(ids, segment_ids, attention_mask).hidden
        chosen = labels != ignore
        targets = labels[chosen]
        if not targets.numel():
            zero = hidden.sum() * 0.0  # nothing hidden: zero loss that still connects to the model
            return MaskedWordResult(zero, targets, targets)
        logits = self.predict(hidden[chosen]).float()
        return MaskedWordResult(F.cross_entropy(logits, targets), logits.argmax(dim=-1), targets)

    def use_fused_attention(self, enabled: bool = True) -> None:
        self.bert.use_fused_attention(enabled)


def _init_weights(module: nn.Module) -> None:
    """BERT's initialization: small normal weights (std 0.02), zero biases, LayerNorm at identity."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
