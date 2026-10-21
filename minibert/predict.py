"""Use a trained model: fill in [MASK]ed words."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from minibert.model import MaskedLanguageModel
from minibert.tokenizer import Tokenizer
from minibert.tokenizer.wordpiece import MASK


@dataclass(frozen=True)
class Guess:
    token: str
    probability: float


@torch.no_grad()
def fill_mask(model: MaskedLanguageModel, tokenizer: Tokenizer, text: str, top: int = 5) -> list[list[Guess]]:
    """The `top` most likely words for each [MASK] in the text, most likely first."""
    encoding = tokenizer.encode(text, max_length=model.config.max_positions)
    positions = [i for i, token in enumerate(encoding.tokens) if token == MASK]
    if not positions:
        raise ValueError(f"Put {MASK} where the missing word goes, like: Paris is the {MASK} of France.")
    model.eval()
    ids = torch.tensor([encoding.ids])
    segments = torch.tensor([encoding.type_ids])
    logits, _ = model(ids, segments)
    guesses = []
    for position in positions:
        probabilities = logits[0, position].softmax(dim=-1)
        best = probabilities.topk(min(top, len(tokenizer)))
        guesses.append(
            [Guess(tokenizer.vocab[i], p) for p, i in zip(best.values.tolist(), best.indices.tolist(), strict=True)]
        )
    return guesses
