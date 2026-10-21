"""Draw BERT-Base's attention: all 144 heads for one sentence, and a close-up of one head.

    python scripts/plot_attention.py            # writes docs/attention.png

Needs matplotlib (pip install matplotlib).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec

from minibert.checkpoint import load

ROOT = Path(__file__).resolve().parent.parent
SENTENCE = "Sara built the shelf, then she painted it."
CLOSE_UP = (5, 4)  # layer 5, head 4 (counting from 1): the coreference head Clark et al. (2019) describe


def main() -> None:
    model, tokenizer = load("bert-base-uncased")
    encoding = tokenizer.encode(SENTENCE)
    with torch.no_grad():
        _, output = model(torch.tensor([encoding.ids]), return_attentions=True)
    maps = torch.stack(output.attentions)[:, 0]  # (layers, heads, tokens, tokens)
    layers, heads = maps.shape[:2]

    figure = plt.figure(figsize=(13, 6.4), facecolor="white")
    grid = GridSpec(layers, heads + 14, figure=figure, wspace=0.08, hspace=0.08)
    for layer in range(layers):
        for head in range(heads):
            ax = figure.add_subplot(grid[layer, head])
            ax.imshow(maps[layer, head], cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            highlighted = (layer + 1, head + 1) == CLOSE_UP
            for spine in ax.spines.values():
                spine.set_edgecolor("#d62728" if highlighted else "#cccccc")
                spine.set_linewidth(2 if highlighted else 0.5)
            if head == 0:
                ax.set_ylabel(str(layer + 1), rotation=0, fontsize=7, labelpad=8, va="center")
            if layer == layers - 1:
                ax.set_xlabel(str(head + 1), fontsize=7)
    figure.text(0.03, 0.5, "layer", rotation=90, va="center", fontsize=9)
    figure.text(0.3, 0.035, "head", ha="center", fontsize=9)
    figure.text(0.3, 0.94, "All 144 heads (each square: which word looks at which)", ha="center", fontsize=11)

    ax = figure.add_subplot(grid[1:11, heads + 3 :])
    layer, head = CLOSE_UP
    ax.imshow(maps[layer - 1, head - 1], cmap="Blues", vmin=0, vmax=1)
    ticks = range(len(encoding.tokens))
    ax.set_xticks(ticks, encoding.tokens, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(ticks, encoding.tokens, fontsize=9)
    ax.set_xlabel("looks at", fontsize=9)
    ax.set_ylabel("word", fontsize=9)
    ax.set_title(f"Layer {layer}, head {head}: “it” looks back at “the (shelf)”, “she” at “Sara”", fontsize=10)
    tokens = encoding.tokens
    for word, target in (("it", "the"), ("she", "sara")):
        row, column = tokens.index(word), tokens.index(target)
        ax.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#d62728", linewidth=2))

    path = ROOT / "docs" / "attention.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
