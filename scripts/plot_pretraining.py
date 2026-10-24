"""Draw docs/pretraining.png from results/pretraining.json: loss and held-out accuracy over 60,000 steps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOOGLE_MINI_VALID = 51.5  # Google's BERT-Mini on the same WikiText-103 validation puzzles (see README)
GOOGLE_BASE_VALID = 62.5


def main() -> None:
    data = json.loads((ROOT / "results" / "pretraining.json").read_text(encoding="utf-8"))
    train = data["log"]["training_loss"]
    held = data["log"]["held_out"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot([s for s, _ in train], [loss for _, loss in train], label="training", color="C0")
    left.plot(
        [s for s, _, _ in held], [loss for _, loss, _ in held], label="held-out text", color="C1", marker="o", ms=3
    )
    left.axvspan(3_000, 7_000, color="gray", alpha=0.15)
    left.annotate(
        "stuck guessing common words",
        xy=(5_000, 6.25),
        xytext=(14_000, 7.6),
        arrowprops={"arrowstyle": "->", "color": "gray"},
        fontsize=9,
    )
    left.annotate(
        "starts using the words around the gap",
        xy=(8_500, 5.2),
        xytext=(14_000, 5.9),
        arrowprops={"arrowstyle": "->", "color": "gray"},
        fontsize=9,
    )
    left.set(xlabel="step", ylabel="loss (lower is better)", title="Loss")
    left.legend()

    right.plot([s for s, _, _ in held], [100 * a for _, _, a in held], label="ours", color="C1", marker="o", ms=3)
    right.axhline(GOOGLE_MINI_VALID, linestyle="--", color="gray", label=f"Google BERT-Mini ({GOOGLE_MINI_VALID}%)")
    right.axhline(GOOGLE_BASE_VALID, linestyle=":", color="gray", label=f"Google BERT-Base ({GOOGLE_BASE_VALID}%)")
    right.set(
        xlabel="step",
        ylabel="% of hidden words guessed right",
        title="Held-out WikiText-103 validation text",
        ylim=(0, 70),
    )
    right.legend(loc="lower right")
    for axis in (left, right):
        axis.set_xticks(range(0, 60_001, 10_000), [f"{k}k" for k in range(0, 61, 10)])
    figure.suptitle(
        f"Pre-training Mini BERT from scratch: {data['training_minutes']:.0f} minutes on a free Colab {data['gpu']}",
        fontsize=11,
    )
    figure.tight_layout()
    out = ROOT / "docs" / "pretraining.png"
    figure.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
