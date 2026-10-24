"""Draw docs/finetuning.png from results/finetuning.json: SST-2 accuracy for each model, learning rate and seed."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLOURS = {"ours (Mini BERT)": "C1", "Google BERT-Mini": "C0"}


def main() -> None:
    data = json.loads((ROOT / "results" / "finetuning.json").read_text(encoding="utf-8"))
    runs = data["runs"]
    models = list(COLOURS)
    rates = sorted({run["learning_rate"] for run in runs}, reverse=True)

    figure, axis = plt.subplots(figsize=(8, 4.4))
    width = 0.38
    for offset, model in zip((-width / 2, width / 2), models, strict=True):
        for i, rate in enumerate(rates):
            scores = [100 * r["accuracy"] for r in runs if r["model"] == model and r["learning_rate"] == rate]
            axis.bar(i + offset, statistics.mean(scores), width, color=COLOURS[model], label=model if i == 0 else None)
            axis.scatter([i + offset] * len(scores), scores, color="black", s=12, zorder=3)
            axis.text(
                i + offset, statistics.mean(scores) + 0.4, f"{statistics.mean(scores):.1f}", ha="center", fontsize=8
            )
    published = 100 * data["google_published_test"]
    axis.axhline(
        published, linestyle="--", color="gray", label=f"Google's published BERT-Mini ({published:.1f}%, test set)"
    )
    axis.set_xticks(range(len(rates)), [f"learning rate {rate:g}" for rate in rates])
    lowest = min(100 * r["accuracy"] for r in runs)
    axis.set_ylim(5 * (lowest // 5) - 5, 90)
    axis.set_ylabel("% of validation sentences right")
    axis.set_title("SST-2 sentiment after fine-tuning (bars: mean of 3 seeds; dots: each seed)", fontsize=11)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
    figure.tight_layout()
    out = ROOT / "docs" / "finetuning.png"
    figure.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
