"""Run Google's BERT-Base through our model and Hugging Face's, and compare outputs and speed.

    python scripts/get_wikitext.py
    python scripts/compare_models.py

Hides 15% of the words in WikiText-103 test sentences, as in BERT's training, and
checks both models give the same numbers and the same best guess for each.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import BertForMaskedLM

from minibert.checkpoint import load

ROOT = Path(__file__).resolve().parent.parent
SENTENCES, BATCH, MAX_TOKENS = 256, 16, 128


def main() -> None:
    torch.manual_seed(0)
    ours, tokenizer = load("bert-base-uncased")
    theirs = {
        "Hugging Face (eager attention)": BertForMaskedLM.from_pretrained(
            "bert-base-uncased", attn_implementation="eager"
        ).eval(),
        "Hugging Face (fused attention)": BertForMaskedLM.from_pretrained(
            "bert-base-uncased", attn_implementation="sdpa"
        ).eval(),
    }
    lines = [
        line
        for line in (ROOT / "data" / "wikitext-103" / "test.txt").read_text(encoding="utf-8").splitlines()
        if len(line.split()) > 8
    ]
    batches = []
    for start in range(0, SENTENCES, BATCH):
        encodings = tokenizer.encode_batch(lines[start : start + BATCH], max_length=MAX_TOKENS)
        ids = torch.tensor([e.ids for e in encodings])
        mask = torch.tensor([e.attention_mask for e in encodings])
        hide = (torch.rand(ids.shape) < 0.15) & mask.bool() & (ids != tokenizer.cls_id) & (ids != tokenizer.sep_id)
        batches.append((ids.masked_fill(hide, tokenizer.mask_id), mask, hide))

    timings = {"ours": 0.0, **dict.fromkeys(theirs, 0.0)}
    largest, same, hidden_words = 0.0, 0, 0
    with torch.inference_mode():
        for ids, mask, hide in batches:  # one batch at a time: all the scores at once would take gigabytes
            started = time.perf_counter()
            a = ours(ids, attention_mask=mask)[0]
            timings["ours"] += time.perf_counter() - started
            for name, model in theirs.items():
                started = time.perf_counter()
                b = model(input_ids=ids, attention_mask=mask).logits
                timings[name] += time.perf_counter() - started
                largest = max(largest, (a - b)[mask.bool()].abs().max().item())
            same += (a[hide].argmax(-1) == b[hide].argmax(-1)).sum().item()
            hidden_words += hide.sum().item()
    tokens = sum(mask.sum().item() for _, mask, _ in batches)
    print(f"{SENTENCES} sentences, {tokens:,} tokens, {hidden_words:,} hidden words")
    print(f"largest difference in any score: {largest:.2e}")
    print(f"same best guess for {same:,} of {hidden_words:,} hidden words")
    for name, seconds in timings.items():
        print(f"  {name:<32} {seconds:5.1f} s  ({tokens / seconds:,.0f} tokens/s)")


if __name__ == "__main__":
    main()
