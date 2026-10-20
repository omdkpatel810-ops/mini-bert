"""Download WikiText-103 (about 100 million words of Wikipedia) into data/wikitext-103/ as plain text.

    pip install -e ".[data]"
    python scripts/get_wikitext.py

Writes train.txt, valid.txt and test.txt with one paragraph or heading per line.
WikiText is by Salesforce Research, under CC BY-SA 3.0: https://huggingface.co/datasets/Salesforce/wikitext
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

FOLDER = Path(__file__).resolve().parent.parent / "data" / "wikitext-103"
SPLITS = {
    "train.txt": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "valid.txt": ["validation-00000-of-00001.parquet"],
    "test.txt": ["test-00000-of-00001.parquet"],
}


def main() -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)
    for name, parts in SPLITS.items():
        lines = 0
        with (FOLDER / name).open("w", encoding="utf-8") as out:
            for part in parts:
                path = hf_hub_download("Salesforce/wikitext", f"wikitext-103-raw-v1/{part}", repo_type="dataset")
                for line in pq.read_table(path).column("text").to_pylist():
                    if line.strip():
                        out.write(line.strip() + "\n")
                        lines += 1
        print(f"{FOLDER / name}: {lines:,} lines")


if __name__ == "__main__":
    main()
