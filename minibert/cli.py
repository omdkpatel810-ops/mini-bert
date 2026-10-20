"""Command line: minibert tokenize | train-tokenizer."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TextIO

from minibert import __version__
from minibert.tokenizer import Tokenizer
from minibert.tokenizer.train import available_cpus, count_words, train_vocab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minibert", description="BERT built from scratch.")
    parser.add_argument("--version", action="version", version=f"minibert {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    tokenize = commands.add_parser("tokenize", help="show how text becomes BERT tokens")
    tokenize.add_argument("text")
    tokenize.add_argument("--pair", help="a second text, as for question answering or sentence pairs")
    tokenize.add_argument("--vocab", type=Path, help="vocab.txt to use (default: Google's bert-base-uncased)")
    tokenize.add_argument("--cased", action="store_true", help="keep capitals and accents")
    tokenize.add_argument("--max-length", type=int, help="cut the input to this many tokens")

    train = commands.add_parser("train-tokenizer", help="learn a WordPiece vocabulary from text files")
    train.add_argument("files", nargs="+", type=Path, metavar="FILE", help="UTF-8 text files")
    train.add_argument("--out", type=Path, required=True, help="where to write vocab.txt")
    train.add_argument("--vocab-size", type=int, default=30_522, help="tokens to learn (default: 30522, as BERT)")
    train.add_argument("--min-frequency", type=int, default=2, help="never merge pairs seen fewer times (default: 2)")
    train.add_argument("--alphabet", type=int, default=1_000, help="most characters to keep (default: 1000)")
    train.add_argument("--cased", action="store_true", help="keep capitals and accents")
    train.add_argument("--workers", type=int, default=0, help="processes for counting words (default: one per core)")
    return parser


def main(argv: Sequence[str] | None = None, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        if args.command == "tokenize":
            return _tokenize(args, out)
        return _train(args, out)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=err)
        return 1


def _tokenize(args: argparse.Namespace, out: TextIO) -> int:
    options = {"lowercase": not args.cased}
    tokenizer = Tokenizer.load(args.vocab, **options) if args.vocab else Tokenizer.bert_base_uncased(**options)
    encoding = tokenizer.encode(args.text, args.pair, max_length=args.max_length)
    width = max(len(token) for token in encoding.tokens)
    for token, token_id, type_id in zip(encoding.tokens, encoding.ids, encoding.type_ids, strict=True):
        segment = f"  sentence {'AB'[type_id]}" if args.pair is not None else ""
        print(f"{token:<{width}}  {token_id:>6}{segment}", file=out)
    print(f"{len(encoding)} tokens", file=out)
    return 0


def _train(args: argparse.Namespace, out: TextIO) -> int:
    for path in args.files:
        if not path.is_file():
            raise ValueError(f"No such file: {path}")
    started = time.perf_counter()
    workers = args.workers or available_cpus()
    counts = count_words(_lines(args.files), lowercase=not args.cased, workers=workers)
    if not counts:
        raise ValueError("The files have no words to learn from.")
    print(
        f"Counted {sum(counts.values()):,} words ({len(counts):,} different) in {time.perf_counter() - started:.1f} s.",
        file=out,
    )
    result = train_vocab(
        counts, vocab_size=args.vocab_size, min_frequency=args.min_frequency, alphabet_size=args.alphabet
    )
    Tokenizer(result.vocab, lowercase=not args.cased).save(args.out)
    print(
        f"Learned {len(result.vocab):,} tokens ({result.merges:,} merges) in {result.seconds:.1f} s. Saved {args.out}.",
        file=out,
    )
    if len(result.vocab) < args.vocab_size:
        print(f"  Stopped early: no pair left that appears {args.min_frequency}+ times.", file=out)
    if result.skipped_words:
        skipped = result.skipped_words
        words, they = ("word", "it") if skipped == 1 else ("words", "they")
        print(
            f"  Left out {skipped:,} {words} with rare characters outside the "
            f"{result.alphabet:,}-character alphabet; {they} will become [UNK].",
            file=out,
        )
    return 0


def _lines(paths: Sequence[Path]) -> Iterator[str]:
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            yield from handle
