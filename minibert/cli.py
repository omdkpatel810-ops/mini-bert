"""Command line: minibert tokenize | train-tokenizer | fill | prepare-data | pretrain | evaluate | finetune | classify."""

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

SST2_LABELS = ("negative", "positive")  # the same as minibert.finetune.SST2_LABELS, without importing PyTorch


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

    fill = commands.add_parser("fill", help="guess the word hidden by [MASK]")
    fill.add_argument("text", help='text with [MASK] where a word is missing, like "Paris is the [MASK] of France."')
    fill.add_argument(
        "--model",
        default="bert-base-uncased",
        help="a Hugging Face BERT name, or a folder saved by this project (default: bert-base-uncased)",
    )
    fill.add_argument("--top", type=int, default=5, help="how many guesses to show (default: 5)")

    prepare = commands.add_parser("prepare-data", help="tokenize a text file for pre-training")
    prepare.add_argument("file", type=Path, help="UTF-8 text, one paragraph per line")
    prepare.add_argument("--out", type=Path, required=True, help="where to write the token ids (.bin)")
    prepare.add_argument("--wikitext", action="store_true", help="undo WikiText's formatting; [SEP] between articles")
    prepare.add_argument("--vocab", type=Path, help="vocab.txt to use (default: Google's bert-base-uncased)")
    prepare.add_argument("--workers", type=int, default=0, help="processes for tokenizing (default: one per core)")

    pretrain = commands.add_parser("pretrain", help="train a BERT from scratch by guessing hidden words")
    pretrain.add_argument("--train", type=Path, required=True, help="training tokens from prepare-data")
    pretrain.add_argument("--valid", type=Path, required=True, help="held-out tokens from prepare-data")
    pretrain.add_argument("--run", type=Path, required=True, help="folder for checkpoints, the log and the model")
    pretrain.add_argument("--vocab", type=Path, help="the vocab.txt the data was prepared with (default: Google's)")
    pretrain.add_argument("--layers", type=int, default=4)
    pretrain.add_argument("--hidden", type=int, default=256)
    pretrain.add_argument("--heads", type=int, default=4)
    pretrain.add_argument("--max-positions", type=int, default=128)
    pretrain.add_argument("--steps", type=int, default=60_000)
    pretrain.add_argument("--batch-size", type=int, default=128)
    pretrain.add_argument("--sequence-length", type=int, default=128)
    pretrain.add_argument("--learning-rate", type=float, default=5e-4)
    pretrain.add_argument("--warmup", type=int, default=3_000, help="warm-up steps")
    pretrain.add_argument("--eval-every", type=int, default=1_000)
    pretrain.add_argument("--save-every", type=int, default=1_000)
    pretrain.add_argument("--max-minutes", type=float, help="stop (with a checkpoint) after this long")
    pretrain.add_argument("--device", help="cuda, cpu or mps (default: cuda if there's a GPU)")

    evaluate = commands.add_parser("evaluate", help="how well a model guesses hidden words in held-out text")
    evaluate.add_argument("model", help="a folder saved by this project, or a Hugging Face BERT name")
    evaluate.add_argument("--valid", type=Path, required=True, help="held-out tokens from prepare-data")
    evaluate.add_argument("--sequences", type=int, default=512, help="how many 128-token windows to use")

    finetune = commands.add_parser("finetune", help="teach a pre-trained BERT to sort sentences into classes")
    finetune.add_argument(
        "--model",
        default="google/bert_uncased_L-4_H-256_A-4",
        help="a folder saved by pretrain, or a Hugging Face BERT name (default: Google's BERT-Mini)",
    )
    finetune.add_argument("--out", type=Path, required=True, help="where to save the fine-tuned classifier")
    finetune.add_argument("--train", type=Path, help="training .tsv or .parquet (default: download SST-2)")
    finetune.add_argument("--valid", type=Path, help="validation .tsv or .parquet (default: download SST-2)")
    finetune.add_argument(
        "--labels", default=",".join(SST2_LABELS), help="class names, in label order (default: negative,positive)"
    )
    finetune.add_argument("--epochs", type=int, default=4)
    finetune.add_argument("--batch-size", type=int, default=32)
    finetune.add_argument("--learning-rate", type=float, default=1e-4)
    finetune.add_argument("--seed", type=int, default=0)
    finetune.add_argument("--device", help="cuda, cpu or mps (default: cuda if there's a GPU)")

    classify = commands.add_parser("classify", help="sort sentences with a classifier saved by finetune")
    classify.add_argument("texts", nargs="+", metavar="TEXT")
    classify.add_argument("--model", type=Path, required=True, help="a folder saved by finetune")
    return parser


def main(argv: Sequence[str] | None = None, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        if args.command == "tokenize":
            return _tokenize(args, out)
        if args.command == "fill":
            return _fill(args, out)
        if args.command == "prepare-data":
            return _prepare(args, out)
        if args.command == "pretrain":
            return _pretrain(args, out)
        if args.command == "evaluate":
            return _evaluate(args, out)
        if args.command == "finetune":
            return _finetune(args, out)
        if args.command == "classify":
            return _classify(args, out)
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


def _fill(args: argparse.Namespace, out: TextIO) -> int:
    # PyTorch takes a moment to import, so only the commands that need it load it.
    from minibert.checkpoint import CheckpointError, load
    from minibert.predict import fill_mask

    if args.top < 1:
        raise ValueError("--top must be at least 1.")
    try:
        model, tokenizer = load(args.model)
    except CheckpointError as exc:
        raise ValueError(str(exc)) from exc
    guesses = fill_mask(model, tokenizer, args.text, top=args.top)
    for number, options in enumerate(guesses, start=1):
        if len(guesses) > 1:
            print(f"[MASK] {number}:", file=out)
        width = max(len(guess.token) for guess in options)
        for guess in options:
            print(f"  {guess.token:<{width}}  {guess.probability:6.1%}", file=out)
    return 0


def _vocab(path: Path | None) -> Tokenizer:
    return Tokenizer.load(path, special_in_text=False) if path else Tokenizer.bert_base_uncased(special_in_text=False)


def _prepare(args: argparse.Namespace, out: TextIO) -> int:
    from minibert.data import prepare_corpus

    if not args.file.is_file():
        raise ValueError(f"No such file: {args.file}")
    started = time.perf_counter()
    with args.file.open(encoding="utf-8", errors="replace") as lines:
        counts = prepare_corpus(
            lines, _vocab(args.vocab), args.out, wikitext=args.wikitext, workers=args.workers or available_cpus()
        )
    print(
        f"Wrote {counts['tokens']:,} tokens from {counts['lines']:,} lines ({counts['documents']:,} documents) "
        f"to {args.out} in {time.perf_counter() - started:.1f} s.",
        file=out,
    )
    return 0


def _pretrain(args: argparse.Namespace, out: TextIO) -> int:
    import torch

    from minibert.data import Corpus
    from minibert.model import BertConfig, MaskedLanguageModel
    from minibert.pretrain import TrainSettings, pretrain

    tokenizer = _vocab(args.vocab)
    settings = TrainSettings(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden,
        layers=args.layers,
        heads=args.heads,
        intermediate_size=4 * args.hidden,
        max_positions=args.max_positions,
    )
    if settings.sequence_length > config.max_positions:
        raise ValueError("--sequence-length can't be longer than --max-positions.")
    torch.manual_seed(settings.seed)
    model = MaskedLanguageModel(config)
    result = pretrain(
        model,
        tokenizer,
        Corpus.open(args.train),
        Corpus.open(args.valid),
        args.run,
        settings,
        device=args.device,
        max_minutes=args.max_minutes,
        report=lambda line: print(line, file=out, flush=True),
    )
    print(
        f"Held-out text: guesses {result['accuracy']:.1%} of hidden words right (loss {result['loss']:.3f}).", file=out
    )
    return 0


def _evaluate(args: argparse.Namespace, out: TextIO) -> int:
    from minibert.checkpoint import CheckpointError, load
    from minibert.data import Corpus
    from minibert.pretrain import evaluate, evaluation_set

    try:
        model, tokenizer = load(args.model)
    except CheckpointError as exc:
        raise ValueError(str(exc)) from exc
    corpus = Corpus.open(args.valid)
    if corpus.vocab_size != len(tokenizer):
        raise ValueError(f"{args.valid} was prepared with a different vocabulary than {args.model} uses.")
    inputs, labels = evaluation_set(corpus, tokenizer, args.sequences, min(128, model.config.max_positions))
    scores = evaluate(model, inputs, labels)
    print(
        f"{args.model}: guesses {scores['accuracy']:.1%} of {scores['words']:,} hidden words right "
        f"(loss {scores['loss']:.3f}, perplexity {scores['perplexity']:,.1f}).",
        file=out,
    )
    return 0


def _finetune(args: argparse.Namespace, out: TextIO) -> int:
    from minibert.checkpoint import CheckpointError, load
    from minibert.finetune import FinetuneSettings, SentenceClassifier, finetune, read_examples, save_classifier, sst2

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if (args.train is None) != (args.valid is None):
        raise ValueError("Give both --train and --valid, or neither to use SST-2.")
    for path in (args.train, args.valid):
        if path is not None and not path.is_file():
            raise ValueError(f"No such file: {path}")
    settings = FinetuneSettings(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed
    )
    try:
        pretrained, tokenizer = load(args.model)
    except CheckpointError as exc:
        raise ValueError(str(exc)) from exc
    train = read_examples(args.train) if args.train else sst2("train")
    valid = read_examples(args.valid) if args.valid else sst2("validation")
    model = SentenceClassifier.from_pretrained(pretrained, labels, seed=args.seed)
    print(f"Fine-tuning {args.model} on {len(train):,} examples, checking on {len(valid):,}.", file=out, flush=True)
    result = finetune(
        model,
        tokenizer,
        train,
        valid,
        settings,
        device=args.device,
        report=lambda line: print(line, file=out, flush=True),
    )
    save_classifier(model.cpu(), tokenizer, args.out)
    print(f"Validation: {result['accuracy']:.1%} right. Saved to {args.out}.", file=out)
    return 0


def _classify(args: argparse.Namespace, out: TextIO) -> int:
    from minibert.finetune import classify, load_classifier

    if not args.model.is_dir():
        raise ValueError(f"No such folder: {args.model}")
    model, tokenizer = load_classifier(args.model)
    for text, prediction in zip(args.texts, classify(model, tokenizer, args.texts), strict=True):
        print(f"{prediction.label} ({prediction.probability:.0%})  {text}", file=out)
    return 0
