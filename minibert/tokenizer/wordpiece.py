"""WordPiece: split each word into the longest pieces the vocabulary knows.

"unaffable" becomes "una", "##ffa", "##ble": the "##" marks a piece that continues
a word. A word that can't be built from known pieces becomes [UNK].
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from minibert.tokenizer.text import split_words

PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
SPECIAL_TOKENS = (PAD, UNK, CLS, SEP, MASK)
CONTINUATION = "##"
MAX_WORD_CHARS = 100  # longer "words" (URLs, hashes) become [UNK], as in Google's code
_CACHE_LIMIT = 200_000
_SPECIAL_IN_TEXT = re.compile("(" + "|".join(re.escape(token) for token in SPECIAL_TOKENS) + ")")


@dataclass(frozen=True)
class Encoding:
    """One input for the model: token ids, which sentence each belongs to, and which are padding."""

    tokens: list[str]
    ids: list[int]
    type_ids: list[int]
    attention_mask: list[int]

    def __len__(self) -> int:
        return len(self.ids)


class Tokenizer:
    """BERT's tokenizer, rule for rule the same as Google's original code.

    One addition: special tokens typed in the text, like "[MASK]", are kept as special
    tokens, so "Paris is the [MASK] of France." works for fill-in-the-blank. Pass
    special_in_text=False to treat them as ordinary text, exactly as Google's code does.
    """

    def __init__(
        self, vocab: Sequence[str], *, lowercase: bool = True, split_cjk: bool = True, special_in_text: bool = True
    ) -> None:
        missing = [token for token in SPECIAL_TOKENS if token not in vocab]
        if missing:
            raise ValueError(f"The vocabulary is missing {', '.join(missing)}.")
        self.vocab = list(vocab)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.lowercase = lowercase
        self.split_cjk = split_cjk
        self.special_in_text = special_in_text
        self.pad_id, self.unk_id, self.cls_id, self.sep_id, self.mask_id = (
            self.token_to_id[token] for token in SPECIAL_TOKENS
        )
        self._pieces: dict[str, list[str]] = {}

    @classmethod
    def load(cls, path: str | Path, **options: bool) -> Tokenizer:
        """Read a vocab.txt with one token per line, the format Google's BERT checkpoints use."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls([line.strip() for line in lines], **options)

    @classmethod
    def bert_base_uncased(cls, **options: bool) -> Tokenizer:
        """Google's own 30,522-token vocabulary from BERT-Base uncased, which ships with this package."""
        vocab = resources.files("minibert") / "data" / "bert-base-uncased-vocab.txt"
        return cls([line.strip() for line in vocab.read_text(encoding="utf-8").splitlines()], **options)

    def save(self, path: str | Path) -> None:
        Path(path).write_text("".join(f"{token}\n" for token in self.vocab), encoding="utf-8")

    def __len__(self) -> int:
        return len(self.vocab)

    # Text to tokens -----------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        parts = _SPECIAL_IN_TEXT.split(text) if self.special_in_text else [text]
        tokens: list[str] = []
        for i, part in enumerate(parts):
            if i % 2:  # re.split puts the matched special tokens at odd positions
                tokens.append(part)
                continue
            for word in split_words(part, self.lowercase, self.split_cjk):
                tokens.extend(self.split_word(word))
        return tokens

    def split_word(self, word: str) -> list[str]:
        """Greedy longest-match-first, from the start of the word."""
        cached = self._pieces.get(word)
        if cached is not None:
            return cached
        pieces = self._split_word(word)
        if len(self._pieces) < _CACHE_LIMIT:
            self._pieces[word] = pieces
        return pieces

    def _split_word(self, word: str) -> list[str]:
        if len(word) > MAX_WORD_CHARS:
            return [UNK]
        pieces: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            while end > start:
                piece = word[start:end] if start == 0 else CONTINUATION + word[start:end]
                if piece in self.token_to_id:
                    break
                end -= 1
            else:
                return [UNK]  # no known piece starts here, so the whole word is unknown
            pieces.append(piece)
            start = end
        return pieces

    # Tokens to model inputs -----------------------------------------------------

    def encode(self, text: str, pair: str | None = None, max_length: int | None = None) -> Encoding:
        """[CLS] text [SEP], or [CLS] text [SEP] pair [SEP], cut down to `max_length` if given.

        When a pair is too long, tokens come off the end of whichever side is longer,
        one at a time, as in Google's fine-tuning code.
        """
        first = self.tokenize(text)
        second = self.tokenize(pair) if pair is not None else None
        if max_length is not None:
            specials = 3 if second is not None else 2
            if max_length < specials:
                raise ValueError(f"max_length must be at least {specials}, to leave room for [CLS] and [SEP].")
            first, second = _truncate(first, second, max_length - specials)

        tokens = [CLS, *first, SEP]
        type_ids = [0] * len(tokens)
        if second is not None:
            tokens += [*second, SEP]
            type_ids += [1] * (len(second) + 1)
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        return Encoding(tokens=tokens, ids=ids, type_ids=type_ids, attention_mask=[1] * len(ids))

    def encode_batch(
        self,
        texts: Sequence[str],
        pairs: Sequence[str] | None = None,
        max_length: int | None = None,
    ) -> list[Encoding]:
        """Encode several inputs and pad them with [PAD] to the length of the longest."""
        if pairs is not None and len(pairs) != len(texts):
            raise ValueError("Give one pair for each text.")
        encodings = [
            self.encode(text, pairs[i] if pairs is not None else None, max_length) for i, text in enumerate(texts)
        ]
        longest = max((len(e) for e in encodings), default=0)
        return [self.pad(e, longest) for e in encodings]

    def pad(self, encoding: Encoding, length: int) -> Encoding:
        extra = length - len(encoding)
        if extra <= 0:
            return encoding
        return Encoding(
            tokens=encoding.tokens + [PAD] * extra,
            ids=encoding.ids + [self.pad_id] * extra,
            type_ids=encoding.type_ids + [0] * extra,
            attention_mask=encoding.attention_mask + [0] * extra,
        )

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Ids back to text. Pieces rejoin into words; spacing around punctuation isn't recoverable."""
        special = set(SPECIAL_TOKENS) if skip_special else set()
        tokens = [self.vocab[i] for i in ids if self.vocab[i] not in special]
        return " ".join(tokens).replace(f" {CONTINUATION}", "")


def _truncate(first: list[str], second: list[str] | None, budget: int) -> tuple[list[str], list[str] | None]:
    if second is None:
        return first[:budget], None
    first, second = list(first), list(second)
    while len(first) + len(second) > budget:
        if len(first) > len(second):
            first.pop()
        else:
            second.pop()
    return first, second
