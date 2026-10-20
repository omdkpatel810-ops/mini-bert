from __future__ import annotations

import unicodedata
from importlib import resources

import google_tokenization as google
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from tokenizers import BertWordPieceTokenizer

from minibert.tokenizer import SPECIAL_TOKENS, Tokenizer
from minibert.tokenizer.wordpiece import MAX_WORD_CHARS

VOCAB = str(resources.files("minibert") / "data" / "bert-base-uncased-vocab.txt")


@pytest.fixture(scope="module")
def bert() -> Tokenizer:
    return Tokenizer.bert_base_uncased()


def tiny(*tokens: str, **options: bool) -> Tokenizer:
    return Tokenizer([*SPECIAL_TOKENS, *tokens], **options)


# Words to pieces --------------------------------------------------------------


def test_splits_words_into_the_longest_known_pieces(bert):
    assert bert.tokenize("unaffable") == ["una", "##ffa", "##ble"]
    assert bert.tokenize("Tokenization works!") == ["token", "##ization", "works", "!"]


def test_a_word_with_no_known_start_is_unknown():
    tokenizer = tiny("un", "##known")
    assert tokenizer.tokenize("unknown xunknown") == ["un", "##known", "[UNK]"]


def test_a_word_that_cannot_be_finished_is_unknown_as_a_whole():
    tokenizer = tiny("un", "##kn")
    assert tokenizer.tokenize("unknown") == ["[UNK]"]


def test_very_long_words_are_unknown(bert):
    assert bert.tokenize("a" * MAX_WORD_CHARS) != ["[UNK]"]
    assert bert.tokenize("a" * (MAX_WORD_CHARS + 1)) == ["[UNK]"]


def test_remembers_words_it_has_split():
    tokenizer = tiny("hug", "##s")
    assert tokenizer.split_word("hugs") is tokenizer.split_word("hugs")


def test_special_tokens_in_the_text_stay_special(bert):
    assert bert.tokenize("Paris is the [MASK] of France.") == ["paris", "is", "the", "[MASK]", "of", "france", "."]
    plain = Tokenizer.bert_base_uncased(special_in_text=False)
    assert plain.tokenize("the [MASK]") == ["the", "[", "mask", "]"]


# Model inputs -----------------------------------------------------------------


def test_encode_one_text(bert):
    encoding = bert.encode("Hello world")

    assert encoding.tokens == ["[CLS]", "hello", "world", "[SEP]"]
    assert encoding.ids == [101, 7592, 2088, 102]
    assert encoding.type_ids == [0, 0, 0, 0]
    assert encoding.attention_mask == [1, 1, 1, 1]
    assert len(encoding) == 4


def test_encode_a_pair(bert):
    encoding = bert.encode("Where is Winnipeg?", "In Manitoba.")

    assert encoding.tokens == ["[CLS]", "where", "is", "winnipeg", "?", "[SEP]", "in", "manitoba", ".", "[SEP]"]
    assert encoding.type_ids == [0] * 6 + [1] * 4


def test_truncates_a_single_text(bert):
    assert bert.encode("one two three four", max_length=4).tokens == ["[CLS]", "one", "two", "[SEP]"]


def test_truncates_the_longer_side_of_a_pair_first(bert):
    encoding = bert.encode("one two three four five", "six seven", max_length=8)

    assert encoding.tokens == ["[CLS]", "one", "two", "three", "[SEP]", "six", "seven", "[SEP]"]
    # When the sides are even, the second loses a token, as in Google's code.
    assert bert.encode("one two", "three four", max_length=6).tokens == [
        "[CLS]",
        "one",
        "two",
        "[SEP]",
        "three",
        "[SEP]",
    ]


@pytest.mark.parametrize(("pair", "smallest"), [(None, 2), ("second", 3)])
def test_max_length_must_leave_room_for_special_tokens(bert, pair, smallest):
    bert.encode("text", pair, max_length=smallest)
    with pytest.raises(ValueError, match=f"max_length must be at least {smallest}"):
        bert.encode("text", pair, max_length=smallest - 1)


def test_encode_batch_pads_to_the_longest(bert):
    short, long = bert.encode_batch(["Hi", "Hello there, world"])

    assert len(short) == len(long) == 6
    assert short.tokens == ["[CLS]", "hi", "[SEP]", "[PAD]", "[PAD]", "[PAD]"]
    assert short.ids[3:] == [0, 0, 0]
    assert short.attention_mask == [1, 1, 1, 0, 0, 0]
    assert long.attention_mask == [1] * 6


def test_encode_batch_with_pairs(bert):
    first, second = bert.encode_batch(["a", "b c"], pairs=["x y z", "w"])

    assert first.type_ids == [0, 0, 0, 1, 1, 1, 1]
    assert second.type_ids == [0, 0, 0, 0, 1, 1, 0]


def test_encode_batch_needs_one_pair_per_text(bert):
    with pytest.raises(ValueError, match="one pair for each text"):
        bert.encode_batch(["a", "b"], pairs=["x"])


def test_encode_batch_of_nothing(bert):
    assert bert.encode_batch([]) == []


def test_padding_never_shortens(bert):
    encoding = bert.encode("Hello world")
    assert bert.pad(encoding, 2) is encoding


def test_decode(bert):
    ids = bert.encode("Tokenization is unaffable.").ids

    assert bert.decode(ids) == "tokenization is unaffable ."
    assert bert.decode(ids, skip_special=False) == "[CLS] tokenization is unaffable . [SEP]"


# The vocabulary -----------------------------------------------------------------


def test_googles_vocabulary_ships_with_the_package(bert):
    assert len(bert) == 30_522
    assert (bert.pad_id, bert.unk_id, bert.cls_id, bert.sep_id, bert.mask_id) == (0, 100, 101, 102, 103)


def test_save_and_load(tmp_path, bert):
    path = tmp_path / "vocab.txt"
    bert.save(path)

    assert Tokenizer.load(path).vocab == bert.vocab
    assert path.read_text(encoding="utf-8").splitlines()[:2] == ["[PAD]", "[unused0]"]


def test_a_vocabulary_needs_the_special_tokens():
    with pytest.raises(ValueError, match=r"missing \[CLS\], \[MASK\]"):
        Tokenizer(["[PAD]", "[UNK]", "[SEP]", "hello"])


# Against the references ---------------------------------------------------------------


@pytest.fixture(scope="module")
def references():
    return {
        "google": google.FullTokenizer(VOCAB, do_lower_case=True),
        "google_cased": google.FullTokenizer(VOCAB, do_lower_case=False),
        "hugging_face": BertWordPieceTokenizer(VOCAB, lowercase=True),
    }


@settings(max_examples=3000, deadline=None, suppress_health_check=list(HealthCheck))
@given(text=st.text(max_size=80))
def test_tokens_match_googles_code_exactly(references, text):
    ours = Tokenizer.bert_base_uncased(special_in_text=False)
    assert ours.tokenize(text) == references["google"].tokenize(text)


@settings(max_examples=1000, deadline=None, suppress_health_check=list(HealthCheck))
@given(text=st.text(max_size=80))
def test_cased_tokens_match_googles_code_exactly(references, text):
    ours = Tokenizer.bert_base_uncased(special_in_text=False, lowercase=False)
    assert ours.tokenize(text) == references["google_cased"].tokenize(text)


# Hugging Face's Rust code uses older Unicode tables than Python, so for characters added in
# recent years (an Adlam accent from 2016, a dash from 2017) the two disagree about what's an
# accent or punctuation. The comparison sticks to characters that already existed in Unicode
# 3.2 (2002), which both know. Ours matches Google's code on everything (tests above).
OLD_CHARACTERS = st.characters(max_codepoint=0x9FFF).filter(lambda char: unicodedata.ucd_3_2_0.category(char) != "Cn")


@settings(max_examples=1500, deadline=None, suppress_health_check=list(HealthCheck))
@given(text=st.text(OLD_CHARACTERS, max_size=80))
def test_ids_match_hugging_face(references, text):
    # Hugging Face also deletes private-use characters, which Google's code keeps (tested below).
    assume(not any(unicodedata.category(char) == "Co" for char in text))
    ours = Tokenizer.bert_base_uncased()
    assert ours.encode(text).ids == references["hugging_face"].encode(text).ids


def test_hugging_face_deletes_private_use_characters_but_google_keeps_them(references):
    text = "caf\U000f0000e"
    assert Tokenizer.bert_base_uncased().tokenize(text) == references["google"].tokenize(text) == ["[UNK]"]
    assert references["hugging_face"].encode(text, add_special_tokens=False).tokens == ["cafe"]


def test_hugging_face_has_a_typo_in_one_cjk_range(references):
    # Google's code splits off ideographs from U+2B820; Hugging Face's Rust code starts that
    # range at U+2B920, so the 256 characters in between stay glued to their neighbours.
    text = "a\U0002b820b"
    assert Tokenizer.bert_base_uncased().tokenize(text) == references["google"].tokenize(text) == ["a", "[UNK]", "b"]
    assert references["hugging_face"].encode(text, add_special_tokens=False).tokens == ["[UNK]"]
