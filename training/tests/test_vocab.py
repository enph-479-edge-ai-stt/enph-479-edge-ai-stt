"""Vocab + CTC greedy-collapse tests. Pure stdlib, no torch, always run in CI."""

from __future__ import annotations

import pytest

from training import vocab


def test_blank_is_index_zero_and_size_is_30():
    assert vocab.BLANK_IDX == 0
    assert vocab.VOCAB_SIZE == 30


def test_encode_decode_round_trip():
    text = "HELLO WORLD"
    assert vocab.decode(vocab.encode(text)) == text


def test_encode_add_eos():
    idx = vocab.encode("HI", add_eos=True)
    assert idx[-1] == vocab.EOS_IDX
    assert vocab.decode(idx) == "HI"  # EOS renders as nothing


def test_normalize_uppercases_and_drops_out_of_vocab():
    assert vocab.normalize("Hello, world!") == "HELLO WORLD"
    assert vocab.normalize("it's") == "IT'S"


def test_encode_rejects_out_of_vocab():
    with pytest.raises(KeyError):
        vocab.encode("hello")  # lowercase is not in the vocab; normalize first


def test_collapse_merges_repeats_then_drops_blanks():
    B = vocab.BLANK_IDX
    h, e, o, el = (vocab.encode(c)[0] for c in "HEOL")
    # H H _ E _ L L _ L O  ->  H E L L O
    path = [h, h, B, e, B, el, el, B, el, o]
    assert vocab.collapse(path) == "HELLO"


def test_collapse_blank_preserves_double_letter():
    B = vocab.BLANK_IDX
    (t,) = vocab.encode("T")
    (o,) = vocab.encode("O")
    assert vocab.collapse([t, o, B, o]) == "TOO"  # blank keeps the two O's
    assert vocab.collapse([t, o, o]) == "TO"  # no blank -> the O's merge


def test_collapse_all_blank_is_empty():
    assert vocab.collapse([vocab.BLANK_IDX] * 5) == ""
