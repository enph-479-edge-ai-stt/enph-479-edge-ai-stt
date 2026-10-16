"""Word-LM adapter tests, on the golden case's tiny trigram. Skips without KenLM."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("kenlm")

from training.decode.golden import GOLDEN_DIR  # noqa: E402
from training.wlm.model import KenLM, perplexity  # noqa: E402

LN10 = math.log(10)


@pytest.fixture(scope="module")
def lm():
    return KenLM(GOLDEN_DIR / "words.arpa")


def test_words_are_the_unigrams_sorted(lm):
    assert lm.words == ["A", "CAT", "HAT", "MAT", "ON", "SAT", "THE"]


def test_word_by_word_matches_whole_sentence(lm):
    state = lm.start()
    total = 0.0
    for word in ["THE", "CAT", "SAT", "ON", "THE", "MAT"]:
        logp, state = lm.score(state, word)
        total += logp
    total += lm.end(state)
    assert total == pytest.approx(lm.model.score("THE CAT SAT ON THE MAT") * LN10)


def test_oov_word_pays_the_penalty(lm):
    unk, _ = lm.score(lm.start(), "DOG")
    expected = lm.model.score("DOG", bos=True, eos=False) * LN10 + lm.oov_penalty
    assert unk == pytest.approx(expected)
    assert lm.oov_penalty == pytest.approx(-10 * LN10)


def test_is_prefix(lm):
    assert lm.is_prefix("")
    assert lm.is_prefix("C")
    assert lm.is_prefix("CAT")
    assert not lm.is_prefix("CATS")
    assert not lm.is_prefix("B")
    assert not lm.is_prefix("Z")  # past the last word


def test_perplexity_matches_kenlm(lm):
    ppl, oov = perplexity(lm, ["THE CAT SAT ON THE MAT"])
    assert ppl == pytest.approx(lm.model.perplexity("THE CAT SAT ON THE MAT"))
    assert oov == 0.0
    assert perplexity(lm, ["THE DOG"])[1] == 0.5
