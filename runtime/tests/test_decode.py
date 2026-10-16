"""Beam search tests: the golden case training wrote, and the search's basic behaviour.

The golden case is what makes this a second copy of training's decoder rather than a
second decoder: same inputs, same N-best, same scores. It needs KenLM (the ``wlm``
extra) and skips without it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from runtime import vocab
from runtime.decode.beam import Config, beam_search

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "shared" / "golden" / "decode"
NO_PRUNE = Config(beam=16, prune=-math.inf)


class TableCharLM:
    """A bigram CM: the state is the last character, its log-probs are that row of ``table``."""

    def __init__(self, table):
        self.table = table

    def start(self):
        return vocab.EOS_IDX, self.table[vocab.EOS_IDX]

    def step(self, states, chars):
        return chars, [self.table[c] for c in chars]


def _spell(path: str, p: float = 0.97) -> np.ndarray:
    """One confident frame per symbol of ``path`` ("_" for the blank)."""
    out = np.full((len(path), vocab.VOCAB_SIZE), (1 - p) / (vocab.VOCAB_SIZE - 1))
    for t, sym in enumerate(path):
        out[t, vocab.BLANK_IDX if sym == "_" else vocab.CHARS.index(sym)] = p
    return np.log(out)


def _golden():
    return json.loads((GOLDEN_DIR / "case.json").read_text(encoding="utf-8"))


def test_vocab_matches_training():
    assert tuple(_golden()["chars"]) == vocab.CHARS


def test_golden_case_reproduces():
    pytest.importorskip("kenlm")
    from runtime.decode.wlm import KenLM

    case = _golden()
    am = np.array(case["am_logp"])
    cm = TableCharLM(case["cm_table"])
    wlm = KenLM(GOLDEN_DIR / "words.arpa")
    for c in case["cases"]:
        got = beam_search(
            am, Config(**c["config"]), cm if c["cm"] else None, wlm if c["wlm"] else None
        )
        assert [t for t, _ in got] == [t for t, _ in c["beam"]], c["name"]
        assert [s for _, s in got] == pytest.approx([s for _, s in c["beam"]], rel=1e-12), c["name"]


def test_confident_path_decodes_to_its_collapse():
    assert beam_search(_spell("HH_E_LL_LL_O_ _WW_O_R_L_DD"), NO_PRUNE)[0][0] == "HELLO WORLD"


def test_repeat_needs_a_blank():
    assert beam_search(_spell("LL"), NO_PRUNE)[0][0] == "L"
    assert beam_search(_spell("L_L"), NO_PRUNE)[0][0] == "LL"


def test_no_leading_or_double_space():
    for text, _ in beam_search(_spell(" _A_  _B_ ", p=0.9), NO_PRUNE):
        assert not text.startswith(" ")
        assert "  " not in text
