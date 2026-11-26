"""The fabric-backed AM and char LM against a stand-in fabric, through the Python search."""

import zlib

import numpy as np
import pytest
from shared import vocab
from shared.decode.beam import Config, beam_search

from runtime.models import (
    FabricAcousticModel,
    FabricCharModel,
    SlotPool,
    log_softmax,
    logits,
)


def fake_scores(history: tuple[int, ...]) -> np.ndarray:
    """Some int8 scores that depend on every character fed since the clear."""
    rng = np.random.default_rng(zlib.crc32(bytes(history)))
    return rng.integers(-40, 40, vocab.VOCAB_SIZE).astype(np.int8)


class FakeFabric:
    """A slot holds the characters fed since it was cleared; the AM echoes the frame."""

    def __init__(self):
        self.slots, self.am_clears = {}, 0

    def clear_slot(self, n):
        self.slots[n] = ()

    def lm(self, src, dst, chars):  # in order, so a later job can read an earlier one's slot
        out = []
        for s, d, c in zip(src, dst, chars, strict=True):
            self.slots[d] = self.slots[s] + (c,)
            out.append(fake_scores(self.slots[d]))
        return np.stack(out)

    def clear_am(self):
        self.am_clears += 1

    def am(self, q):
        return q[:, : vocab.VOCAB_SIZE].astype(np.int8)


class HistoryCM:
    """The same char LM with no slots: a state is its history."""

    def start(self):
        h = (vocab.EOS_IDX,)
        return h, log_softmax(logits(fake_scores(h)))

    def step(self, states, chars):
        new = [s + (c,) for s, c in zip(states, chars, strict=True)]
        return new, log_softmax(logits(np.stack([fake_scores(h) for h in new])))


def test_pool_reuses_and_runs_out():
    pool = SlotPool(2)
    a, b = pool.take(), pool.take()
    assert {a.n, b.n} == {0, 1} and pool.live == 2 and pool.peak == 2
    with pytest.raises(RuntimeError, match="all 2 char-LM slots are live"):
        pool.take()
    n = a.n
    del a
    assert pool.live == 1 and pool.take().n == n


def test_cm_on_slots_matches_history_through_the_search():
    rng = np.random.default_rng(0)
    logp = log_softmax(rng.normal(0, 3, (60, vocab.VOCAB_SIZE)))
    cfg = Config(beam=8, alpha=0.5, beta=0.0, gamma=1.0)
    pool = SlotPool(64)
    cm = FabricCharModel(FakeFabric(), pool)
    assert beam_search(logp, cfg, cm) == beam_search(logp, cfg, HistoryCM())
    assert cm.steps > 0 and 0 < pool.peak <= 2 * cfg.beam + 1
    assert pool.live == 0  # every slot back once the search is done


def test_am_clears_then_runs_every_frame():
    fabric = FakeFabric()
    q = np.arange(5 * 123).reshape(5, 123).astype(np.int8)
    out = FabricAcousticModel(fabric).scores(q)
    assert fabric.am_clears == 1 and out.shape == (5, vocab.VOCAB_SIZE)
    assert (out == q[:, : vocab.VOCAB_SIZE]).all()
