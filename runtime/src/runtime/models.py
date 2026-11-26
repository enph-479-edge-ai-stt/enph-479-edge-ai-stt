"""The AM and the char LM as the decoder sees them, both computed on the fabric.

``FabricAcousticModel`` turns a clip's int8 feature frames into int8 scores, one AM job
per frame. ``FabricCharModel`` is the beam search's ``CharLM`` (``shared.decode.beam``):
a state is a ``Slot``, one of the fabric's 256 char-LM context slots, and stepping a
state is one LM job that reads the parent's slot and writes a free one. A slot goes back
to the ``SlotPool`` when the search drops the last prefix holding it.

When the search steps the CM, the live states are the last beam's (up to 128) and the new
children (up to 128), so 256 slots always suffice. That needs the search to let go of the
root prefix like any other, which ``shared/cpp/beam_cpp.cpp`` does.

Each search step's children go to the fabric as one batch (``Fabric.lm``), and a clip's
frames as one (``Fabric.am``), so the C++ job loop runs them without Python in between.
Takes any object with ``Fabric``'s ``clear_am``, ``clear_slot``, ``am`` and ``lm``, so
the tests run these on a stand-in without the board.
"""

from __future__ import annotations

import time

import numpy as np
from shared import vocab

# The fabric's int8 scores are the logits * 2**SCORE_FRAC (shared/specs/weight_image.md,
# "Activations and cell").
SCORE_FRAC = 2
N_SLOTS = 256  # fabric.N_SLOTS, repeated so this module imports without pynq


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Along the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def logits(scores: np.ndarray) -> np.ndarray:
    """The fabric's int8 scores as float logits."""
    return scores.astype(np.float64) * 2.0**-SCORE_FRAC


class SlotPool:
    """The free char-LM slots."""

    def __init__(self, n: int = N_SLOTS) -> None:
        """All ``n`` slots start free."""
        self.n = n
        self._free = list(range(n))
        self.peak = 0

    @property
    def live(self) -> int:
        """Slots held by a state."""
        return self.n - len(self._free)

    def take(self) -> Slot:
        """A free slot, held until the returned ``Slot`` is dropped."""
        if not self._free:
            raise RuntimeError(f"all {self.n} char-LM slots are live")
        slot = Slot(self._free.pop(), self._free)
        self.peak = max(self.peak, self.live)
        return slot


class Slot:
    """A char-LM state: slot ``n`` of the context memory, freed when dropped."""

    __slots__ = ("_free", "n")

    def __init__(self, n: int, free: list[int]) -> None:
        """Hold slot ``n`` of the pool whose free list is ``free``."""
        self.n, self._free = n, free

    def __del__(self) -> None:
        """Give the slot back."""
        self._free.append(self.n)


class FabricCharModel:
    """The char LM on the fabric, behind the search's ``CharLM`` interface."""

    def __init__(self, fabric, pool: SlotPool | None = None) -> None:
        """Zero the counters ``transcribe`` reports."""
        self.fabric = fabric
        self.pool = pool or SlotPool()
        self.seconds = 0.0
        self.steps = self.calls = 0

    def start(self) -> tuple[Slot, np.ndarray]:
        """Feed EOS from the zero state, as at a sentence boundary in training."""
        slot = self.pool.take()
        self.fabric.clear_slot(slot.n)
        scores = self.fabric.lm([slot.n], [slot.n], [vocab.EOS_IDX])  # in place
        return slot, log_softmax(logits(scores[0]))

    def step(self, states: list[Slot], chars: list[int]) -> tuple[list[Slot], np.ndarray]:
        """Feed ``chars[n]`` to ``states[n]``: one fabric job each, sent as one batch."""
        t0 = time.perf_counter()
        new = [self.pool.take() for _ in chars]
        scores = self.fabric.lm([s.n for s in states], [c.n for c in new], chars)
        self.seconds += time.perf_counter() - t0
        self.steps += len(chars)
        self.calls += 1
        return new, log_softmax(logits(scores))


class FabricAcousticModel:
    """The AM on the fabric: a clip's int8 feature frames in, int8 scores out."""

    def __init__(self, fabric) -> None:
        """Run on ``fabric``."""
        self.fabric = fabric

    def scores(self, q: np.ndarray) -> np.ndarray:
        """int8 frames ``[T, N_IN]`` -> int8 scores ``[T, V]``, from a cleared context."""
        self.fabric.clear_am()
        return self.fabric.am(q)
