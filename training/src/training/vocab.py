"""Character vocabulary for the acoustic model and CTC decoding.

PROVISIONAL until ``shared/specs`` is frozen. LibriSpeech transcripts contain
only A-Z, space and apostrophe; with an end-of-sentence marker and the CTC
blank that is 30 symbols. The blank is index 0 (``nn.CTCLoss``'s default).
Changing this mapping invalidates every trained checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable

BLANK = "<blank>"
EOS = "<eos>"

CHARS: tuple[str, ...] = (BLANK, *"ABCDEFGHIJKLMNOPQRSTUVWXYZ '", EOS)
VOCAB_SIZE = len(CHARS)
BLANK_IDX = 0

_CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
EOS_IDX = _CHAR_TO_IDX[EOS]


def encode(text: str) -> list[int]:
    """Transcript -> label indices. Raises ``KeyError`` on an out-of-vocab character."""
    return [_CHAR_TO_IDX[c] for c in text]


def decode(indices: Iterable[int]) -> str:
    """Label indices -> string, with no CTC collapse. Blank and EOS render as nothing."""
    chars = (CHARS[int(i)] for i in indices)
    return "".join(c for c in chars if c not in (BLANK, EOS))


def collapse(indices: Iterable[int]) -> str:
    """Greedy CTC collapse of a per-frame argmax path: merge repeats, then drop blanks.

    A blank between two equal characters keeps both (``L _ L`` -> ``LL``), while
    ``L L`` with no blank collapses to one ``L``.
    """
    kept: list[int] = []
    prev = None
    for raw in indices:
        i = int(raw)
        if i != prev and i != BLANK_IDX:
            kept.append(i)
        prev = i
    return decode(kept)
