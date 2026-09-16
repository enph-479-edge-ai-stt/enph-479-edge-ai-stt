"""Fixed character vocabulary for the acoustic model and CTC decoding.

PROVISIONAL (smoke-test) vocab. LibriSpeech transcripts are strictly A-Z, space
and apostrophe (verified on dev/test-clean), so the character set is 28 symbols;
with an end-of-sentence marker and the CTC blank that is 30. The paper's 31 (it
carried 3 WSJ punctuation symbols LibriSpeech does not have) is the alternative
still open in ``shared/specs``. Two hard rules that outlive the smoke test:

- The blank stays at **index 0** (``nn.CTCLoss`` default, and the index every
  component -- decoder, RTL -- must agree on).
- This mapping is frozen once training artifacts exist. Changing it invalidates
  every checkpoint, golden vector, and packed weight file.

Pure stdlib, no torch, so it imports on a bare runtime and is cheap to test.
"""

from __future__ import annotations

from collections.abc import Iterable

BLANK = "<blank>"
EOS = "<eos>"

# Index 0 is the CTC blank by convention. 26 letters + space + apostrophe = 28
# real characters, then EOS, for 30 symbols total.
CHARS: tuple[str, ...] = (BLANK, *"ABCDEFGHIJKLMNOPQRSTUVWXYZ '", EOS)
VOCAB_SIZE = len(CHARS)
BLANK_IDX = 0
EOS_IDX = CHARS.index(EOS)

_CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
_IDX_TO_CHAR = dict(enumerate(CHARS))
# The characters a transcript may legally contain (excludes the two markers).
LEGAL_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")


def normalize(text: str) -> str:
    """Upper-case and drop any character not in the vocab.

    LibriSpeech is already clean; this is defensive so an unexpected symbol
    never becomes a label.
    """
    return "".join(c for c in text.upper() if c in LEGAL_CHARS)


def encode(text: str, *, add_eos: bool = False) -> list[int]:
    """Map a normalized transcript string to label indices.

    ``text`` must already be normalized; an out-of-vocab character raises
    ``KeyError``. CTC targets normally omit EOS.
    """
    idx = [_CHAR_TO_IDX[c] for c in text]
    if add_eos:
        idx.append(EOS_IDX)
    return idx


def decode(indices: Iterable[int]) -> str:
    """Indices -> string, with no CTC collapse. Blank and EOS render as nothing."""
    out = []
    for i in indices:
        c = _IDX_TO_CHAR[int(i)]
        if c not in (BLANK, EOS):
            out.append(c)
    return "".join(out)


def collapse(indices: Iterable[int]) -> str:
    """Greedy CTC collapse of a per-frame argmax path.

    Merge runs of identical indices, then drop blanks, then map to characters.
    A blank between two equal characters preserves a double letter
    (``L _ L`` -> ``LL``), while ``L L`` with no blank collapses to a single ``L``.
    """
    kept: list[int] = []
    prev: int | None = None
    for raw in indices:
        i = int(raw)
        if i != prev:  # a change ends the current run
            if i != BLANK_IDX:
                kept.append(i)
            prev = i
    return decode(kept)
