"""The 30-symbol character vocab, the same mapping as ``training/src/training/vocab.py``.

PROVISIONAL until ``shared/specs`` freezes it; a copy because the verticals share no
code. The decode golden case (``shared/golden/decode/case.json``) lists training's
symbols, and the tests check this one against it.
"""

from __future__ import annotations

BLANK = "<blank>"
EOS = "<eos>"

CHARS: tuple[str, ...] = (BLANK, *"ABCDEFGHIJKLMNOPQRSTUVWXYZ '", EOS)
VOCAB_SIZE = len(CHARS)
BLANK_IDX = 0
EOS_IDX = CHARS.index(EOS)
