"""KenLM trigram -> the decoder's word-LM interface.

The decoder scores a word when a hypothesis completes it (a space, or the end of the
utterance) and carries a KenLM state per hypothesis. While a word is being spelled it
asks whether any word in the LM starts that way, so a non-word pays its penalty as soon
as it can't be a word any more. Scores are natural log; KenLM's are log10.

``training/src/training/wlm/model.py`` is this adapter again (it also has the word
perplexity training reports); both are checked against ``shared/golden/decode/``.
"""

from __future__ import annotations

import bisect
import gzip
import math
from pathlib import Path

import kenlm

_LN10 = math.log(10)
# KenLM gives a word outside the LM's vocab the <UNK> probability, about 10**-2.3 for
# the OpenSLR trigram: cheap enough that the beam would happily spell out non-words.
# Each one also pays this, in log10 (pyctcdecode's default).
OOV_PENALTY = -10.0


def _unigrams(arpa: Path) -> list[str]:
    """The words of an ARPA file (its 1-grams, without ``<s>``, ``</s>``, ``<unk>``), sorted."""
    words = []
    with (gzip.open if arpa.suffix == ".gz" else open)(arpa, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("\\1-grams:"):
                break
        for line in f:
            fields = line.split()
            if not fields:
                break
            if not fields[1].startswith("<"):
                words.append(fields[1])
    return sorted(words)


class KenLM:
    """An ARPA model scoring one word at a time, in natural log."""

    oov_penalty = OOV_PENALTY * _LN10  # in every ``score`` of a word outside the vocab

    def __init__(self, arpa: str | Path) -> None:
        """Load the model (KenLM reads a gzipped ARPA directly) and its word list."""
        self.model = kenlm.Model(str(arpa))
        self.words = _unigrams(Path(arpa))

    def start(self) -> kenlm.State:
        """The state at the start of a sentence (after ``<s>``)."""
        state = kenlm.State()
        self.model.BeginSentenceWrite(state)
        return state

    def score(self, state: kenlm.State, word: str) -> tuple[float, kenlm.State]:
        """Log-prob of ``word`` after ``state``, and the state after it."""
        out = kenlm.State()
        logp = self.model.BaseScore(state, word, out)
        if word not in self.model:
            logp += OOV_PENALTY
        return logp * _LN10, out

    def is_prefix(self, text: str) -> bool:
        """Whether some word in the LM starts with ``text`` (or is ``text``)."""
        i = bisect.bisect_left(self.words, text)
        return i < len(self.words) and self.words[i].startswith(text)

    def end(self, state: kenlm.State) -> float:
        """Log-prob of the sentence ending after ``state`` (``</s>``)."""
        return self.model.BaseScore(state, "</s>", kenlm.State()) * _LN10
