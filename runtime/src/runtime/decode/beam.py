"""CTC prefix beam search fused with a character LM and a word LM (the paper's N-best search).

The beam holds up to ``beam`` prefixes (partial transcripts). Every frame, each one is
extended by a blank, by repeating its last character, or by a new character, and the
probabilities of the alignments that collapse to the same prefix are summed (Hannun et
al., 2014). A prefix is ranked by

    log P_ctc(prefix) + alpha * log P_cm(prefix) + beta * log P_wlm(words) + gamma * #words

The character model (CM) scores every character as it is appended. The word LM (WLM)
scores a word when it is completed, by a space or by the end of the utterance, which
also closes the CM with EOS and the WLM with ``</s>``. A word outside the WLM's vocab
pays a penalty in that score, and the same penalty again for every character it is
spelled past the point where no word in the vocab starts that way. Without the second
part, a prefix that runs on without spaces (one long non-word) would put off every WLM
cost to the end and win. A space can't start a transcript or follow another space.

The LMs come in through small interfaces (``CharLM``, ``WordLM``), so the same search
runs the CM on the fabric here and in torch in training, and runs with either LM left
out. The search itself is plain Python floats.

``training/src/training/decode/beam.py`` is this file again, line for line: the
verticals share no code. Training's is the reference that writes ``shared/golden/decode/``
and this one must reproduce it; change them together.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from runtime import vocab

NEG_INF = -math.inf

_SPACE = " "
# What a prefix can be extended by: the characters, not the blank or EOS (the AM never emits EOS).
_EXTEND = [i for i, c in enumerate(vocab.CHARS) if len(c) == 1]
_INDEX = {vocab.CHARS[i]: i for i in _EXTEND}


@dataclass(frozen=True)
class Config:
    """Search settings and LM weights. The weights are tuned on dev-clean."""

    beam: int = 128  # prefixes kept per frame (the paper's N)
    alpha: float = 0.5  # CM weight
    beta: float = 1.0  # WLM weight
    gamma: float = 0.0  # bonus per word, against the LMs' preference for short output
    prune: float = math.log(1e-3)  # a frame only tries characters at least this likely


class CharLM(Protocol):
    """Next-character log-probs, one state per prefix."""

    def start(self) -> tuple[Any, Sequence[float]]:
        """The state after EOS (the start of a sentence) and its next-char log-probs ``[V]``."""
        ...

    def step(self, states: list[Any], chars: list[int]) -> tuple[list[Any], Sequence]:
        """Feed ``chars[n]`` to ``states[n]``: the new states and their log-probs ``[N, V]``."""
        ...


class WordLM(Protocol):
    """Log-probs of whole words, natural log, one state per prefix."""

    def start(self) -> Any:
        """The state at the start of a sentence."""
        ...

    def score(self, state: Any, word: str) -> tuple[float, Any]:
        """Log-prob of ``word`` after ``state``, and the state after it."""
        ...

    def end(self, state: Any) -> float:
        """Log-prob of the sentence ending after ``state``."""
        ...

    def is_prefix(self, text: str) -> bool:
        """Whether some word in the vocab starts with ``text`` (or is ``text``)."""
        ...

    @property
    def oov_penalty(self) -> float:
        """What ``score`` adds for a word outside the vocab, and ``beam_search`` per character."""
        ...


@dataclass
class _Prefix:
    """What depends on the prefix alone, not on how it is aligned to the frames."""

    text: str
    lm: float  # the weighted LM terms and word bonuses so far
    word_start: int  # where the word being spelled starts in ``text``
    wstate: Any
    cstate: Any = None
    clogp: list[float] | None = None  # CM log-probs of the next character
    parent: _Prefix | None = None  # with ``char``, what ``cstate`` is computed from
    char: int = 0


def _logadd(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if b == NEG_INF:
        return a
    return a + math.log1p(math.exp(b - a))


def _extend(pre: _Prefix, c: int, cfg: Config, wlm: WordLM | None) -> _Prefix:
    """``pre`` with character ``c`` appended. Its CM state is computed later, in a batch."""
    ch = vocab.CHARS[c]
    lm = pre.lm
    if pre.clogp is not None:
        lm += cfg.alpha * pre.clogp[c]
    wstate, word_start = pre.wstate, pre.word_start
    if ch == _SPACE:
        lm += cfg.gamma
        if wlm is not None:
            logp, wstate = wlm.score(wstate, pre.text[word_start:])
            lm += cfg.beta * logp
        word_start = len(pre.text) + 1
    elif wlm is not None and not wlm.is_prefix(pre.text[word_start:] + ch):
        lm += cfg.beta * wlm.oov_penalty
    parent = pre if pre.clogp is not None else None
    return _Prefix(pre.text + ch, lm, word_start, wstate, parent=parent, char=c)


def _advance_cm(cm: CharLM, prefixes: list[_Prefix]) -> None:
    """Run the CM one step for every prefix that doesn't have its state yet, as one batch."""
    todo = [p for p in prefixes if p.clogp is None]
    if not todo:
        return
    states, logps = cm.step([p.parent.cstate for p in todo], [p.char for p in todo])
    logps = np.asarray(logps, dtype=np.float64).tolist()
    for p, state, logp in zip(todo, states, logps, strict=True):
        p.cstate, p.clogp, p.parent = state, logp, None


def _finish(pre: _Prefix, cfg: Config, wlm: WordLM | None) -> float:
    """The LM terms for ending the utterance after ``pre``: the last word and end of sentence."""
    lm = 0.0
    if pre.clogp is not None:
        lm += cfg.alpha * pre.clogp[vocab.EOS_IDX]
    word = pre.text[pre.word_start :]
    if word:
        lm += cfg.gamma
    if wlm is not None:
        state = pre.wstate
        if word:
            logp, state = wlm.score(state, word)
            lm += cfg.beta * logp
        lm += cfg.beta * wlm.end(state)
    return lm


def beam_search(
    logp: np.ndarray,
    cfg: Config,
    cm: CharLM | None = None,
    wlm: WordLM | None = None,
) -> list[tuple[str, float]]:
    """Decode one utterance's AM output ``[T, V]`` (natural-log, per frame).

    Returns the final beam as (transcript, score), best first, each transcript once (the
    N-best list). Leaving out ``cm`` or ``wlm`` drops its terms (``alpha``, or ``beta``
    and the OOV penalties).
    """
    root = _Prefix("", 0.0, 0, wlm.start() if wlm is not None else None)
    if cm is not None:
        root.cstate, clogp = cm.start()
        root.clogp = np.asarray(clogp, dtype=np.float64).tolist()
    # prefix text -> [log P(alignments ending in blank), ... ending in a character, _Prefix]
    beam: dict[str, list] = {"": [0.0, NEG_INF, root]}

    for row in np.asarray(logp, dtype=np.float64).tolist():
        blank = row[vocab.BLANK_IDX]
        chars = [c for c in _EXTEND if row[c] >= cfg.prune]
        nxt: dict[str, list] = {}
        for text, (p_b, p_nb, pre) in beam.items():
            p = _logadd(p_b, p_nb)
            # Blank, or the last character again: the prefix stays as it is.
            entry = nxt.get(text)
            if entry is None:
                entry = nxt[text] = [NEG_INF, NEG_INF, pre]
            entry[0] = _logadd(entry[0], p + blank)
            if text:
                entry[1] = _logadd(entry[1], p_nb + row[_INDEX[text[-1]]])
            # A new character. Repeating the last one needs a blank in between.
            for c in chars:
                ch = vocab.CHARS[c]
                if ch == _SPACE and (not text or text[-1] == _SPACE):
                    continue
                new = text + ch
                entry = nxt.get(new)
                if entry is None:
                    old = beam.get(new)
                    pre_new = old[2] if old is not None else _extend(pre, c, cfg, wlm)
                    entry = nxt[new] = [NEG_INF, NEG_INF, pre_new]
                from_p = p_b if text and text[-1] == ch else p
                entry[1] = _logadd(entry[1], from_p + row[c])

        kept = heapq.nlargest(
            cfg.beam, nxt.items(), key=lambda kv: _logadd(kv[1][0], kv[1][1]) + kv[1][2].lm
        )
        beam = dict(kept)
        if cm is not None:
            _advance_cm(cm, [entry[2] for entry in beam.values()])

    # A prefix with a trailing space ends as the same transcript as the one without: keep
    # the better of the two.
    final: dict[str, float] = {}
    for text, (p_b, p_nb, pre) in beam.items():
        score = _logadd(p_b, p_nb) + pre.lm + _finish(pre, cfg, wlm)
        text = text.rstrip(_SPACE)
        if score > final.get(text, NEG_INF):
            final[text] = score
    return sorted(final.items(), key=lambda ts: ts[1], reverse=True)
