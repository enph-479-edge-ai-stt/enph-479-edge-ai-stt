"""Write the decoder's golden case, ``shared/golden/decode/case.json``.

Training's beam search is the reference; ``runtime/`` must reproduce its output from
the same inputs. The inputs are synthetic: AM log-probs for a noisy "THE CAT SAT ON THE
MAT", a bigram table standing in for the CM (``TableCharLM``) and the tiny trigram
``words.arpa`` next to the case as the word LM. The case decodes them three ways (no
LM, word LM only, both) and stores each final beam.

Run ``python -m training.decode.golden`` after any change to the search, then check that
``runtime/`` still passes against the new case.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from training import vocab
from training.decode.beam import Config, beam_search

GOLDEN_DIR = Path(__file__).resolve().parents[4] / "shared" / "golden" / "decode"
SENTENCE = "THE CAT SAT ON THE MAT"
CASES = {
    "am": (Config(beam=8), False, False),
    "wlm": (Config(beam=16, beta=1.0, gamma=1.5), False, True),
    "cm_wlm": (Config(beam=16, alpha=0.5, beta=1.0, gamma=1.5), True, True),
}


class TableCharLM:
    """A bigram CM: the state is the last character, its log-probs are that row of ``table``."""

    def __init__(self, table: list[list[float]]) -> None:
        """``table[prev][next]`` is the log-prob of ``next`` after ``prev``."""
        self.table = table

    def start(self) -> tuple[int, list[float]]:
        """After EOS."""
        return vocab.EOS_IDX, self.table[vocab.EOS_IDX]

    def step(self, states: list[int], chars: list[int]) -> tuple[list[int], list[list[float]]]:
        """The new state is just the character."""
        return chars, [self.table[c] for c in chars]


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def make_inputs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """AM log-probs ``[T, V]`` and a bigram log-prob table ``[V, V]``, rounded to 6 places.

    Each character gets two frames and a blank frame after it. The AM hesitates between
    C and H at the start of CAT, and noise puts other characters within reach.
    """
    rng = np.random.default_rng(seed)
    v = vocab.VOCAB_SIZE
    frames = []
    for ch in SENTENCE:
        target = np.zeros(v)
        target[vocab.encode(ch)[0]] = 5.0
        if ch == "C":
            target[vocab.encode("H")[0]] = 4.5
        frames += [target, target, np.eye(v)[vocab.BLANK_IDX] * 4.0]
    logits = np.stack(frames) + rng.normal(0.0, 1.0, (len(frames), v))
    logits[:, vocab.EOS_IDX] = -10.0  # the AM never emits EOS
    am = _log_softmax(logits)

    bigram = rng.normal(0.0, 1.0, (v, v))
    text = [vocab.EOS_IDX, *vocab.encode(SENTENCE), vocab.EOS_IDX]
    for prev, nxt in zip(text, text[1:], strict=False):
        bigram[prev, nxt] += 2.0
    bigram[:, vocab.BLANK_IDX] = -10.0  # the blank never occurs in text
    cm = _log_softmax(bigram)
    return am.round(6), cm.round(6)


def main(out_dir: Path = GOLDEN_DIR) -> None:
    """Decode the inputs every way in ``CASES`` and write the case file."""
    from training.wlm.model import KenLM  # here, so the tests can import this without kenlm

    am, table = make_inputs()
    cm = TableCharLM(table.tolist())
    wlm = KenLM(out_dir / "words.arpa")
    cases = []
    for name, (cfg, use_cm, use_wlm) in CASES.items():
        beam = beam_search(am, cfg, cm if use_cm else None, wlm if use_wlm else None)
        cases.append(
            {"name": name, "config": asdict(cfg), "cm": use_cm, "wlm": use_wlm, "beam": beam}
        )
        print(f"{name}: {beam[0][0]!r} ({beam[0][1]:.3f})")
    case = {
        "chars": list(vocab.CHARS),
        "am_logp": am.tolist(),
        "cm_table": table.tolist(),
        "cases": cases,
    }
    assert all(math.isfinite(s) for c in cases for _, s in c["beam"])
    (out_dir / "case.json").write_text(json.dumps(case) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'case.json'}")


if __name__ == "__main__":
    main()
