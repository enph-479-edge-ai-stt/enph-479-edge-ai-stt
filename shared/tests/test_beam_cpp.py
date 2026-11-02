"""The C++ search (``shared.decode.beam_cpp``) against the golden case and the Python search.

The C++ port keeps the Python search's order of floating-point operations, so on one
machine the two agree exactly. Skips unless the module is built (``cpp/build.sh``); CI
builds it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

beam_cpp = pytest.importorskip("shared.decode.beam_cpp")

from shared.decode.beam import Config, beam_search  # noqa: E402
from shared.decode.golden import GOLDEN_DIR, TableCharLM  # noqa: E402

CASE = json.loads((GOLDEN_DIR / "case.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASE["cases"], ids=lambda c: c["name"])
def test_golden_case(case):
    cm = TableCharLM(CASE["cm_table"]) if case["cm"] else None
    wlm = beam_cpp.WordLM(str(GOLDEN_DIR / "words.arpa")) if case["wlm"] else None
    got = beam_cpp.beam_search(np.array(CASE["am_logp"]), Config(**case["config"]), cm, wlm)
    assert [text for text, _ in got] == [text for text, _ in case["beam"]]
    for (_, score), (_, want) in zip(got, case["beam"], strict=True):
        assert score == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("beam", [1, 4, 32])
def test_same_as_python_on_noise(beam):
    """Noisy AM output with many characters in reach, without and with the bigram CM."""
    rng = np.random.default_rng(beam)
    logits = rng.normal(0.0, 2.0, (40, len(CASE["chars"])))
    am = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    cfg = Config(beam=beam, alpha=0.5, gamma=1.0, prune=math.log(1e-3))
    for cm in (None, TableCharLM(CASE["cm_table"])):
        assert beam_cpp.beam_search(am, cfg, cm) == beam_search(am, cfg, cm)
