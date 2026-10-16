"""Beam search tests: CTC prefix merging, the LM terms, and the golden case runtime/ shares.

The search is plain Python, so these run on CI. The golden case needs KenLM (the ``wlm``
extra) and the torch char-LM test needs torch; each skips without it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from training import vocab
from training.decode.beam import Config, beam_search
from training.decode.golden import GOLDEN_DIR, TableCharLM

NO_PRUNE = Config(beam=16, prune=-math.inf)
# Only the spelled symbols are tried (the character pruning doesn't depend on the LMs)
# and the beam never fills, so the CTC sums don't change with the LM terms.
EXACT = Config(beam=100_000, prune=math.log(0.01))


def _am(*frames: dict[str, float]) -> np.ndarray:
    """AM log-probs, one frame per dict of {symbol: probability}; "_" is the blank.

    Whatever probability a frame leaves over is spread over the other symbols (at least
    1e-9 each, so every log-prob is finite).
    """
    out = []
    for frame in frames:
        rest = max((1.0 - sum(frame.values())) / (vocab.VOCAB_SIZE - len(frame)), 1e-9)
        p = np.full(vocab.VOCAB_SIZE, rest)
        for sym, prob in frame.items():
            p[vocab.BLANK_IDX if sym == "_" else vocab.encode(sym)[0]] = prob
        out.append(np.log(p))
    return np.stack(out)


def _spell(path: str, p: float = 0.97) -> np.ndarray:
    """One confident frame per symbol of ``path`` ("_" for the blank)."""
    return _am(*({sym: p} for sym in path))


class UnigramLM:
    """A word LM without context: log-prob per word, OOV words at -5 plus the penalty."""

    oov_penalty = -10.0

    def __init__(self, logp: dict[str, float]) -> None:
        self.logp = logp

    def start(self):
        return None

    def score(self, state, word):
        return self.logp.get(word, -5.0 + self.oov_penalty), None

    def end(self, state):
        return 0.0

    def is_prefix(self, text):
        return any(w.startswith(text) for w in self.logp)


def test_confident_path_decodes_to_its_collapse():
    am = _spell("HH_E_LL_LL_O_ _WW_O_R_L_DD")
    best = beam_search(am, NO_PRUNE)[0][0]
    assert best == "HELLO WORLD" == vocab.collapse(am.argmax(-1))


def test_repeat_needs_a_blank():
    assert beam_search(_spell("LL"), NO_PRUNE)[0][0] == "L"
    assert beam_search(_spell("L_L"), NO_PRUNE)[0][0] == "LL"


def test_sums_alignments_where_greedy_does_not():
    # Greedy takes blank, blank -> "". But "A" has three alignments (A_, _A, AA) worth
    # .4*.6 + .6*.4 + .4*.4 = .64, against "" at .6*.6 = .36.
    am = _am({"A": 0.4, "_": 0.6}, {"A": 0.4, "_": 0.6})
    assert vocab.collapse(am.argmax(-1)) == ""
    nbest = dict(beam_search(am, NO_PRUNE))
    assert nbest["A"] == pytest.approx(math.log(0.4 * 0.6 + 0.6 * 0.4 + 0.4 * 0.4), abs=1e-3)
    assert next(iter(nbest)) == "A"


def test_no_leading_or_double_space():
    am = _spell(" _A_  _B_ ", p=0.9)
    for text, _ in beam_search(am, NO_PRUNE):
        assert not text.startswith(" ")
        assert "  " not in text


def test_nbest_is_sorted_and_unique():
    am = _am(*[{"A": 0.3, "B": 0.3, " ": 0.2, "_": 0.2}] * 5)
    nbest = beam_search(am, Config(beam=8, prune=-math.inf))
    scores = [s for _, s in nbest]
    assert scores == sorted(scores, reverse=True)
    assert len({t for t, _ in nbest}) == len(nbest)
    assert len(nbest) <= 8


def test_prune_skips_unlikely_characters():
    am = _am({"A": 0.9, "B": 2e-4, "_": 0.0998}, {"_": 1.0})
    texts = {t for t, _ in beam_search(am, Config(beam=16, prune=math.log(1e-3)))}
    assert "B" not in texts
    assert "B" in {t for t, _ in beam_search(am, NO_PRUNE)}


def _cat_or_hat() -> np.ndarray:
    """An AM that slightly prefers HAT over CAT."""
    return _am({"H": 0.5, "C": 0.45}, {"_": 0.95}, {"A": 0.95}, {"_": 0.95}, {"T": 0.95})


def test_char_lm_tips_the_balance():
    assert beam_search(_cat_or_hat(), NO_PRUNE)[0][0] == "HAT"
    table = np.full((vocab.VOCAB_SIZE, vocab.VOCAB_SIZE), math.log(1 / vocab.VOCAB_SIZE))
    table[vocab.EOS_IDX, vocab.encode("C")[0]] = math.log(0.5)
    table[vocab.EOS_IDX, vocab.encode("H")[0]] = math.log(0.01)
    cfg = Config(beam=16, alpha=1.0, prune=-math.inf)
    assert beam_search(_cat_or_hat(), cfg, cm=TableCharLM(table.tolist()))[0][0] == "CAT"


def test_word_lm_tips_the_balance():
    wlm = UnigramLM({"CAT": math.log(0.1), "HAT": math.log(0.001)})
    cfg = Config(beam=16, beta=1.0, prune=-math.inf)
    assert beam_search(_cat_or_hat(), cfg, wlm=wlm)[0][0] == "CAT"


def test_word_lm_scores_every_word_and_the_end():
    wlm = UnigramLM({"A": -1.0, "B": -2.0})
    wlm.end = lambda state: -0.5
    am = _spell("A_ _B")
    am_only = dict(beam_search(am, EXACT))["A B"]
    cfg = Config(beam=EXACT.beam, beta=2.0, gamma=0.25, prune=EXACT.prune)
    fused = dict(beam_search(am, cfg, wlm=wlm))["A B"]
    assert fused == pytest.approx(am_only + 2.0 * (-1.0 - 2.0 - 0.5) + 2 * 0.25)


def test_oov_word_pays_per_character_past_the_vocab():
    # "AXY" leaves the vocab's prefixes at X: two characters at the penalty, then the
    # word's own OOV score when it ends.
    wlm = UnigramLM({"AB": -1.0})
    am = _spell("A_X_Y")
    am_only = dict(beam_search(am, EXACT))["AXY"]
    fused = dict(beam_search(am, EXACT, wlm=wlm))["AXY"]
    assert fused == pytest.approx(am_only + 2 * wlm.oov_penalty + (-5.0 + wlm.oov_penalty))


def _golden():
    return json.loads((GOLDEN_DIR / "case.json").read_text(encoding="utf-8"))


def test_golden_case_uses_this_vocab():
    assert tuple(_golden()["chars"]) == vocab.CHARS


def test_golden_case_reproduces():
    """The case was written by this search; a change to it must regenerate the case."""
    pytest.importorskip("kenlm")
    from training.wlm.model import KenLM

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


def test_torch_char_lm_steps_like_the_model():
    torch = pytest.importorskip("torch")
    from training.cm.model import CharModel
    from training.decode.run import TorchCharLM

    torch.manual_seed(0)
    model = CharModel(n_hidden=16, n_layers=2)
    lm = TorchCharLM(model)
    state, logp = lm.start()
    # Two prefixes, "HI" and "IT", stepped as one batch, against the model on each.
    h, i, t = (vocab.encode(c)[0] for c in "HIT")
    states, _ = lm.step([state, state], [h, i])
    _, logps = lm.step(states, [i, t])
    for k, text in enumerate(["HI", "IT"]):
        ref, _ = model(torch.tensor([[vocab.EOS_IDX, *vocab.encode(text)]]))
        np.testing.assert_allclose(logps[k], ref[0, -1].detach().numpy(), atol=1e-5)
    ref0, _ = model(torch.tensor([[vocab.EOS_IDX]]))
    np.testing.assert_allclose(logp, ref0[0, -1].detach().numpy(), atol=1e-6)


def test_notebook_helpers_end_to_end(tmp_path):
    """load_models -> am_log_probs -> tune, on the tiny synthetic subset, CPU."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from conftest import TEXTS, write_subset

    from training.am.dataset import LibriSpeechFeatures, compute_cmvn_over, list_utterances
    from training.am.model import AcousticModel
    from training.cm.model import CharModel
    from training.decode import run

    torch.manual_seed(0)
    items = list_utterances(write_subset(tmp_path / "subset"))
    mean, std = compute_cmvn_over(items)
    am_state = {"model": AcousticModel().state_dict(), "cmvn_mean": mean, "cmvn_std": std}
    torch.save(am_state, tmp_path / "am.pt")
    torch.save({"model": CharModel().state_dict()}, tmp_path / "cm.pt")

    am, cm, mean, std = run.load_models(tmp_path / "am.pt", tmp_path / "cm.pt", "cpu")
    logps = run.am_log_probs(am, items, mean, std, batch_size=3, num_workers=0)
    # Batched with padding, each utterance's output is what it gets on its own.
    for logp, (feats, _) in zip(logps, LibriSpeechFeatures(items, mean, std), strict=True):
        alone = am(feats.unsqueeze(0), torch.tensor([len(feats)]))[0].detach().numpy()
        np.testing.assert_allclose(logp, alone, atol=1e-5)

    log = tmp_path / "decode.log"
    grid = {"alpha": [0.5, 1.0], "gamma": [0.0]}
    best = run.tune(logps, TEXTS, Config(beam=4), grid, log, cm=run.TorchCharLM(cm))
    assert best.alpha in (0.5, 1.0) and best.beam == 4
    assert len(log.read_text().splitlines()) == 3  # two points and the best
    assert len(run.greedy(logps)) == len(TEXTS)
