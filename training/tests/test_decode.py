"""The decoder's torch side: the CM behind the search's interface, and the notebook's helpers.

The search itself and its golden case are tested in ``shared/``. These need torch and the
built C++ search (``run`` decodes with it), and skip without them.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared import vocab
from shared.decode.beam import Config


def test_torch_char_lm_steps_like_the_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("shared.decode.beam_cpp")
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
    pytest.importorskip("shared.decode.beam_cpp")
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
    # Batched with padding, each utterance's output is what it gets on its own, from the
    # int8 features the fabric takes.
    for logp, (feats, _) in zip(logps, LibriSpeechFeatures(items, mean, std), strict=True):
        feats = run.int8_features(feats, am.x_frac)
        alone = am(feats.unsqueeze(0), torch.tensor([len(feats)]))[0].detach().numpy()
        np.testing.assert_allclose(logp, alone, atol=1e-5)

    log = tmp_path / "decode.log"
    grid = {"alpha": [0.5, 1.0], "gamma": [0.0]}
    best = run.tune(logps, TEXTS, Config(beam=4), grid, log, cm=run.TorchCharLM(cm))
    assert best.alpha in (0.5, 1.0) and best.beam == 4
    assert len(log.read_text().splitlines()) == 3  # two points and the best
    assert len(run.greedy(logps)) == len(TEXTS)


def test_int8_features_round_and_clamp():
    torch = pytest.importorskip("torch")
    pytest.importorskip("shared.decode.beam_cpp")
    from training.decode import run

    x = torch.tensor([0.01, 0.02, -1.0, 3.99, 4.5, -5.0])
    got = run.int8_features(x, 5) * 32
    assert got.tolist() == [0.0, 1.0, -32.0, 127.0, 127.0, -128.0]
