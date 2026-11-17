"""The features against their golden case, ``shared/golden/features/``.

Fails when ``fbank.py`` changes without the case being regenerated
(``python -m training.features.golden``), since the runtime's numpy port is held to the
case and not to this code. Skips without torch/torchaudio, as on CI.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from training.features import golden  # noqa: E402


@pytest.fixture(scope="module")
def case():
    return np.load(golden.GOLDEN_DIR / "case.npz")


def test_features_match_the_case(case):
    feats = golden.extract(golden.GOLDEN_DIR / golden.CLIP).numpy()
    np.testing.assert_allclose(feats, case["feats"], rtol=0, atol=1e-4)


def test_int8_matches_the_case(case):
    mean, std = torch.from_numpy(case["cmvn_mean"]), torch.from_numpy(case["cmvn_std"])
    q = golden.to_int8(torch.from_numpy(case["feats"]), mean, std, int(case["x_frac"]))
    np.testing.assert_array_equal(q.numpy(), case["int8"])
