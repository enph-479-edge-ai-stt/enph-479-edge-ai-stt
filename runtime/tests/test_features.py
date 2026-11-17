"""The numpy features against training's, through the golden case in shared/golden/features/."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from runtime.features import X_FRAC, Fbank

GOLDEN = Path(__file__).resolve().parents[2] / "shared" / "golden" / "features"
# Training computes in float32 and this port in float64, so a value this close to a
# rounding tie can land on either side of it.
TIE = 1e-3


@pytest.fixture(scope="module")
def case():
    return np.load(GOLDEN / "case.npz")


@pytest.fixture(scope="module")
def fbank(case):
    return Fbank(case["cmvn_mean"], case["cmvn_std"])


@pytest.fixture(scope="module")
def feats(fbank):
    wav, sr = sf.read(str(GOLDEN / "6930-75918-0000.flac"), dtype="float32")
    assert sr == 16_000
    return fbank.extract(wav)


def test_x_frac_is_the_cases(case):
    assert int(case["x_frac"]) == X_FRAC


def test_features_match_training(case, feats):
    np.testing.assert_allclose(feats, case["feats"], rtol=0, atol=1e-4)


def test_int8_matches_training_off_ties(case, fbank, feats):
    q = fbank.int8(feats).astype(int)
    want = case["int8"].astype(int)
    x = (feats - fbank.mean) / fbank.std * 2.0**X_FRAC
    near_tie = np.abs(np.abs(x) % 1 - 0.5) < TIE
    off = q != want
    assert np.all(np.abs(q - want) <= 1)
    assert not np.any(off & ~near_tie), f"{np.count_nonzero(off & ~near_tie)} values off a tie"
    assert np.count_nonzero(off) <= 0.001 * q.size
