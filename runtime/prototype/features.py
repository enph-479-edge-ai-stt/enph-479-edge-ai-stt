"""The board's feature extraction: numpy port of ``training/src/training/features/fbank.py``.

16 kHz waveform -> 123-dim log-mel features -> CMVN -> int8 at step 2**-5, the AM's input
as the fabric takes it. Matches torchaudio's ``MelSpectrogram`` with the training
constants: ``torch.stft`` centres the 400-sample periodic Hamming window in each 512-sample
frame (``center=False``), HTK mel scale from 0 Hz to 8 kHz with no filter normalization,
and ``ComputeDeltas`` (replicate padding, kernel -2..2 / 10).
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 40
LOG_EPS = 1e-10
DELTA_N = 2  # ComputeDeltas(win_length=5)
X_FRAC = 5  # AcousticModel.x_frac


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filters() -> np.ndarray:
    """Torchaudio's ``melscale_fbanks`` for these constants: ``[n_fft // 2 + 1, n_mels]``."""
    freqs = np.linspace(0, SAMPLE_RATE // 2, N_FFT // 2 + 1)
    m = np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(SAMPLE_RATE / 2)), N_MELS + 2)
    f_pts = _mel_to_hz(m)
    f_diff = np.diff(f_pts)
    slopes = f_pts[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up))


def _window() -> np.ndarray:
    """``torch.hamming_window(400)`` (periodic), zero-padded to 512 the way ``torch.stft`` does."""
    n = np.arange(WIN_LENGTH)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * n / WIN_LENGTH)
    left = (N_FFT - WIN_LENGTH) // 2
    return np.pad(w, (left, N_FFT - WIN_LENGTH - left))


def _deltas(x: np.ndarray) -> np.ndarray:
    """``ComputeDeltas(win_length=5)`` along time for ``x`` ``[T, D]``."""
    pad = np.pad(x, ((DELTA_N, DELTA_N), (0, 0)), mode="edge")
    t = x.shape[0]
    out = sum(k * pad[DELTA_N + k : DELTA_N + k + t] for k in range(-DELTA_N, DELTA_N + 1))
    return out / (DELTA_N * (DELTA_N + 1) * (2 * DELTA_N + 1) / 3)


class Fbank:
    """Feature extraction with the filterbank and window built once."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Store the AM's CMVN statistics and build the window and mel filters."""
        self.mean = mean.astype(np.float64)
        self.std = std.astype(np.float64)
        self.window = _window()
        self.filters = mel_filters()

    def extract(self, wav: np.ndarray) -> np.ndarray:
        """Mono 16 kHz waveform ``[N]`` -> features ``[T, 123]``, before CMVN."""
        n_frames = 1 + (len(wav) - N_FFT) // HOP_LENGTH
        idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
        frames = wav.astype(np.float64)[idx] * self.window
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # [T, 257]
        mel = power @ self.filters  # [T, 40]
        static = np.concatenate(
            [np.log(mel + LOG_EPS), np.log(mel.sum(axis=1, keepdims=True) + LOG_EPS)], axis=1
        )
        d1 = _deltas(static)
        return np.concatenate([static, d1, _deltas(d1)], axis=1)

    def int8(self, feats: np.ndarray) -> np.ndarray:
        """CMVN, then int8 at step ``2**-X_FRAC``: what the runtime sends the fabric."""
        x = (feats - self.mean) / self.std * 2.0**X_FRAC
        return np.clip(np.round(x), -128, 127).astype(np.int8)
