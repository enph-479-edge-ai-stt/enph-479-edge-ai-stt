"""123-dim log-mel filterbank features, plus global CMVN.

Not MFCCs: there is no DCT. Per 10 ms frame:
    400-sample Hamming window (zero-padded to 512) -> |FFT|^2 -> 40 mel filters
    -> log, plus one log-energy = 41 static -> delta, double-delta = 123

PROVISIONAL until ``shared/specs`` is frozen. The board's numpy port has to
reproduce these constants exactly.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torchaudio

SAMPLE_RATE = 16_000
N_FFT = 512
WIN_LENGTH = 400  # 25 ms
HOP_LENGTH = 160  # 10 ms -> 100 frames/s
N_MELS = 40
LOG_EPS = 1e-10
DELTA_WIN = 5  # +/-2 frames

FEATURE_DIM = (N_MELS + 1) * 3  # 123; the model's input size is derived from this

MelSpec = torchaudio.transforms.MelSpectrogram
Deltas = torchaudio.transforms.ComputeDeltas


def build_transforms() -> tuple[MelSpec, Deltas]:
    """Build the mel filterbank and delta transforms (once, reused per utterance)."""
    mel = MelSpec(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
        center=False,
        window_fn=torch.hamming_window,
    )
    return mel, Deltas(win_length=DELTA_WIN)


def extract(waveform: torch.Tensor, mel: MelSpec, deltas: Deltas) -> torch.Tensor:
    """Mono 16 kHz waveform ``[1, N]`` -> features ``[T, 123]``, before CMVN."""
    power_mel = mel(waveform)  # [1, 40, T]
    logmel = torch.log(power_mel + LOG_EPS).squeeze(0)  # [40, T]
    energy = torch.log(power_mel.sum(dim=1) + LOG_EPS)  # [1, T], log of summed mel power
    static = torch.cat([logmel, energy])  # [41, T]
    d1 = deltas(static)
    return torch.cat([static, d1, deltas(d1)]).T.contiguous()  # [T, 123]


def compute_cmvn(feats: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Global per-dimension mean and std over ``[T, 123]`` tensors (training set only).

    Accumulates in float64 so the sums stay exact over millions of frames.
    """
    total = 0
    s = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    ss = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    for f in feats:
        f = f.to(torch.float64)
        total += f.size(0)
        s += f.sum(0)
        ss += (f * f).sum(0)
    mean = s / total
    std = (ss / total - mean * mean).clamp_min(1e-8).sqrt()
    return mean.float(), std.float()


def apply_cmvn(feats: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Element-wise ``(feats - mean) / std``."""
    return (feats - mean) / std
