"""123-dim acoustic features: 40 log-mel + energy + delta + double-delta, + CMVN.

PROVISIONAL feature spec for the smoke test. Every constant here (window, FFT
size, mel range, the energy definition, the delta window, the CMVN recipe) is
what ``shared/specs`` must freeze and the runtime numpy port must reproduce
bit-for-bit. See ``feature-extraction-implementation.md`` in the notes workspace.

The chain, per frame:
    400 samples -> Hamming window -> FFT(512) -> |.|^2 -> 40 mel filters -> log
    (+ one log frame-energy = 41 static) -> delta, double-delta = 123 -> CMVN

``FEATURE_DIM`` is *derived* from ``N_MELS``; it is never a free parameter. The
model's ``input_size`` and the RTL's weight count must equal it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
import torchaudio

SAMPLE_RATE = 16_000
N_FFT = 512
WIN_LENGTH = 400  # 25 ms at 16 kHz
HOP_LENGTH = 160  # 10 ms -> 100 frames/s
N_MELS = 40
LOG_EPS = 1e-10
DELTA_WIN = 5  # +/-2 frames

FEATURE_DIM = (N_MELS + 1) * 3  # 123, single source of truth

MelSpec = torchaudio.transforms.MelSpectrogram
Deltas = torchaudio.transforms.ComputeDeltas


def build_transforms() -> tuple[MelSpec, Deltas]:
    """Build the stateless transforms reused for every utterance.

    Cheap to hold once per worker rather than rebuild per call.
    """
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
        center=False,  # no center padding, closer to a streaming/Kaldi frame layout
        window_fn=torch.hamming_window,
    )
    deltas = torchaudio.transforms.ComputeDeltas(win_length=DELTA_WIN)
    return mel, deltas


def extract(waveform: torch.Tensor, mel: MelSpec, deltas: Deltas) -> torch.Tensor:
    """Waveform ``[C, N]`` (16 kHz) -> features ``[T, 123]``, before CMVN.

    Energy is the log of the summed mel-band power (the one-line "bin sum"
    variant); the exact energy definition is a spec choice still to be frozen.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(0, keepdim=True)  # defensive downmix to mono
    power_mel = mel(waveform)  # [1, N_MELS, T]
    logmel = torch.log(power_mel + LOG_EPS).squeeze(0)  # [N_MELS, T]
    energy = torch.log(power_mel.sum(dim=1) + LOG_EPS)  # [1, T]
    static = torch.cat([logmel, energy], dim=0)  # [41, T]
    d1 = deltas(static)
    d2 = deltas(d1)
    feats = torch.cat([static, d1, d2], dim=0)  # [123, T]
    return feats.transpose(0, 1).contiguous()  # [T, 123]


def compute_cmvn(feature_frames: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Global CMVN stats over an iterable of ``[T, 123]`` tensors.

    Feed **training-set utterances only**; the returned mean/std are reused
    unchanged for dev, test and the board. Accumulates in float64 for stability.
    """
    total = 0
    s = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    ss = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    for feats in feature_frames:
        f = feats.to(torch.float64)
        total += f.size(0)
        s += f.sum(0)
        ss += (f * f).sum(0)
    if total == 0:
        raise ValueError("no frames to compute CMVN over")
    mean = s / total
    var = (ss / total) - mean * mean
    std = var.clamp_min(1e-8).sqrt()
    return mean.to(torch.float32), std.to(torch.float32)


def apply_cmvn(feats: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Element-wise (feats - mean) / std."""
    return (feats - mean) / std


def _iter_features(waveforms: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
    """Extract features for a sequence of raw waveforms with shared transforms.

    Used by callers that want CMVN over in-memory audio.
    """
    mel, deltas = build_transforms()
    for wav in waveforms:
        yield extract(wav, mel, deltas)
