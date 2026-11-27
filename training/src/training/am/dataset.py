"""LibriSpeech utterances -> (features, labels) for CTC training.

Features are extracted from the FLAC on the fly, inside the DataLoader workers
(no feature cache). ``collate`` packs a batch into what ``nn.CTCLoss`` wants:
padded features, concatenated labels, and both length vectors.

With ``augment`` set (the training set only), each utterance is changed afresh every time
it's read: speed perturbation on the waveform, then SpecAugment-style masks on the
CMVN-normalized features. The weight image and the frontend don't change, so a model
trained this way drops straight into the decode and the board.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset

from shared import vocab
from training.features import fbank

Utterance = tuple[Path, str]
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# Speed perturbation (Ko et al. 2015): each read plays the utterance at one of these speeds.
SPEEDS = (0.9, 1.0, 1.1)
# SpecAugment-style masks, after CMVN (so a masked value is the training mean, 0).
FREQ_MASKS, FREQ_MASK_MAX = 2, 7  # up to 7 of the 40 mel bins each
TIME_MASKS, TIME_MASK_MAX, TIME_MASK_FRAC = 2, 40, 0.2  # up to 40 frames, <= 20% each


def speed_perturb(wav: torch.Tensor, speed: float) -> torch.Tensor:
    """``wav`` ``[1, N]`` played at ``speed`` (0.9: 11% longer and lower), still 16 kHz."""
    if speed == 1.0:
        return wav
    return AF.resample(wav, round(fbank.SAMPLE_RATE * speed), fbank.SAMPLE_RATE)


def spec_augment(feats: torch.Tensor) -> torch.Tensor:
    """A copy of ``feats`` ``[T, 123]`` with frequency and time masks zeroed.

    The 123 columns are 40 log-mel + energy, then their deltas, then the double deltas
    (``fbank.extract``). A frequency mask zeroes the same mel bins in all three blocks and
    leaves the energy columns alone. A time mask zeroes every column over a span of frames.
    """
    out = feats.clone()
    block = fbank.N_MELS + 1
    for _ in range(FREQ_MASKS):
        f = int(torch.randint(0, FREQ_MASK_MAX + 1, ()))
        k = int(torch.randint(0, fbank.N_MELS - f + 1, ()))
        for b in range(3):
            out[:, b * block + k : b * block + k + f] = 0
    n = out.size(0)
    t_max = min(TIME_MASK_MAX, int(TIME_MASK_FRAC * n))
    for _ in range(TIME_MASKS):
        t = int(torch.randint(0, t_max + 1, ()))
        s = int(torch.randint(0, n - t + 1, ()))
        out[s : s + t] = 0
    return out


def list_utterances(subset_dir: str | Path) -> list[Utterance]:
    """(flac path, transcript) for every utterance in a LibriSpeech subset dir, sorted."""
    items: list[Utterance] = []
    for trans in sorted(Path(subset_dir).glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, _, text = line.partition(" ")
            items.append((trans.parent / f"{utt_id}.flac", text))
    return items


class LibriSpeechFeatures(Dataset):
    """(features ``[T, 123]``, labels ``[L]``) per utterance, CMVN-normalized if given stats."""

    def __init__(
        self,
        items: list[Utterance],
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        augment: bool = False,
    ) -> None:
        """Store the utterances and CMVN stats, and build the feature transforms."""
        self.items = items
        self.mean = mean
        self.std = std
        self.augment = augment
        self._mel, self._deltas = fbank.build_transforms()

    def __len__(self) -> int:
        """Return the number of utterances."""
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load utterance ``i``'s FLAC (16 kHz mono) and return (features, labels)."""
        flac, text = self.items[i]
        wav, _ = sf.read(str(flac), dtype="float32")
        wav = torch.from_numpy(wav).unsqueeze(0)
        if self.augment:
            wav = speed_perturb(wav, SPEEDS[int(torch.randint(len(SPEEDS), ()))])
        feats = fbank.extract(wav, self._mel, self._deltas)
        if self.mean is not None:
            feats = fbank.apply_cmvn(feats, self.mean, self.std)
        if self.augment:
            feats = spec_augment(feats)
        return feats, torch.tensor(vocab.encode(text))


def compute_cmvn_over(
    items: list[Utterance], num_workers: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global CMVN mean/std over ``items`` (pass training utterances only)."""
    loader = DataLoader(LibriSpeechFeatures(items), batch_size=None, num_workers=num_workers)
    return fbank.compute_cmvn(feats for feats, _ in loader)


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> Batch:
    """Pad a batch: (feats ``[B, Tmax, 123]``, feat lengths, labels ``[sum L]``, label lengths)."""
    feats, labels = zip(*batch, strict=True)
    feat_len = torch.tensor([f.size(0) for f in feats])
    label_len = torch.tensor([lab.size(0) for lab in labels])
    return (
        torch.nn.utils.rnn.pad_sequence(feats, batch_first=True),
        feat_len,
        torch.cat(labels),
        label_len,
    )
