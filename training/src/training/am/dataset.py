"""LibriSpeech utterances -> (features, labels) for CTC training.

Features are extracted from the FLAC on the fly, inside the DataLoader workers
(no feature cache). ``collate`` packs a batch into what ``nn.CTCLoss`` wants:
padded features, concatenated labels, and both length vectors.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from training import vocab
from training.features import fbank

Utterance = tuple[Path, str]
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


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
    ) -> None:
        """Store the utterances and CMVN stats, and build the feature transforms."""
        self.items = items
        self.mean = mean
        self.std = std
        self._mel, self._deltas = fbank.build_transforms()

    def __len__(self) -> int:
        """Return the number of utterances."""
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load utterance ``i``'s FLAC (16 kHz mono) and return (features, labels)."""
        flac, text = self.items[i]
        wav, _ = sf.read(str(flac), dtype="float32")
        feats = fbank.extract(torch.from_numpy(wav).unsqueeze(0), self._mel, self._deltas)
        if self.mean is not None:
            feats = fbank.apply_cmvn(feats, self.mean, self.std)
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
