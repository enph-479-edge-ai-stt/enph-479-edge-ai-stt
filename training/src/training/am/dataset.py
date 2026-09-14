"""LibriSpeech -> (features, label) dataset for CTC training.

Extracts 123-dim features from FLAC on the fly. This is the self-contained
smoke-test path; the production run reads cached feature shards instead of
decoding audio every epoch. ``collate`` packs a batch into exactly what
``nn.CTCLoss`` wants: padded features + concatenated labels + both length
vectors. No alignment and no per-frame labels: that is CTC's job, not the
dataset's.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from training import vocab
from training.features import mfcc

Utterance = tuple[Path, str]
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def list_utterances(subset_dir: str | Path) -> list[Utterance]:
    """(flac_path, normalized_transcript) for every utterance under a subset dir,
    skipping any transcript line whose FLAC is missing."""
    subset_dir = Path(subset_dir)
    items: list[Utterance] = []
    for trans in sorted(subset_dir.glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, _, text = line.partition(" ")
            flac = trans.parent / f"{utt_id}.flac"
            if flac.is_file():
                items.append((flac, vocab.normalize(text)))
    return items


class LibriSpeechFeatures(Dataset):
    """One item = (features ``[T, 123]``, label indices ``[L]``). CMVN mean/std,
    when given, are applied to every item (pass the train-set stats)."""

    def __init__(
        self,
        items: list[Utterance],
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        limit: int | None = None,
    ) -> None:
        self.items = items[:limit] if limit else items
        self.mean = mean
        self.std = std
        self._mel, self._deltas = mfcc.build_transforms()

    def __len__(self) -> int:
        return len(self.items)

    def features_for(self, flac: str | Path) -> torch.Tensor:
        # Load with soundfile (bundled libsndfile), not torchaudio.load: recent
        # torchaudio routes I/O through TorchCodec, an extra native dependency we
        # do not want on the board or in CI. torchaudio stays for the transforms.
        data, sr = sf.read(str(flac), dtype="float32", always_2d=True)  # [N, C]
        wav = torch.from_numpy(data.T.copy())  # [C, N]
        if sr != mfcc.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, mfcc.SAMPLE_RATE)
        feats = mfcc.extract(wav, self._mel, self._deltas)
        if self.mean is not None and self.std is not None:
            feats = mfcc.apply_cmvn(feats, self.mean, self.std)
        return feats

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        flac, text = self.items[i]
        feats = self.features_for(flac)
        labels = torch.tensor(vocab.encode(text), dtype=torch.long)
        return feats, labels


def compute_cmvn_over(
    items: list[Utterance], limit: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global CMVN over the given utterances (pass TRAIN items only)."""
    ds = LibriSpeechFeatures(items, limit=limit)
    return mfcc.compute_cmvn(ds.features_for(flac) for flac, _ in ds.items)


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> Batch:
    """[(feats ``[T,123]``, labels ``[L]``), ...] -> (feats_padded ``[B,Tmax,123]``,
    feat_lengths ``[B]``, labels_cat ``[sum L]``, label_lengths ``[B]``)."""
    feats, labels = zip(*batch, strict=True)
    feat_lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    label_lengths = torch.tensor([lab.size(0) for lab in labels], dtype=torch.long)
    feats_padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    labels_cat = torch.cat(labels)
    return feats_padded, feat_lengths, labels_cat, label_lengths
