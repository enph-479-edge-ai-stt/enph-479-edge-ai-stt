"""CTC training of the acoustic model, on the shared loop in ``lstm/train.py``.

This module supplies what is the AM's own: its data loaders, the CTC loss, and greedy
test CER as the per-epoch evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import jiwer
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from training import vocab
from training.am.dataset import collate
from training.am.model import AcousticModel
from training.lstm import train as lstm


@dataclass
class TrainConfig(lstm.TrainConfig):
    """Hyperparameters for one training run."""

    dropout: float = 0.2
    num_workers: int = 2


@torch.no_grad()
def greedy_cer(model: AcousticModel, loader: DataLoader, device: torch.device) -> float:
    """Dev CER of per-frame argmax -> CTC collapse, no language model (gate V1)."""
    model.eval()
    refs: list[str] = []
    hyps: list[str] = []
    for feats, feat_len, labels, label_len in loader:
        pred = model(feats.to(device), feat_len).argmax(-1).cpu()
        hyps += [vocab.collapse(p[:n].tolist()) for p, n in zip(pred, feat_len, strict=True)]
        refs += [vocab.decode(lab.tolist()) for lab in labels.split(label_len.tolist())]
    return float(jiwer.cer(refs, hyps))


def train(
    cfg: TrainConfig,
    datasets: dict[str, list[Dataset]],
    log_path: str | Path,
    init_state: dict[str, torch.Tensor] | None = None,
) -> AcousticModel:
    """Train on all of ``datasets["train"]``; after each epoch, log CER on all of ``"test"``.

    Each role is a list of datasets in any format, pooled together. A dataset only has to
    yield CMVN-normalized (features ``[T, 123]``, label indices ``[L]``) pairs. Progress and
    epoch lines are printed and written to ``log_path`` (overwritten). Returns the model.

    ``init_state`` and ``cfg.weight_bits`` are the 6-bit fine-tune, see ``lstm.fit``.
    """
    device = torch.device(cfg.device)
    loader_kw = {
        "batch_size": cfg.batch_size,
        "collate_fn": collate,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(ConcatDataset(datasets["train"]), shuffle=True, **loader_kw)
    test_loader = DataLoader(ConcatDataset(datasets["test"]), **loader_kw)

    model = AcousticModel(n_hidden=cfg.n_hidden, n_layers=cfg.n_layers, dropout=cfg.dropout)
    ctc = nn.CTCLoss(blank=vocab.BLANK_IDX, zero_infinity=True)

    def losses() -> Iterator[torch.Tensor]:
        for feats, feat_len, labels, label_len in train_loader:
            logp = model(feats.to(device, non_blocking=True), feat_len)  # [B, T, N]
            # CTCLoss wants [T, B, N]; the length vectors stay on the CPU.
            yield ctc(logp.transpose(0, 1), labels.to(device), feat_len, label_len)

    lstm.fit(
        model,
        cfg,
        losses,
        len(train_loader),
        lambda loss: f"train loss {loss:.3f}",
        lambda: f"test CER {greedy_cer(model, test_loader, device):.4f}",
        log_path,
        init_state,
    )
    return model
