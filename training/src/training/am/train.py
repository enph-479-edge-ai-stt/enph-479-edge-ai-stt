"""CTC training loop for the acoustic model.

Runs start to finish in one session: logs greedy test CER after every epoch (to the
notebook output and a log file) and returns the trained model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import jiwer
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from training import vocab
from training.am.dataset import collate
from training.am.model import AcousticModel


@dataclass
class TrainConfig:
    """Hyperparameters for one training run."""

    n_hidden: int = 256
    n_layers: int = 3
    dropout: float = 0.2
    lr: float = 3e-4
    grad_clip: float = 5.0
    batch_size: int = 32
    epochs: int = 20
    num_workers: int = 2
    log_every: int = 50  # batches between progress lines
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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


def _log(log_path: Path, msg: str) -> None:
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def train(
    cfg: TrainConfig, datasets: dict[str, list[Dataset]], log_path: str | Path
) -> AcousticModel:
    """Train on all of ``datasets["train"]``; after each epoch, log CER on all of ``"test"``.

    Each role is a list of datasets in any format, pooled together. A dataset only has to
    yield CMVN-normalized (features ``[T, 123]``, label indices ``[L]``) pairs. Progress and
    epoch lines are printed and written to ``log_path`` (overwritten). Returns the model.
    """
    log_path = Path(log_path)
    log_path.write_text("", encoding="utf-8")
    _log(log_path, str(cfg))
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
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ctc = nn.CTCLoss(blank=vocab.BLANK_IDX, zero_infinity=True)

    n_batches = len(train_loader)
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        loss_sum = 0.0
        for b, (feats, feat_len, labels, label_len) in enumerate(train_loader, 1):
            logp = model(feats.to(device, non_blocking=True), feat_len)  # [B, T, N]
            # CTCLoss wants [T, B, N]; the length vectors stay on the CPU.
            loss = ctc(logp.transpose(0, 1), labels.to(device), feat_len, label_len)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)  # LSTMs explode
            opt.step()
            loss_sum += loss.item()
            if b % cfg.log_every == 0:
                s_per_batch = (time.perf_counter() - t0) / b
                _log(
                    log_path,
                    f"epoch {epoch} batch {b}/{n_batches} loss {loss_sum / b:.3f} "
                    f"({s_per_batch:.2f} s/batch)",
                )
        train_s = time.perf_counter() - t0

        cer = greedy_cer(model, test_loader, device)
        _log(
            log_path,
            f"[epoch {epoch}] train loss {loss_sum / n_batches:.3f} | test CER {cer:.4f} | "
            f"train {train_s:.0f}s, eval {time.perf_counter() - t0 - train_s:.0f}s",
        )
    return model
