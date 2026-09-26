"""CTC training loop for the acoustic model.

Each epoch ends with greedy dev CER and a checkpoint: ``latest.pt`` every epoch,
``best.pt`` when dev CER improves, both in ``cfg.run_dir``. Point ``run_dir`` at
Drive; calling ``train`` again with the same run_dir resumes from ``latest.pt``.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import jiwer
import torch
from torch import nn
from torch.utils.data import DataLoader

from training import vocab
from training.am.dataset import LibriSpeechFeatures, Utterance, collate
from training.am.model import AcousticModel


@dataclass
class TrainConfig:
    """Hyperparameters and the run directory for one training run."""

    n_hidden: int = 256
    n_layers: int = 3
    dropout: float = 0.2
    lr: float = 3e-4
    grad_clip: float = 5.0
    batch_size: int = 32
    epochs: int = 20
    num_workers: int = 2
    log_every: int = 50  # batches between progress lines
    run_dir: str = "runs/am"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def save_checkpoint(path: Path, state: dict) -> None:
    """``torch.save`` to a temp file, then rename, so a disconnect can't corrupt ``path``."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


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
    train_items: list[Utterance],
    dev_items: list[Utterance],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> float:
    """Train (or resume) the run in ``cfg.run_dir``. Returns the best dev CER."""
    device = torch.device(cfg.device)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    loader_kw = {
        "batch_size": cfg.batch_size,
        "collate_fn": collate,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        LibriSpeechFeatures(train_items, mean, std), shuffle=True, **loader_kw
    )
    dev_loader = DataLoader(LibriSpeechFeatures(dev_items, mean, std), **loader_kw)

    model = AcousticModel(n_hidden=cfg.n_hidden, n_layers=cfg.n_layers, dropout=cfg.dropout)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ctc = nn.CTCLoss(blank=vocab.BLANK_IDX, zero_infinity=True)

    latest = run_dir / "latest.pt"
    start_epoch, best_cer = 0, float("inf")
    if latest.exists():
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best_cer = ck["epoch"], ck["best_cer"]
        print(f"resuming {latest} at epoch {start_epoch} (best dev CER {best_cer:.4f})")

    n_batches = len(train_loader)
    for epoch in range(start_epoch, cfg.epochs):
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
                print(
                    f"epoch {epoch} batch {b}/{n_batches} loss {loss_sum / b:.3f} "
                    f"({s_per_batch:.2f} s/batch)"
                )
        train_s = time.perf_counter() - t0

        cer = greedy_cer(model, dev_loader, device)
        improved = cer < best_cer
        best_cer = min(best_cer, cer)
        state = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch + 1,
            "best_cer": best_cer,
            "config": dataclasses.asdict(cfg),
            "cmvn_mean": mean,
            "cmvn_std": std,
        }
        if improved:
            save_checkpoint(run_dir / "best.pt", state)
        save_checkpoint(latest, state)
        print(
            f"[epoch {epoch}] train loss {loss_sum / n_batches:.3f} | dev CER {cer:.4f}"
            f"{' (best)' if improved else ''} | train {train_s:.0f}s, "
            f"eval {time.perf_counter() - t0 - train_s:.0f}s"
        )
    return best_cer
