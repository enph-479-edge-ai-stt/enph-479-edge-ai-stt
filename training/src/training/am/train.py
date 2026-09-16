"""Prototype CTC training loop for the acoustic model.

Smoke-test scope: prove the loss drops past the blank wall, checkpoint/resume
survives a killed Colab session, and greedy dev CER is computable. Not tuned for
WER. Checkpointing is atomic (write a temp file, then ``os.replace``) and resume
is the default path, per the Colab discipline rules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import jiwer
import torch
from torch import nn
from torch.utils.data import DataLoader

from training import vocab
from training.am.dataset import Batch, LibriSpeechFeatures, Utterance, collate
from training.am.model import AcousticModel
from training.features.mfcc import FEATURE_DIM

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    """Hyperparameters and paths for a prototype training run."""

    n_feats: int = FEATURE_DIM
    n_hidden: int = 256
    n_layers: int = 3
    n_out: int = vocab.VOCAB_SIZE
    dropout: float = 0.2
    lr: float = 3e-4
    grad_clip: float = 5.0
    batch_size: int = 8
    epochs: int = 20
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 200  # steps between mid-epoch checkpoints
    log_every: int = 20  # steps between loss prints
    device: str = _DEFAULT_DEVICE


def _build_model(cfg: TrainConfig, device: torch.device) -> AcousticModel:
    return AcousticModel(cfg.n_feats, cfg.n_hidden, cfg.n_layers, cfg.n_out, cfg.dropout).to(device)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_cer: float,
) -> None:
    """Write a checkpoint atomically: torch.save to a temp file, then replace.

    Replacing the target rather than writing in place means a disconnect
    mid-write can never corrupt the latest checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_cer": best_cer,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    opt: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> tuple[int, int, float]:
    """Load model (and optional optimizer) state; return (epoch, step, best_cer)."""
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model"])
    if opt is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    return ck["epoch"], ck["step"], ck.get("best_cer", float("inf"))


def _ctc_step(
    model: AcousticModel,
    ctc: nn.CTCLoss,
    batch: Batch,
    device: torch.device,
) -> torch.Tensor:
    feats, feat_len, labels, label_len = batch
    logp = model(feats.to(device), feat_len)  # [B, T, N]
    # CTCLoss wants [T, B, N]; lengths stay on CPU.
    return ctc(logp.transpose(0, 1), labels.to(device), feat_len, label_len)


@torch.no_grad()
def greedy_cer(model: AcousticModel, loader: DataLoader, device: torch.device) -> float:
    """Argmax -> CTC collapse -> jiwer CER over a loader. Verification gate V1."""
    model.eval()
    refs: list[str] = []
    hyps: list[str] = []
    for feats, feat_len, labels, label_len in loader:
        pred = model(feats.to(device), feat_len).argmax(-1).cpu()  # [B, T]
        offset = 0
        for b in range(pred.size(0)):
            length = int(label_len[b])
            frames = int(feat_len[b])
            refs.append(vocab.decode(labels[offset : offset + length]) or " ")
            hyps.append(vocab.collapse(pred[b, :frames].tolist()) or " ")
            offset += length
    model.train()
    return float(jiwer.cer(refs, hyps))


def train(
    cfg: TrainConfig,
    train_items: list[Utterance],
    dev_items: list[Utterance],
    mean: torch.Tensor,
    std: torch.Tensor,
    resume: bool = True,
) -> tuple[AcousticModel, float]:
    """Run the smoke-test training loop. Returns (model, best dev CER)."""
    device = torch.device(cfg.device)
    train_loader = DataLoader(
        LibriSpeechFeatures(train_items, mean, std),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        LibriSpeechFeatures(dev_items, mean, std),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = _build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ctc = nn.CTCLoss(blank=vocab.BLANK_IDX, zero_infinity=True)

    ckpt = Path(cfg.ckpt_dir) / "latest.pt"
    start_epoch, step, best_cer = 0, 0, float("inf")
    if resume and ckpt.exists():
        start_epoch, step, best_cer = load_checkpoint(ckpt, model, opt, cfg.device)
        print(f"[resume] epoch {start_epoch}, step {step}, best CER {best_cer:.3f}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        for batch in train_loader:
            loss = _ctc_step(model, ctc, batch, device)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)  # LSTMs explode
            opt.step()
            step += 1
            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.3f}")
            if step % cfg.ckpt_every == 0:
                save_checkpoint(ckpt, model, opt, epoch, step, best_cer)
        cer = greedy_cer(model, dev_loader, device)
        best_cer = min(best_cer, cer)
        save_checkpoint(ckpt, model, opt, epoch + 1, step, best_cer)
        print(f"[epoch {epoch}] dev CER {cer:.3f} ({time.perf_counter() - t0:.0f}s)")
    return model, best_cer


def overfit_one_batch(
    cfg: TrainConfig,
    items: list[Utterance],
    n_utts: int = 2,
    steps: int = 100,
) -> list[float]:
    """Overfit a tiny batch to near-zero CTC loss as a wiring sanity check.

    Independent of data volume. Returns the loss at each step.
    """
    device = torch.device(cfg.device)
    ds = LibriSpeechFeatures(items[:n_utts])
    batch = collate([ds[i] for i in range(len(ds))])
    model = _build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctc = nn.CTCLoss(blank=vocab.BLANK_IDX, zero_infinity=True)
    losses: list[float] = []
    for _ in range(steps):
        loss = _ctc_step(model, ctc, batch, device)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        losses.append(loss.item())
    return losses
