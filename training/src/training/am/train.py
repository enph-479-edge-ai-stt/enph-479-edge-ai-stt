"""CTC training loop for the acoustic model.

Runs start to finish in one session: logs greedy test CER after every epoch (to the
notebook output and a log file) and returns the trained model. With ``weight_bits``
set it fine-tunes a trained model with its weights quantized, the paper's "retraining
based fixed-point optimization" (weights only; activations and the cell stay float).
The weights land on the FPGA's fixed-point grid, so ``am/export.py`` can pack the result.
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
from training.am.export import LSTM_FRAC, PROJ_FRAC
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
    weight_bits: int | None = None  # quantize the weights to this many bits (fine-tuning)
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


def _weight_steps(model: AcousticModel) -> dict[str, float]:
    """Per weight matrix, its step size on the FPGA (the fixed-point formats in ``export``).

    Powers of two, and the same for a layer's input and hidden weights: the fabric sums
    both products in one accumulator and rescales with a shift.
    """
    return {
        name: 2.0 ** -(PROJ_FRAC if name.startswith("proj") else LSTM_FRAC)
        for name, _ in model.named_parameters()
        if "weight" in name
    }


def _quantize(model: AcousticModel, steps: dict[str, float], qmax: int) -> dict[str, torch.Tensor]:
    """Swap each weight in ``steps`` for its quantized value; return the float originals."""
    params = dict(model.named_parameters())
    floats = {}
    with torch.no_grad():
        for name, step in steps.items():
            floats[name] = params[name].clone()
            params[name].copy_(torch.clamp(torch.round(params[name] / step), -qmax, qmax) * step)
    return floats


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

    ``init_state`` is a state dict to start from instead of a random init. With
    ``cfg.weight_bits`` set (which needs a trained ``init_state``), every forward and
    backward pass runs on quantized weights, so the logged CER and the returned model are
    the quantized model's.
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
    if init_state is not None:
        model.load_state_dict(init_state)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ctc = nn.CTCLoss(blank=vocab.BLANK_IDX, zero_infinity=True)

    # weight_bits=6 keeps each weight at step * an integer in [-31, 31]. The model holds
    # the quantized weights; the float ones come back only for the optimizer update. With
    # weight_bits unset, steps is empty and none of this touches the model.
    params = dict(model.named_parameters())
    qmax = 2 ** (cfg.weight_bits - 1) - 1 if cfg.weight_bits else 0
    steps = _weight_steps(model) if cfg.weight_bits else {}
    floats = _quantize(model, steps, qmax)

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
            # The gradient was taken at the quantized weights; apply it to the float ones
            # (one update is much smaller than a quantization step), then re-quantize.
            with torch.no_grad():
                for name, w in floats.items():
                    params[name].copy_(w)
            opt.step()
            floats = _quantize(model, steps, qmax)
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
