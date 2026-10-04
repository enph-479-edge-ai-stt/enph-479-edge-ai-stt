"""The training loop the acoustic and character models share.

Runs start to finish in one session: logs progress and, after every epoch, the model's
own evaluation (to the notebook output and a log file). With ``weight_bits`` set it
fine-tunes a trained model with its weights quantized, the paper's "retraining based
fixed-point optimization" (weights only; activations and the cell stay float). The
weights land on the FPGA's fixed-point grid, so ``lstm/export.py`` can pack the result.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from training.lstm.export import LSTM_FRAC, PROJ_FRAC
from training.lstm.model import LstmNet


@dataclass
class TrainConfig:
    """Hyperparameters every run has. Each model subclasses this with its own."""

    n_hidden: int = 256
    n_layers: int = 3
    lr: float = 3e-4
    grad_clip: float = 5.0
    batch_size: int = 32
    epochs: int = 20
    weight_bits: int | None = None  # quantize the weights to this many bits (fine-tuning)
    log_every: int = 50  # batches between progress lines
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def fine_tune(self) -> TrainConfig:
        """The run that follows this one: the same model with its weights held at 6 bits.

        A tenth of the learning rate for a quarter of the epochs. Both are first guesses.
        """
        # Rounded so the log shows 3e-05, not 2.9999999999999997e-05.
        lr = round(self.lr / 10, 10)
        return replace(self, lr=lr, epochs=max(1, self.epochs // 4), weight_bits=6)


def _log(log_path: Path, msg: str) -> None:
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _weight_steps(model: LstmNet) -> dict[str, float]:
    """Per weight matrix, its step size on the FPGA (the fixed-point formats in ``export``).

    Powers of two, and the same for a layer's input and hidden weights: the fabric sums
    both products in one accumulator and rescales with a shift.
    """
    return {
        name: 2.0 ** -(PROJ_FRAC if name.startswith("proj") else LSTM_FRAC)
        for name, _ in model.named_parameters()
        if "weight" in name
    }


def _quantize(model: LstmNet, steps: dict[str, float], qmax: int) -> dict[str, torch.Tensor]:
    """Swap each weight in ``steps`` for its quantized value; return the float originals."""
    params = dict(model.named_parameters())
    floats = {}
    with torch.no_grad():
        for name, step in steps.items():
            floats[name] = params[name].clone()
            params[name].copy_(torch.clamp(torch.round(params[name] / step), -qmax, qmax) * step)
    return floats


def fit(
    model: LstmNet,
    cfg: TrainConfig,
    losses: Callable[[], Iterator[torch.Tensor]],
    n_batches: int,
    show_loss: Callable[[float], str],
    evaluate: Callable[[], str],
    log_path: str | Path,
    init_state: dict[str, torch.Tensor] | None = None,
) -> None:
    """Train ``model`` in place on ``cfg.device``, logging to ``log_path`` (overwritten).

    The model supplies what differs. ``losses()`` yields one epoch's loss a batch at a
    time (``n_batches`` of them); each is backpropagated before the next is asked for.
    ``show_loss`` formats a mean training loss for the log, and ``evaluate()`` returns
    the text logged after an epoch.

    ``init_state`` is a state dict to start from instead of the model's random init. With
    ``cfg.weight_bits`` set (which needs a trained ``init_state``), every forward and
    backward pass runs on quantized weights, so what ``evaluate`` sees and what the model
    is left with are the quantized model's.
    """
    log_path = Path(log_path)
    log_path.write_text("", encoding="utf-8")
    _log(log_path, str(cfg))
    if init_state is not None:
        model.load_state_dict(init_state)
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # weight_bits=6 keeps each weight at step * an integer in [-31, 31]. The model holds
    # the quantized weights; the float ones come back only for the optimizer update. With
    # weight_bits unset, steps is empty and none of this touches the model.
    params = dict(model.named_parameters())
    qmax = 2 ** (cfg.weight_bits - 1) - 1 if cfg.weight_bits else 0
    steps = _weight_steps(model) if cfg.weight_bits else {}
    floats = _quantize(model, steps, qmax)

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        loss_sum = 0.0
        for b, loss in enumerate(losses(), 1):
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
                ms_per_batch = (time.perf_counter() - t0) / b * 1e3
                _log(
                    log_path,
                    f"epoch {epoch} batch {b}/{n_batches} {show_loss(loss_sum / b)} "
                    f"({ms_per_batch:.0f} ms/batch)",
                )
        train_s = time.perf_counter() - t0

        result = evaluate()
        _log(
            log_path,
            f"[epoch {epoch}] {show_loss(loss_sum / n_batches)} | train {train_s:.0f}s, "
            f"eval {time.perf_counter() - t0 - train_s:.0f}s | {result}",
        )
