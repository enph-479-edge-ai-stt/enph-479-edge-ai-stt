"""Truncated-BPTT training loop for the character model (CM).

The stream is cut into ``batch_size`` contiguous parallel streams and walked in
``bptt``-character chunks, carrying the LSTM state from one chunk to the next
(detached), so the model sees context across sentence boundaries just like the
EOS-joined stream it trains on. Optimizer is Adam, like the AM (the paper used
AdaDelta, which was much slower to converge here). Runs start to finish in one
session: logs dev bits per character and a sampled line of text after every epoch
(to the notebook output and a log file) and returns the trained model.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training import vocab
from training.cm.model import CharModel


@dataclass
class TrainConfig:
    """Hyperparameters for one training run."""

    n_hidden: int = 256
    n_layers: int = 2
    lr: float = 2e-3
    grad_clip: float = 5.0
    batch_size: int = 128  # parallel streams
    bptt: int = 100  # characters per truncated-BPTT chunk
    epochs: int = 5
    log_every: int = 1000  # chunks between progress lines
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _streams(stream: np.ndarray, batch_size: int) -> torch.Tensor:
    """Stream ``[N]`` -> ``[batch_size, N // batch_size]`` contiguous parallel streams."""
    n = len(stream) // batch_size * batch_size
    return torch.from_numpy(stream[:n].reshape(batch_size, -1))


def _chunks(streams: torch.Tensor, bptt: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """(inputs, next-char targets), each ``[B, <=bptt]``, walking the streams left to right."""
    for t in range(0, streams.size(1) - 1, bptt):
        y = streams[:, t + 1 : t + 1 + bptt]
        yield streams[:, t : t + y.size(1)], y


def _nll(logp: torch.Tensor, y: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.nll_loss(logp.flatten(0, 1), y.long().flatten(), reduction=reduction)


@torch.no_grad()
def bits_per_char(model: CharModel, streams: torch.Tensor, bptt: int) -> float:
    """Average next-character cross-entropy over ``streams`` in bits (EOS predictions included)."""
    model.eval()
    state = None
    total = 0.0
    for x, y in _chunks(streams, bptt):
        logp, state = model(x, state)
        total += _nll(logp, y, "sum").item()
    return total / (streams.size(0) * (streams.size(1) - 1)) / math.log(2)


@torch.no_grad()
def sample(model: CharModel, n_chars: int = 200, seed: int = 0) -> str:
    """Sample ``n_chars`` characters starting after an EOS; EOS renders as `` | ``."""
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    x = torch.tensor([[vocab.EOS_IDX]], device=device)
    state = None
    out: list[str] = []
    for _ in range(n_chars):
        logp, state = model(x, state)
        i = int(torch.multinomial(logp[0, -1].exp().cpu(), 1, generator=gen))
        out.append(" | " if i == vocab.EOS_IDX else vocab.CHARS[i])
        x = torch.tensor([[i]], device=device)
    return "".join(out)


def _log(log_path: Path, msg: str) -> None:
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def train(cfg: TrainConfig, streams: dict[str, np.ndarray], log_path: str | Path) -> CharModel:
    """Train on ``streams["train"]``; after each epoch, log bits/char on ``streams["test"]``.

    Each stream is a ``uint8`` array of vocab indices (see ``dataset.to_stream``). Progress
    and epoch lines are printed and written to ``log_path`` (overwritten). Returns the model.
    """
    log_path = Path(log_path)
    log_path.write_text("", encoding="utf-8")
    _log(log_path, str(cfg))
    device = torch.device(cfg.device)
    # uint8 on the device (1 byte/char); chunks are cast to long inside the model.
    train_streams = _streams(streams["train"], cfg.batch_size).to(device)
    test_streams = _streams(streams["test"], cfg.batch_size).to(device)

    model = CharModel(n_hidden=cfg.n_hidden, n_layers=cfg.n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_chunks = math.ceil((train_streams.size(1) - 1) / cfg.bptt)
    _log(log_path, f"train {train_streams.numel():,} chars as {n_chunks} chunks/epoch")
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        loss_sum = 0.0
        state = None
        for b, (x, y) in enumerate(_chunks(train_streams, cfg.bptt), 1):
            logp, state = model(x, state)
            state = tuple(s.detach() for s in state)  # truncate BPTT at the chunk edge
            loss = _nll(logp, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)  # LSTMs explode
            opt.step()
            loss_sum += loss.item()
            if b % cfg.log_every == 0:
                s_per_chunk = (time.perf_counter() - t0) / b
                _log(
                    log_path,
                    f"epoch {epoch} chunk {b}/{n_chunks} train bpc "
                    f"{loss_sum / b / math.log(2):.3f} ({s_per_chunk * 1e3:.0f} ms/chunk)",
                )
        train_s = time.perf_counter() - t0

        bpc = bits_per_char(model, test_streams, cfg.bptt)
        _log(
            log_path,
            f"[epoch {epoch}] train bpc {loss_sum / n_chunks / math.log(2):.3f} | "
            f"test bpc {bpc:.3f} | train {train_s:.0f}s, "
            f"eval {time.perf_counter() - t0 - train_s:.0f}s",
        )
        _log(log_path, f"  sample: {sample(model)}")
    return model
