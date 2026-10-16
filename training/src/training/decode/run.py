"""Decoding a test set end to end: AM log-probs, the CM in torch, WER, tuning the LM weights.

The AM runs once over the whole set; its log-probs are kept so the beam search can be
rerun with other weights without it. The search runs on the CPU, one utterance at a
time; the CM's steps are batched across the beam.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import replace
from pathlib import Path

import jiwer
import numpy as np
import torch
from torch.utils.data import DataLoader

from training import vocab
from training.am.dataset import LibriSpeechFeatures, Utterance, collate
from training.am.model import AcousticModel
from training.cm.model import CharModel, State
from training.decode.beam import CharLM, Config, WordLM, beam_search


def log(log_path: str | Path, msg: str) -> None:
    """Print ``msg`` and append it to ``log_path``."""
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_models(
    am_path: str | Path, cm_path: str | Path, device: str
) -> tuple[AcousticModel, CharModel, torch.Tensor, torch.Tensor]:
    """The AM, the CM (both in eval mode on ``device``) and the AM's CMVN statistics.

    Takes the checkpoints ``save_run`` writes (``am.pt`` / ``am_6bit.pt``, ``cm.pt`` /
    ``cm_6bit.pt``), at the default sizes, 3 x 256 and 2 x 256.
    """
    am_ckpt = torch.load(am_path, map_location="cpu")
    am = AcousticModel()
    am.load_state_dict(am_ckpt["model"])
    cm = CharModel()
    cm.load_state_dict(torch.load(cm_path, map_location="cpu")["model"])
    return am.to(device).eval(), cm.to(device).eval(), am_ckpt["cmvn_mean"], am_ckpt["cmvn_std"]


@torch.no_grad()
def am_log_probs(
    model: AcousticModel,
    items: list[Utterance],
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int = 32,
    num_workers: int = 2,
) -> list[np.ndarray]:
    """Per-frame AM log-probs ``[T, V]`` for every utterance in ``items``, in order."""
    device = next(model.parameters()).device
    loader = DataLoader(
        LibriSpeechFeatures(items, mean, std),
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
    )
    model.eval()
    out: list[np.ndarray] = []
    for feats, feat_len, _, _ in loader:
        logp = model(feats.to(device), feat_len).cpu().numpy()
        out += [lp[:n] for lp, n in zip(logp, feat_len.tolist(), strict=True)]
    return out


class TorchCharLM:
    """The CM as the beam search's ``CharLM``: one ``(h, c)`` per prefix, a step per batch."""

    def __init__(self, model: CharModel) -> None:
        """Wrap a trained CM; it runs on whatever device it is on."""
        self.model = model.eval()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def start(self) -> tuple[State, np.ndarray]:
        """Feed EOS from the zero state, as at a sentence boundary in training."""
        logp, (h, c) = self.model(torch.tensor([[vocab.EOS_IDX]], device=self.device))
        return (h[:, 0], c[:, 0]), logp[0, 0].cpu().numpy()

    @torch.no_grad()
    def step(self, states: list[State], chars: list[int]) -> tuple[list[State], np.ndarray]:
        """Feed ``chars[n]`` to ``states[n]``, all in one LSTM call."""
        h = torch.stack([s[0] for s in states], dim=1)  # [layers, N, hidden]
        c = torch.stack([s[1] for s in states], dim=1)
        x = torch.tensor(chars, device=self.device).unsqueeze(1)
        logp, (h, c) = self.model(x, (h, c))
        return list(zip(h.unbind(1), c.unbind(1), strict=True)), logp[:, 0].cpu().numpy()


def greedy(logps: list[np.ndarray]) -> list[str]:
    """Per-frame argmax, CTC-collapsed: no search, no LM."""
    return [vocab.collapse(lp.argmax(-1)) for lp in logps]


def decode(
    logps: list[np.ndarray],
    cfg: Config,
    cm: CharLM | None = None,
    wlm: WordLM | None = None,
) -> list[str]:
    """The best transcript for every utterance, printing progress every 200."""
    hyps: list[str] = []
    t0 = time.perf_counter()
    for i, logp in enumerate(logps, 1):
        hyps.append(beam_search(logp, cfg, cm, wlm)[0][0])
        if i % 200 == 0:
            print(f"  {i}/{len(logps)} ({(time.perf_counter() - t0) / i:.2f} s/utt)", flush=True)
    return hyps


def error_rates(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    """WER and CER of ``hyps`` against ``refs``."""
    return float(jiwer.wer(refs, hyps)), float(jiwer.cer(refs, hyps))


def tune(
    logps: list[np.ndarray],
    refs: list[str],
    cfg: Config,
    grid: dict[str, list[float]],
    log_path: str | Path,
    cm: CharLM | None = None,
    wlm: WordLM | None = None,
) -> Config:
    """Grid-search the ``Config`` fields in ``grid`` for the lowest WER; return the best.

    Every point and its error rates are printed and written to ``log_path`` (appended).
    """
    best: tuple[float, Config] | None = None
    for values in itertools.product(*grid.values()):
        point = replace(cfg, **dict(zip(grid, values, strict=True)))
        t0 = time.perf_counter()
        wer, cer = error_rates(refs, decode(logps, point, cm, wlm))
        settings = ", ".join(f"{k}={v}" for k, v in zip(grid, values, strict=True))
        log(log_path, f"{settings}: WER {wer:.4f} CER {cer:.4f} ({time.perf_counter() - t0:.0f}s)")
        if best is None or wer < best[0]:
            best = (wer, point)
    log(log_path, f"best: {best[1]}")
    return best[1]
