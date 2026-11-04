"""Numpy stand-ins for the fabric: the AM and the CM as float LSTMs with the 6-bit weights.

Prototype only. At deployment the LSTMs run on the fabric (AGENTS.md: no RNN math in
software); these stand in for it so the ARM side can be run and timed on the board
before the fabric exists. They compute what ``decode.ipynb`` computed in torch (float
LSTMs holding the fine-tuned 6-bit weights), so the decode results should match its run.
Both return logits: the log-softmax is the ARM's job and is timed as ARM work.

The weights come from ``export_npz.py``: per layer ``w_ih{l}`` ``[4H, n_in]``, ``w_hh{l}``
``[4H, H]`` and ``b{l}`` (``b_ih + b_hh``), plus ``proj_w`` and ``proj_b``. Gate order is
``nn.LSTM``'s (i, f, g, o).
"""

from __future__ import annotations

import time

import numpy as np

from shared import vocab


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _cell(gates: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i, f, g, o = np.split(gates, 4, axis=-1)
    c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
    return _sigmoid(o) * np.tanh(c), c


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Along the last axis. The ARM's step after either model."""
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class Lstm:
    """Weights of a stacked LSTM with an output layer, in float32."""

    def __init__(self, npz: dict[str, np.ndarray]) -> None:
        """Take the arrays ``export_npz.py`` wrote."""
        self.n_layers = sum(k.startswith("w_ih") for k in npz)
        self.w_ih = [npz[f"w_ih{n}"].astype(np.float32) for n in range(self.n_layers)]
        self.w_hh_t = [npz[f"w_hh{n}"].T.astype(np.float32).copy() for n in range(self.n_layers)]
        self.b = [npz[f"b{n}"].astype(np.float32) for n in range(self.n_layers)]
        self.proj_w_t = npz["proj_w"].T.astype(np.float32).copy()
        self.proj_b = npz["proj_b"].astype(np.float32)
        self.hidden = self.w_hh_t[0].shape[0]


class AcousticModel(Lstm):
    """The AM over a whole utterance, layer by layer."""

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Features ``[T, 123]`` (the int8 values at their real scale) -> logits ``[T, V]``."""
        seq = x.astype(np.float32)
        for n in range(self.n_layers):
            pre = seq @ self.w_ih[n].T + self.b[n]  # the input half of every frame at once
            h = np.zeros(self.hidden, np.float32)
            c = np.zeros(self.hidden, np.float32)
            out = np.empty((len(seq), self.hidden), np.float32)
            for t in range(len(seq)):
                h, c = _cell(pre[t] + h @ self.w_hh_t[n], c)
                out[t] = h
            seq = out
        return seq @ self.proj_w_t + self.proj_b


class CharModel(Lstm):
    """The CM as the beam search's ``CharLM``, one ``(h, c)`` ``[layers, H]`` per prefix.

    ``standin_s`` adds up the time ``step`` spends in the LSTM (the fabric's share, not the
    log-softmax after it), so the search's ARM time can be separated from it. ``steps``
    counts prefixes stepped and ``calls`` batched steps.
    """

    def __init__(self, npz: dict[str, np.ndarray]) -> None:
        """Take the arrays ``export_npz.py`` wrote and zero the counters."""
        super().__init__(npz)
        self.standin_s = 0.0
        self.steps = self.calls = 0

    def _forward(
        self, chars: list[int], h: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One batched step: ``h``, ``c`` ``[layers, N, H]`` -> logits ``[N, V]``, new states."""
        x = self.w_ih[0][:, chars].T  # one-hot input: a column of the weights
        h_new, c_new = np.empty_like(h), np.empty_like(c)
        for n in range(self.n_layers):
            if n:
                x = h_new[n - 1] @ self.w_ih[n].T
            h_new[n], c_new[n] = _cell(x + h[n] @ self.w_hh_t[n] + self.b[n], c[n])
        return h_new[-1] @ self.proj_w_t + self.proj_b, h_new, c_new

    def start(self) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Feed EOS from the zero state, as at a sentence boundary in training."""
        zeros = np.zeros((self.n_layers, 1, self.hidden), np.float32)
        logits, h, c = self._forward([vocab.EOS_IDX], zeros, zeros)
        return (h[:, 0], c[:, 0]), log_softmax(logits)[0]

    def step(
        self, states: list[tuple[np.ndarray, np.ndarray]], chars: list[int]
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Feed ``chars[n]`` to ``states[n]``, all in one batch.

        Gathering and splitting the states counts as stand-in time: on the board they stay
        in the fabric's slots.
        """
        t0 = time.perf_counter()
        h = np.stack([s[0] for s in states], axis=1)
        c = np.stack([s[1] for s in states], axis=1)
        logits, h, c = self._forward(chars, h, c)
        new = [(h[:, k], c[:, k]) for k in range(len(chars))]
        self.standin_s += time.perf_counter() - t0
        self.steps += len(chars)
        self.calls += 1
        return new, log_softmax(logits)
