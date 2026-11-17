"""Integer model of the fabric: the AM and CM as the RTL's arithmetic computes them, in numpy.

Prototype. It reads the weight image (``shared/specs/weight_image.md``) rather than a
checkpoint, so its integers are exactly what the fabric holds, and runs the LSTM with the
fabric's integer arithmetic under one of two settings:

- ``RTL``: ``hardware/rtl`` as of main. The gate is accumulator bits [15:0]
  (``ACC_LSB`` 0), its bits [7:4] index 16-entry sigmoid and tanh tables, the cell sits at
  the scale of ``i * tanh(g)`` (2**-15, so +-1 in 16 bits), ``tanh(c)`` is indexed by cell
  bits [11:8], ``h`` is the low byte of ``o * tanh(c)`` and a score is the output
  accumulator's low byte. Those slices are placeholders in the RTL ("will move when
  shared/specs pins them down").
- ``Fmt``: formats lined up with the trained scales, each a parameter, to choose them by
  error rate: the table's index bits and step, the cell's fractional bits, and the
  output scores as full accumulators or an int8 slice.

``float_npz`` gives the float stand-ins (``standin.py``) the same image's weights, so every
comparison uses the same weights. ``IntAcousticModel`` and ``IntCharModel`` have the
stand-ins' interfaces, so ``bench.py`` runs them in place of the float ones.
``python fixed.py --n 10`` prints the AM's greedy CER under each setting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import time

import numpy as np
from standin import AcousticModel, log_softmax

from shared import vocab

H_FRAC = 7  # h is int8 * 2**-7
PASSES = ((0, 1), (3, 2))  # per pass: gate on PE array 0, on array 1 (nn.LSTM's i, f, g, o)


def _fields(rows: np.ndarray, width: int, n: int) -> np.ndarray:
    """``rows`` ``[R, words]`` uint32 -> ``[R, n]`` signed ``width``-bit fields, LSB first."""
    bits = np.unpackbits(rows.astype("<u4").view(np.uint8), axis=1, bitorder="little")
    v = (bits[:, : n * width].reshape(len(rows), n, width) << np.arange(width)).sum(-1)
    return np.where(v >= 1 << (width - 1), v - (1 << width), v).astype(np.int64)


def read_ints(mem: str | Path) -> dict:
    """The image's integers: per layer ``w`` ``[4H, n_x + H]`` and ``b`` ``[4H]`` in
    nn.LSTM gate order, ``proj_w`` ``[V, H]``, ``proj_b`` ``[V]``, and the manifest."""
    mem = Path(mem)
    m = json.loads(mem.with_suffix(".json").read_text())
    hidden, n_out = m["n_hidden"], m["n_out"]
    words = np.array([int(x, 16) for x in mem.read_text().split()], dtype=np.uint32)
    image = words.reshape(m["rows"], -1)
    r, layers = 0, []

    def take_pass(n_x: int) -> tuple[np.ndarray, np.ndarray]:
        nonlocal r
        b = _fields(image[r : r + 4], 24, hidden // 2)  # [4, H/2]: H/4 per array per row
        b = np.stack([b[:, : hidden // 4].ravel(), b[:, hidden // 4 :].ravel()])  # [2, H]
        w = _fields(image[r + 4 : r + 4 + n_x + hidden], 6, 2 * hidden)  # [inputs, 2H]
        r += 4 + n_x + hidden
        return w.T.reshape(2, hidden, -1), b

    for layer in range(m["n_layers"]):
        n_x = m["n_in"] if layer == 0 else hidden
        w = np.empty((4, hidden, n_x + hidden), np.int64)
        b = np.empty((4, hidden), np.int64)
        for gates in PASSES:
            wp, bp = take_pass(n_x)
            w[list(gates)], b[list(gates)] = wp, bp
        layers.append({"w": w.reshape(4 * hidden, -1), "b": b.ravel(), "n_x": n_x})
    wp, bp = take_pass(0)  # the output layer: its biases, then one row per top-layer h
    assert r == len(image)
    return {"layers": layers, "proj_w": wp[0, :n_out], "proj_b": bp[0, :n_out], "m": m}


def float_npz(ints: dict) -> dict[str, np.ndarray]:
    """The image's weights as the float stand-ins take them (``export_npz.py``'s arrays):
    the 6-bit model in float, what decode.ipynb runs in torch."""
    lf, pf = ints["m"]["lstm_frac"], ints["m"]["proj_frac"]
    npz = {}
    for n, layer in enumerate(ints["layers"]):
        npz[f"w_ih{n}"] = layer["w"][:, : layer["n_x"]] * 2.0**-lf
        npz[f"w_hh{n}"] = layer["w"][:, layer["n_x"] :] * 2.0**-lf
        npz[f"b{n}"] = layer["b"] * 2.0 ** -(lf + H_FRAC)
    npz["proj_w"] = ints["proj_w"] * 2.0**-pf
    npz["proj_b"] = ints["proj_b"] * 2.0 ** -(pf + H_FRAC)
    return npz


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer matmul through float64 BLAS: exact, since every sum here is far below 2**53."""
    return np.rint(a.astype(np.float64) @ b).astype(np.int64)


def _wrap(x: np.ndarray, bits: int) -> np.ndarray:
    """Two's-complement wrap to ``bits``, as a register of that width keeps it."""
    return ((x + (1 << (bits - 1))) & ((1 << bits) - 1)) - (1 << (bits - 1))


def _rshift(x: np.ndarray, k: int) -> np.ndarray:
    """Arithmetic right shift by ``k`` with round-half-up (``k`` may be 0 or negative)."""
    return x << -k if k <= 0 else (x + (1 << (k - 1))) >> k


# The RTL's 16-entry tables (sigmoid_lut.v, tanh_lut.v), offset binary: zero at index 8.
RTL_SIG = np.array([0, 0, 1, 2, 5, 12, 30, 69, 128, 186, 225, 243, 250, 253, 254, 255])
RTL_TANH = np.array([-127] * 5 + [-126, -122, -97, 0, 97, 122, 126] + [127] * 4)


class RTL:
    """The arithmetic in hardware/rtl as of main, slice for slice, whatever the model's scale."""

    def gates(self, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """24-bit accumulators -> (sigmoid 0..255, tanh -127..127) of every lane."""
        idx = ((_wrap(acc, 16) >> 4) & 0xF) ^ 0x8
        return RTL_SIG[idx], RTL_TANH[idx]

    def cell(self, i, f, g, c):
        return _wrap(_wrap((f * c) >> 8, 16) + i * g, 16)

    def hidden(self, o, c):
        return _wrap(o * RTL_TANH[((c >> 8) & 0xF) ^ 0x8], 8)

    def logits(self, acc: np.ndarray, out_frac: int) -> np.ndarray:
        return _wrap(acc, 8).astype(np.float64)  # the int8 score, taken as a logit


@dataclass
class Fmt:
    """Formats lined up with the trained scales.

    A gate's accumulator (2**-frac) is rounded to steps of 2**-step_log2, saturated to an
    ``index_bits`` table index, and looked up in sigmoid (u8, /256) and tanh (s8, /128)
    tables. The cell is 16-bit at 2**-c_frac and indexes the same tanh table. ``h`` is
    ``o * tanh(c)`` (2**-15) rounded to int8 at 2**-7. Output scores are the full
    accumulators, or with ``score_frac`` set, int8 at 2**-score_frac.
    """

    frac: int
    index_bits: int = 8
    step_log2: int = 5  # 8 bits at 2**-5: the table spans +-4
    c_frac: int = 11  # +-16 in 16 bits
    score_frac: int | None = None

    def __post_init__(self) -> None:
        """Build the two tables."""
        half = 1 << (self.index_bits - 1)
        v = (np.arange(2 * half) - half) * 2.0**-self.step_log2
        self.sig = np.clip(np.round(256 / (1 + np.exp(-v))), 0, 255).astype(np.int64)
        self.tanh = np.clip(np.round(128 * np.tanh(v)), -127, 127).astype(np.int64)
        self.half = half

    def _index(self, x: np.ndarray, frac: int) -> np.ndarray:
        return np.clip(_rshift(x, frac - self.step_log2), -self.half, self.half - 1) + self.half

    def gates(self, acc):
        idx = self._index(acc, self.frac)
        return self.sig[idx], self.tanh[idx]

    def cell(self, i, f, g, c):
        c = _rshift(f * c, 8) + _rshift(i * g, 15 - self.c_frac)
        return np.clip(c, -(1 << 15), (1 << 15) - 1)

    def hidden(self, o, c):
        return np.clip(_rshift(o * self.tanh[self._index(c, self.c_frac)], 8), -128, 127)

    def logits(self, acc, out_frac):
        if self.score_frac is None:
            return acc * 2.0**-out_frac
        s = np.clip(_rshift(acc, out_frac - self.score_frac), -128, 127)
        return s * 2.0**-self.score_frac


# The formats bench.py's --fabric runs, picked by the greedy sweep (fabric-integer-model).
FABRIC = {"index_bits": 9, "step_log2": 5, "c_frac": 11, "score_frac": 2}


def fabric_fmt(ints: dict) -> Fmt:
    """``FABRIC``'s formats for a model: its gate accumulators sit at 2**-(lstm_frac + 7)."""
    return Fmt(ints["m"]["lstm_frac"] + H_FRAC, **FABRIC)


class _IntLstm:
    """The fabric's view of one model: integer weights, its shifts and ``arith``'s formats."""

    def __init__(self, ints: dict, arith) -> None:
        """``arith`` is ``RTL`` or a ``Fmt``."""
        self.arith = arith
        m = ints["m"]
        self.x_shift = H_FRAC - m["x_frac"]  # layer 0's x-phase products are shifted left
        self.out_frac = m["proj_frac"] + H_FRAC
        self.w_x = [layer["w"][:, : layer["n_x"]] for layer in ints["layers"]]  # [4H, n_x]
        self.w_h_t = [
            layer["w"][:, layer["n_x"] :].T.astype(np.float64) for layer in ints["layers"]
        ]
        self.b = [layer["b"] for layer in ints["layers"]]
        self.proj_t = ints["proj_w"].T.astype(np.float64)
        self.proj_b = ints["proj_b"]
        self.n_layers, self.hidden = len(self.b), len(self.b[0]) // 4

    def _layer(self, pre: np.ndarray, h: np.ndarray, c: np.ndarray, n: int):
        """Layer ``n``'s step: ``pre`` (bias + x-phase), ``h``, ``c`` ``[..., H]`` -> new h, c."""
        sig, tanh = self.arith.gates(_wrap(pre + _matmul(h, self.w_h_t[n]), 24))
        hid = self.hidden
        c = self.arith.cell(
            sig[..., :hid], sig[..., hid : 2 * hid], tanh[..., 2 * hid : 3 * hid], c
        )
        return self.arith.hidden(sig[..., 3 * hid :], c), c

    def out(self, h: np.ndarray) -> np.ndarray:
        """Top-layer ``h`` -> logits from the output layer's accumulators."""
        acc = _wrap(_matmul(h, self.proj_t) + self.proj_b, 24)
        return self.arith.logits(acc, self.out_frac)


class IntAcousticModel(_IntLstm):
    """The AM over an utterance, layer by layer, as the fabric computes it."""

    def logits(self, q: np.ndarray) -> np.ndarray:
        """int8 features ``[T, n_in]`` -> logits ``[T, V]``."""
        seq = q.astype(np.int64) << self.x_shift
        for n in range(self.n_layers):
            pre = _matmul(seq, self.w_x[n].T.astype(np.float64)) + self.b[n]  # every frame
            h = np.zeros(self.hidden, np.int64)
            c = np.zeros(self.hidden, np.int64)
            out = np.empty((len(seq), self.hidden), np.int64)
            for t in range(len(seq)):
                h, c = self._layer(pre[t], h, c, n)
                out[t] = h
            seq = out
        return self.out(seq)


class IntCharModel(_IntLstm):
    """The CM as the beam search's ``CharLM`` (``standin.CharModel``'s interface), as the
    fabric computes it. A prefix's state is its integer ``(h, c)`` ``[layers, H]``."""

    def __init__(self, ints: dict, arith) -> None:
        """Zero the counters ``bench.py`` reads."""
        super().__init__(ints, arith)
        self.standin_s = 0.0
        self.steps = self.calls = 0

    def _forward(self, chars: list[int], h: np.ndarray, c: np.ndarray):
        """One batched step: ``h``, ``c`` ``[layers, N, H]`` -> logits ``[N, V]``, new states."""
        x = self.w_x[0][:, chars].T << self.x_shift  # one-hot: a weight column, shifted
        h_new, c_new = np.empty_like(h), np.empty_like(c)
        for n in range(self.n_layers):
            if n:
                x = _matmul(h_new[n - 1], self.w_x[n].T.astype(np.float64))
            h_new[n], c_new[n] = self._layer(x + self.b[n], h[n], c[n], n)
        return self.out(h_new[-1]), h_new, c_new

    def start(self):
        """Feed EOS from the zero state, as at a sentence boundary in training."""
        zeros = np.zeros((self.n_layers, 1, self.hidden), np.int64)
        logits, h, c = self._forward([vocab.EOS_IDX], zeros, zeros)
        return (h[:, 0], c[:, 0]), log_softmax(logits)[0]

    def step(self, states, chars):
        """Feed ``chars[n]`` to ``states[n]``, all in one batch."""
        t0 = time.perf_counter()
        h = np.stack([s[0] for s in states], axis=1)
        c = np.stack([s[1] for s in states], axis=1)
        logits, h, c = self._forward(chars, h, c)
        new = [(h[:, k], c[:, k]) for k in range(len(chars))]
        self.standin_s += time.perf_counter() - t0
        self.steps += len(chars)
        self.calls += 1
        return new, log_softmax(logits)


def main() -> None:
    """Greedy CER of the float stand-in, the RTL as-is and the lined-up formats."""
    import jiwer
    import soundfile as sf
    from bench import eval_sample, list_utterances
    from runtime.features import Fbank

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--root", type=Path, default=Path.home() / "stt-proto")
    p.add_argument("--am", type=Path, default=Path.home() / "sr_accel/weights/am_fabric.mem")
    args = p.parse_args()

    ints = read_ints(args.am)
    m = ints["m"]
    fb = Fbank(np.array(m["cmvn_mean"]), np.array(m["cmvn_std"]))
    frac = m["lstm_frac"] + H_FRAC
    models = {
        "float": AcousticModel(float_npz(ints)),
        "rtl": IntAcousticModel(ints, RTL()),
        "fmt": IntAcousticModel(ints, Fmt(frac)),
        "fmt int8 scores": IntAcousticModel(ints, Fmt(frac, score_frac=2)),
    }
    items = eval_sample(list_utterances(args.root / "data/LibriSpeech/test-clean"))[: args.n]
    refs, hyps = [], {k: [] for k in models}
    for k, (flac, ref) in enumerate(items):
        wav, _ = sf.read(str(flac), dtype="float32")
        q = fb.int8(fb.extract(wav))
        refs.append(ref)
        for name, model in models.items():
            x = q * 2.0 ** -m["x_frac"] if name == "float" else q
            hyps[name].append(vocab.collapse(np.asarray(model.logits(x)).argmax(-1)))
        print(f"[{k + 1}/{len(items)}] {flac.stem}", flush=True)
    for name in models:
        print(
            f"{name:16s} greedy CER {jiwer.cer(refs, hyps[name]):.4f}  "
            f"WER {jiwer.wer(refs, hyps[name]):.4f}  | {hyps[name][0][:70]!r}"
        )
    print(f"{'ref':16s} {refs[0][:70]!r}")


if __name__ == "__main__":
    main()
