"""Repack a training run's weight image into the rows the fabric's loader takes.

Training writes one image per model (``<name>_fabric.mem``, ``shared/specs/weight_image.md``):
per LSTM pass its 4 bias rows then one weight row per input, and the output layer as one
more pass. The RTL keeps the LSTM weights, the LSTM biases, the output-layer weights and
the output-layer biases in four memories instead, each its own load target with its own
row format (``shared/specs/fabric_interface.md``, "Loading"). ``repack`` turns an image
into the four loads; ``fabric.py`` streams them in.

Pure numpy, so it runs on the board and on a PC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WEIGHT_BITS = 6
BIAS_BITS = 24

# LOAD_TARGET values (fabric_interface.md).
T_WEIGHT = 1
T_BIAS = 2
T_OUT_WEIGHT = 3
T_OUT_BIAS = 4


@dataclass
class Load:
    """One DMA stream into the fabric: ``words`` written from row ``addr`` of ``target``."""

    target: int
    addr: int
    words: np.ndarray  # uint32, rows back to back, least significant word first
    rows: int  # what LOAD_COUNT should read once the stream is in


@dataclass
class Base:
    """Where a model's rows go in each memory. The AM fills each from 0, the CM follows."""

    weight_row: int
    layer: int  # LSTM bias row = 2 * (layer + model's layer) + pass
    out_weight_row: int
    out_bias_row: int


AM_BASE = Base(weight_row=0, layer=0, out_weight_row=0, out_bias_row=0)


def cm_base(am: dict) -> Base:
    """The CM's rows start where the AM's (manifest ``am``) end."""
    return Base(
        weight_row=_lstm_weight_rows(am),
        layer=am["n_layers"],
        out_weight_row=am["n_hidden"],
        out_bias_row=1,
    )


def read_image(mem: str | Path) -> tuple[np.ndarray, dict]:
    """A ``.mem`` image as ``[rows, words per row]`` uint32, and its ``.json`` manifest."""
    mem = Path(mem)
    manifest = json.loads(mem.with_suffix(".json").read_text(encoding="utf-8"))
    words = np.array([int(line, 16) for line in mem.read_text().split()], dtype=np.uint32)
    per_row = _words_per_row(manifest["n_hidden"])
    if words.size != manifest["rows"] * per_row:
        raise ValueError(f"{mem}: {words.size} words, manifest says {manifest['rows']} rows")
    return words.reshape(manifest["rows"], per_row), manifest


def repack(image: np.ndarray, manifest: dict, base: Base) -> list[Load]:
    """The four loads that put one model's image where the controller reads it."""
    hidden, n_in, n_layers = manifest["n_hidden"], manifest["n_in"], manifest["n_layers"]
    bias_rows = BIAS_BITS // WEIGHT_BITS  # a pass's biases fill 4 rows of the weight width
    half = image.shape[1] // 2  # words per PE array in an image row

    weights, biases = [], []
    r = 0
    for layer in range(n_layers):
        n_x = n_in if layer == 0 else hidden
        for _ in range(2):  # passes
            b = image[r : r + bias_rows]
            # A bias row holds a slice of each array's biases, array 0 in its low half.
            # The fabric's bias row is all of array 0's, then all of array 1's.
            biases.append(np.concatenate([b[:, :half].ravel(), b[:, half:].ravel()]))
            weights.append(image[r + bias_rows : r + bias_rows + n_x + hidden])
            r += bias_rows + n_x + hidden
    # Output layer: its n_out PEs are the first of array 0, the low bits of each row (and
    # of array 0's biases, joined across the pass's bias rows as above).
    out_weight_words = -(-manifest["n_out"] * WEIGHT_BITS // 32)
    out_bias_words = -(-manifest["n_out"] * BIAS_BITS // 32)
    out_bias = image[r : r + bias_rows, :half].ravel()[:out_bias_words]
    out_weights = image[r + bias_rows : r + bias_rows + hidden, :out_weight_words]
    r += bias_rows + hidden
    if r != image.shape[0]:
        raise ValueError(f"image has {image.shape[0]} rows, the manifest's model needs {r}")

    def load(target: int, addr: int, rows: list[np.ndarray] | np.ndarray) -> Load:
        rows = np.concatenate(rows) if isinstance(rows, list) else rows
        words = np.ascontiguousarray(rows, dtype=np.uint32).ravel()
        return Load(target, addr, words, len(rows))

    return [
        load(T_WEIGHT, base.weight_row, weights),
        load(T_BIAS, 2 * base.layer, np.stack(biases)),
        load(T_OUT_WEIGHT, base.out_weight_row, out_weights),
        load(T_OUT_BIAS, base.out_bias_row, out_bias[None]),
    ]


def _words_per_row(hidden: int) -> int:
    return 2 * hidden * WEIGHT_BITS // 32


def _lstm_weight_rows(m: dict) -> int:
    return 2 * (m["n_in"] + m["n_hidden"]) + (m["n_layers"] - 1) * 4 * m["n_hidden"]
