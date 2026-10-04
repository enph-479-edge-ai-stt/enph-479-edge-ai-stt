"""Weight image for the FPGA: a 6-bit model, packed the way the fabric reads it.

``export`` writes ``<name>_fabric.mem``, the contents of the fabric's weight memory as
32-bit hex words in the order the board streams them in, and ``<name>_fabric.json``, what
the ARM needs to prepare the model's inputs. The layout and the fixed-point formats below
are the contract with ``hardware/`` and ``runtime/``, specified in
``shared/specs/weight_image.md``. Change the two together.

``save_run`` is what a notebook ends with: both checkpoints, both logs and the image.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import torch

from training.lstm.model import LstmNet

WEIGHT_BITS = 6  # signed: integers in [-31, 31]
LSTM_FRAC = 7  # an LSTM weight is its integer * 2**-7
PROJ_FRAC = 6  # an output-layer weight is its integer * 2**-6
H_FRAC = 7  # a hidden state is its int8 * 2**-7
BIAS_BITS = 24  # signed, at the scale of the accumulator it preloads

# The fabric has two arrays of PEs, so a layer takes two passes of two gates each. Per
# pass: the gate on PE array 0, then on PE array 1, as indices into nn.LSTM's stacking
# order (i, f, g, o). The RTL calls g "c".
PASSES = ((0, 1), (3, 2))


def _integers(w: torch.Tensor, frac: int) -> torch.Tensor:
    """The integers of weights already quantized to ``WEIGHT_BITS`` at step ``2**-frac``."""
    z = w * 2**frac
    if not torch.equal(z, z.round()) or z.abs().max() >= 2 ** (WEIGHT_BITS - 1):
        raise ValueError("weights are not on the 6-bit grid; export the fine-tuned model")
    return z.long()


def _words(values: list[int], width: int) -> list[int]:
    """Pack signed ``values`` side by side, value n at bits ``[width * n +: width]``.

    Returns the packed row as 32-bit words, least significant first.
    """
    row = 0
    for n, v in enumerate(values):
        row |= (v & (2**width - 1)) << (width * n)
    return [(row >> s) & 0xFFFFFFFF for s in range(0, width * len(values), 32)]


def _pass_rows(weights: torch.Tensor, biases: torch.Tensor) -> Iterator[list[int]]:
    """The rows of one pass: its biases, then one weight row per input.

    ``weights`` is ``[2, H, n_in]`` and ``biases`` ``[2, H]``, PE array 0 then 1. A row
    holds array 0 in its low half and array 1 in its high half. A weight row is one input's
    weight for every PE; the biases fill ``BIAS_BITS // WEIGHT_BITS`` rows of that width.
    """
    per_row = biases.shape[1] * WEIGHT_BITS // BIAS_BITS
    for r in range(0, biases.shape[1], per_row):
        yield _words(biases[:, r : r + per_row].flatten().tolist(), BIAS_BITS)
    for j in range(weights.shape[2]):
        yield _words(weights[:, :, j].flatten().tolist(), WEIGHT_BITS)


def export(model: LstmNet, out_dir: str | Path, name: str, **extra: object) -> None:
    """Write ``<name>_fabric.mem`` and ``.json`` for a model fine-tuned with 6-bit weights.

    ``extra`` is anything else the ARM needs for this model (the AM's CMVN statistics). It
    goes into the JSON next to the model's dimensions and number formats.
    """
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    hidden, n_out = model.lstm.hidden_size, model.proj.out_features
    rows: list[list[int]] = []
    for layer in range(model.lstm.num_layers):
        # [4H, n_in + H]: the x-phase inputs, then the h-phase ones. The fabric has one
        # bias per gate where nn.LSTM has two, so they add.
        w = torch.cat([sd[f"lstm.weight_ih_l{layer}"], sd[f"lstm.weight_hh_l{layer}"]], dim=1)
        b = sd[f"lstm.bias_ih_l{layer}"] + sd[f"lstm.bias_hh_l{layer}"]
        w = _integers(w, LSTM_FRAC).view(4, hidden, -1)
        b = torch.round(b * 2 ** (LSTM_FRAC + H_FRAC)).long().view(4, hidden)
        for gates in PASSES:
            rows += _pass_rows(w[list(gates)], b[list(gates)])
    # The output layer runs on the first n_out PEs of array 0; every other PE gets zeros.
    w = torch.zeros(2, hidden, hidden, dtype=torch.long)
    b = torch.zeros(2, hidden, dtype=torch.long)
    w[0, :n_out] = _integers(sd["proj.weight"], PROJ_FRAC)
    b[0, :n_out] = torch.round(sd["proj.bias"] * 2 ** (PROJ_FRAC + H_FRAC)).long()
    rows += _pass_rows(w, b)

    manifest = {
        "n_in": model.lstm.input_size,
        "n_hidden": hidden,
        "n_layers": model.lstm.num_layers,
        "n_out": n_out,
        "weight_bits": WEIGHT_BITS,
        "lstm_frac": LSTM_FRAC,
        "proj_frac": PROJ_FRAC,
        "h_frac": H_FRAC,
        "x_frac": model.x_frac,
        "bias_bits": BIAS_BITS,
        "rows": len(rows),
        "words_per_row": len(rows[0]),
        **extra,
    }
    out_dir = Path(out_dir)
    image = "".join(f"{word:08x}\n" for row in rows for word in row)
    (out_dir / f"{name}_fabric.mem").write_text(image, encoding="utf-8")
    (out_dir / f"{name}_fabric.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def save_run(
    run_dir: str | Path, name: str, model: LstmNet, qmodel: LstmNet, **extra: torch.Tensor
) -> None:
    """Save everything a training run leaves behind to ``run_dir``.

    That is the float model and the 6-bit one (``<name>.pt``, ``<name>_6bit.pt``), their
    logs (``train.log`` and ``train_6bit.log``, where the notebooks write them) and the
    6-bit model's weight image. ``extra`` is what the weights are useless without (the
    AM's CMVN statistics): it goes into both checkpoints and into the image's JSON.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for stem, m in [(name, model), (f"{name}_6bit", qmodel)]:
        torch.save({"model": m.state_dict(), **extra}, run_dir / f"{stem}.pt")
    for log in ["train.log", "train_6bit.log"]:
        shutil.copy(log, run_dir)
    export(qmodel, run_dir, name, **{k: v.tolist() for k, v in extra.items()})
