"""Checkpoint (``am_6bit.pt`` / ``cm_6bit.pt``) -> ``.npz`` for ``standin.py``. Needs torch.

The board has no torch, so this runs where training does. The AM's ``.npz`` also carries
its CMVN statistics.

    python export_npz.py am_6bit.pt am.npz
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def main(src: str, dst: str) -> None:
    """Write ``src``'s LSTM, output layer and (for the AM) CMVN arrays to ``dst``."""
    ckpt = torch.load(src, map_location="cpu")
    sd = {k: v.numpy() for k, v in ckpt["model"].items()}
    out = {"proj_w": sd["proj.weight"], "proj_b": sd["proj.bias"]}
    n = 0
    while f"lstm.weight_ih_l{n}" in sd:
        out[f"w_ih{n}"] = sd[f"lstm.weight_ih_l{n}"]
        out[f"w_hh{n}"] = sd[f"lstm.weight_hh_l{n}"]
        out[f"b{n}"] = sd[f"lstm.bias_ih_l{n}"] + sd[f"lstm.bias_hh_l{n}"]
        n += 1
    if "cmvn_mean" in ckpt:
        out["cmvn_mean"] = ckpt["cmvn_mean"].numpy()
        out["cmvn_std"] = ckpt["cmvn_std"].numpy()
    np.savez(dst, **out)
    print(f"{dst}: {n} layers, " + ", ".join(f"{k} {v.shape}" for k, v in out.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
