"""Write the features' golden case, ``shared/golden/features/case.npz``.

It pins what ``fbank.py`` computes, so the board's numpy port (``runtime/features.py``,
which may not import this package) can check it computes the same. The input is one real
test-clean utterance next to the case. The CMVN statistics are the clip's own, standing
in for an AM's: what is pinned is the arithmetic, not a checkpoint. The case holds

- ``feats``: ``fbank.extract``'s ``[T, 123]`` float32 features, before CMVN
- ``cmvn_mean``, ``cmvn_std``: ``[123]`` float32
- ``x_frac``: the AM's input step, ``2**-x_frac``
- ``int8``: ``[T, 123]``, the frames the fabric gets (CMVN, scale, round, clamp)

Run ``python -m training.features.golden`` after any change to the features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from training.am.model import AcousticModel
from training.features import fbank

GOLDEN_DIR = Path(__file__).resolve().parents[4] / "shared" / "golden" / "features"
CLIP = "6930-75918-0000.flac"  # "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS", 3.5 s


def to_int8(feats: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, x_frac: int):
    """CMVN'd features as the fabric gets them: int8 at step ``2**-x_frac``."""
    x = fbank.apply_cmvn(feats, mean, std) * 2.0**x_frac
    return torch.round(x).clamp(-128, 127).to(torch.int8)


def extract(flac: Path) -> torch.Tensor:
    """``fbank.extract`` on a FLAC, read as training's dataset reads it."""
    wav, sr = sf.read(str(flac), dtype="float32")
    assert sr == fbank.SAMPLE_RATE
    mel, deltas = fbank.build_transforms()
    return fbank.extract(torch.from_numpy(wav).unsqueeze(0), mel, deltas)


def main(out_dir: Path = GOLDEN_DIR) -> None:
    """Extract the clip's features, CMVN them on its own statistics and write the case."""
    feats = extract(out_dir / CLIP)
    mean, std = fbank.compute_cmvn([feats])
    x_frac = AcousticModel.x_frac
    q = to_int8(feats, mean, std, x_frac)
    np.savez(
        out_dir / "case.npz",
        feats=feats.numpy(),
        cmvn_mean=mean.numpy(),
        cmvn_std=std.numpy(),
        x_frac=np.int64(x_frac),
        int8=q.numpy(),
    )
    print(f"wrote {out_dir / 'case.npz'}: {tuple(feats.shape)} features")


if __name__ == "__main__":
    main()
