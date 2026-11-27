r"""One utterance through the fabric, checked score for score against the integer model.

Prototype. Two steps, because the fabric needs root and PYNQ's venv, and the integer
model runs in the prototype's venv. From this folder, with ``stt-runtime`` and
``stt-shared`` installed in both (``runtime/README.md``):

    ~/stt-proto/venv/bin/python predict_one.py prep ~/sr_accel/weights one.npz
    sudo /usr/local/share/pynq-venv/bin/python3 predict_one.py run \
        ~/sr_accel/srcdst/sr_accel.bit ~/sr_accel/weights one.npz

``prep`` takes decode.ipynb's first test-clean utterance, makes its int8 features and the
integer model's expected scores (``fixed.FABRIC`` formats): every AM frame, and one CM step
(EOS from the zero state, as ``IntCharModel.start``). ``run`` loads the bitstream and both
weight images (``runtime.fabric.Fabric``), runs one AM job per frame and the CM step, and
compares.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from shared import vocab


def prep(images: Path, out: Path) -> None:
    """Features and expected integer scores for one utterance, saved to ``out``."""
    import fixed
    import soundfile as sf
    from bench import eval_sample, list_utterances

    from runtime.features import Fbank

    am_ints = fixed.read_ints(images / "am_fabric.mem")
    cm_ints = fixed.read_ints(images / "cm_fabric.mem")
    m = am_ints["m"]
    test = list_utterances(Path.home() / "stt-proto/data/LibriSpeech/test-clean")
    flac, ref = eval_sample(test)[0]
    wav, _ = sf.read(str(flac), dtype="float32")
    fb = Fbank(np.array(m["cmvn_mean"]), np.array(m["cmvn_std"]))
    q = fb.int8(fb.extract(wav))

    am_fmt, cm_fmt = fixed.fabric_fmt(am_ints), fixed.fabric_fmt(cm_ints)
    am = fixed.IntAcousticModel(am_ints, am_fmt)
    am_scores = np.rint(am.logits(q) * 2**am_fmt.score_frac).astype(np.int8)
    cm = fixed.IntCharModel(cm_ints, cm_fmt)
    zeros = np.zeros((cm.n_layers, 1, cm.hidden), np.int64)
    cm_logits, _, _ = cm._forward([vocab.EOS_IDX], zeros, zeros)
    cm_scores = np.rint(cm_logits[0] * 2**cm_fmt.score_frac).astype(np.int8)

    np.savez(out, q=q, am_scores=am_scores, cm_scores=cm_scores, ref=ref, utt=flac.stem)
    print(f"{flac.stem}: {len(q)} frames, ref {ref!r}")
    print(f"integer model greedy: {vocab.collapse(am_scores.argmax(-1))!r}")


def run(bitfile: str, images: Path, npz: Path) -> None:
    """Every frame of the saved utterance and one CM step on the fabric, against ``prep``."""
    from runtime.fabric import STATUS, Fabric
    from runtime.models import FabricAcousticModel

    d = np.load(npz)
    q, want_am, want_cm = d["q"], d["am_scores"], d["cm_scores"]
    t = time.perf_counter()
    fabric = Fabric(bitfile)
    fabric.load_images(images / "am_fabric.mem", images / "cm_fabric.mem")
    print(f"bitstream and weights loaded in {time.perf_counter() - t:.2f} s")

    t = time.perf_counter()
    got = FabricAcousticModel(fabric).scores(q)
    dt = time.perf_counter() - t
    print(f"AM: {len(q)} frames in {dt:.2f} s ({dt / len(q) * 1e3:.2f} ms per frame, "
          f"{len(q) / 100 / dt:.1f}x real time incl. Python)")

    exact = (got == want_am).all(-1)
    diff = np.abs(got.astype(int) - want_am)
    print(f"AM scores: {exact.sum()}/{len(q)} frames exact, {np.mean(diff == 0):.2%} of scores, "
          f"max |diff| {diff.max()}")
    if not exact.all():
        k = int(np.argmin(exact))
        print(f"  first mismatch, frame {k}:")
        print(f"    fabric {got[k].tolist()}\n    model  {want_am[k].tolist()}")
    print(f"ref:            {str(d['ref'])!r}")
    print(f"fabric greedy:  {vocab.collapse(got.argmax(-1))!r}")
    print(f"integer greedy: {vocab.collapse(want_am.argmax(-1))!r}")

    fabric.clear_slot(0)
    cm = fabric.lm([0], [0], [vocab.EOS_IDX])[0]
    print(f"CM step (EOS, zero state): {'exact' if (cm == want_cm).all() else 'MISMATCH'}"
          f"\n    fabric {cm.tolist()}\n    model  {want_cm.tolist()}")
    print(f"STATUS 0x{fabric.regs.read(STATUS):08x}")


def main() -> None:
    """``prep`` or ``run``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep")
    s.add_argument("images", type=Path)
    s.add_argument("out", type=Path)
    s = sub.add_parser("run")
    s.add_argument("bitfile")
    s.add_argument("images", type=Path)
    s.add_argument("npz", type=Path)
    args = p.parse_args()
    if args.cmd == "prep":
        prep(args.images, args.out)
    else:
        run(args.bitfile, args.images, args.npz)


if __name__ == "__main__":
    sys.exit(main())
