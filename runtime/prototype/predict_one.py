r"""One utterance through the fabric, checked score for score against the integer model.

Prototype. Two steps, because the fabric needs root and PYNQ's venv, and the features and
the integer model run in the prototype's venv:

    # ~/stt-proto/pr/runtime/prototype, PYTHONPATH=~/stt-proto/pr/shared/src
    ~/stt-proto/venv/bin/python predict_one.py prep ~/sr_accel/weights one.npz
    sudo env PYTHONPATH=$HOME/sr_accel/runtime_src:$HOME/stt-proto/pr/shared/src \
        /usr/local/share/pynq-venv/bin/python3 predict_one.py run \
        ~/sr_accel/full/sr_accel.bit ~/sr_accel/weights one.npz

``prep`` takes decode.ipynb's first test-clean utterance, makes its int8 features and the
integer model's expected scores (``fixed.FABRIC`` formats): every AM frame, and one CM step
(EOS from the zero state, as ``IntCharModel.start``). ``run`` loads the bitstream and both
weight images, runs one AM job per frame and the CM step, and compares.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from shared import vocab

CTRL, JOB, BEAM, LM_CHAR, STATUS = 0x00, 0x04, 0x08, 0x0C, 0x10
LOAD_TARGET, LOAD_ADDR, LOAD_COUNT = 0x14, 0x18, 0x1C
SRC_BEAM = 0x28
JOB_AM, JOB_LM, JOB_CLEAR_AM, JOB_CLEAR_BEAM = 0, 1, 2, 3
N_IN, VOCAB = 123, 30


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


class Accel:
    """The overlay's registers and DMA, with buffers for one frame and one score vector."""

    def __init__(self, bitfile: str, images: Path) -> None:
        """Load the bitstream and stream both weight images in."""
        from pynq import allocate

        from runtime.fabric import Fabric
        from runtime.weights import AM_BASE, cm_base, read_image, repack

        self.fabric = Fabric(bitfile)
        self.regs, self.dma = self.fabric.regs, self.fabric.dma
        am_image, am = read_image(images / "am_fabric.mem")
        cm_image, cm = read_image(images / "cm_fabric.mem")
        for load in repack(am_image, am, AM_BASE) + repack(cm_image, cm, cm_base(am)):
            self.fabric.load(load)
        self.frame = allocate(shape=(31,), dtype=np.uint32)  # 124 bytes, 123 used
        self.scores = allocate(shape=(8,), dtype=np.uint32)

    def run(
        self, job: int, beam: int = 0, char: int = 0, src: int | None = None
    ) -> np.ndarray | None:
        """Start ``job`` and wait for it; AM and LM jobs return their 30 int8 scores.

        An LM job reads slot ``src`` and writes slot ``beam`` (``src`` defaults to ``beam``,
        an in-place step).
        """
        scored = job in (JOB_AM, JOB_LM)
        if scored:
            self.dma.recvchannel.transfer(self.scores)
        self.regs.write(JOB, job)
        self.regs.write(SRC_BEAM, beam if src is None else src)
        self.regs.write(BEAM, beam)
        self.regs.write(LM_CHAR, char)
        self.regs.write(CTRL, 1)
        t = time.perf_counter()
        while not self.regs.read(STATUS) & 0x2:
            if time.perf_counter() - t > 1.0:
                status = self.regs.read(STATUS)
                raise RuntimeError(f"job {job} not done after 1 s, STATUS 0x{status:x}")
        if self.regs.read(STATUS) & 0x4:
            raise RuntimeError(f"job {job}: STATUS error bit set")
        if not scored:
            return None
        self.dma.recvchannel.wait()
        self.scores.invalidate()
        return self.scores.view(np.int8)[:VOCAB].copy()

    def load_frame(self, q: np.ndarray) -> None:
        """Write one int8 feature frame into the frame memory."""
        b = np.zeros(124, np.int8)
        b[:N_IN] = q
        self.frame[:] = b.view(np.uint32)
        self.frame.flush()
        self.regs.write(LOAD_TARGET, 0)
        self.regs.write(LOAD_ADDR, 0)
        self.dma.sendchannel.transfer(self.frame)
        self.dma.sendchannel.wait()
        if self.regs.read(LOAD_COUNT) != N_IN:
            raise RuntimeError(f"frame LOAD_COUNT {self.regs.read(LOAD_COUNT)}, expected {N_IN}")


def run(bitfile: str, images: Path, npz: Path) -> None:
    """Every frame of the saved utterance and one CM step on the fabric, against ``prep``."""
    d = np.load(npz)
    q, want_am, want_cm = d["q"], d["am_scores"], d["cm_scores"]
    t = time.perf_counter()
    accel = Accel(bitfile, images)
    print(f"bitstream and weights loaded in {time.perf_counter() - t:.2f} s")

    accel.run(JOB_CLEAR_AM)
    got = np.empty_like(want_am)
    t = time.perf_counter()
    for k, frame in enumerate(q):
        accel.load_frame(frame)
        got[k] = accel.run(JOB_AM)
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

    accel.run(JOB_CLEAR_BEAM, beam=0)
    cm = accel.run(JOB_LM, beam=0, char=vocab.EOS_IDX)
    print(f"CM step (EOS, zero state): {'exact' if (cm == want_cm).all() else 'MISMATCH'}"
          f"\n    fabric {cm.tolist()}\n    model  {want_cm.tolist()}")
    print(f"STATUS 0x{accel.regs.read(STATUS):08x}")


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
