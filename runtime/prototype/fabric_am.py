r"""The AM on the fabric for decode.ipynb's test-clean utterances: int8 scores for every frame.

Prototype. Two steps, as ``predict_one.py`` (the fabric needs root and PYNQ's venv):

    # ~/stt-proto/pr/runtime/prototype, PYTHONPATH=~/stt-proto/pr/shared/src
    ~/stt-proto/venv/bin/python fabric_am.py feats ~/sr_accel/weights feats.npz --n 500
    sudo env PYTHONPATH=$HOME/sr_accel/runtime_src:$HOME/stt-proto/pr/shared/src \
        /usr/local/share/pynq-venv/bin/python3 fabric_am.py run \
        ~/sr_accel/epu50/sr_accel.bit ~/sr_accel/weights feats.npz scores.npz

``run`` clears the AM context before each utterance. Both files are keyed by utterance id.
``bench.py --fabric --am-scores scores.npz`` then decodes with these scores and checks each
frame against the integer model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def feats(images: Path, out: Path, n: int) -> None:
    """The int8 features of the first ``n`` utterances, as ``bench.py`` makes them."""
    import fixed
    import soundfile as sf
    from bench import eval_sample, list_utterances
    from features import Fbank

    m = fixed.read_ints(images / "am_fabric.mem")["m"]
    fb = Fbank(np.array(m["cmvn_mean"]), np.array(m["cmvn_std"]))
    test = list_utterances(Path.home() / "stt-proto/data/LibriSpeech/test-clean")
    qs = {}
    for flac, _ in eval_sample(test)[:n]:
        wav, _ = sf.read(str(flac), dtype="float32")
        qs[flac.stem] = fb.int8(fb.extract(wav))
    np.savez(out, **qs)
    print(f"{len(qs)} utterances, {sum(len(q) for q in qs.values())} frames -> {out}")


def run(bitfile: str, images: Path, feats_npz: Path, out: Path) -> None:
    """One AM job per frame of every utterance; the scores saved to ``out``."""
    from predict_one import JOB_AM, JOB_CLEAR_AM, Accel

    qs = np.load(feats_npz)
    accel = Accel(bitfile, images)
    scores, frames = {}, 0
    t = time.perf_counter()
    for k, utt in enumerate(qs.files):
        q = qs[utt]
        accel.run(JOB_CLEAR_AM)
        s = np.empty((len(q), 30), np.int8)
        for i, frame in enumerate(q):
            accel.load_frame(frame)
            s[i] = accel.run(JOB_AM)
        scores[utt] = s
        frames += len(q)
        if (k + 1) % 50 == 0:
            print(f"[{k + 1}/{len(qs.files)}] {frames} frames, {time.perf_counter() - t:.0f} s",
                  flush=True)
    dt = time.perf_counter() - t
    np.savez(out, **scores)
    print(f"{len(scores)} utterances, {frames} frames in {dt:.1f} s "
          f"({dt / frames * 1e3:.2f} ms per frame, {frames / 100 / dt:.1f}x real time) -> {out}")
    print(f"STATUS 0x{accel.regs.read(0x10):08x}")


def main() -> None:
    """``feats`` or ``run``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("feats")
    s.add_argument("images", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--n", type=int, default=500)
    s = sub.add_parser("run")
    s.add_argument("bitfile")
    s.add_argument("images", type=Path)
    s.add_argument("feats", type=Path)
    s.add_argument("out", type=Path)
    args = p.parse_args()
    if args.cmd == "feats":
        feats(args.images, args.out, args.n)
    else:
        run(args.bitfile, args.images, args.feats, args.out)


if __name__ == "__main__":
    main()
