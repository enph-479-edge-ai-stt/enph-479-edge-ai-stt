r"""Char LM forks on the fabric (SRC_BEAM, 256 slots), checked job by job against the integer model.

Prototype, two steps like ``predict_one.py``:

    ~/stt-proto/venv/bin/python fork_test.py prep ~/sr_accel/weights forks.npz
    sudo /usr/local/share/pynq-venv/bin/python3 fork_test.py run \
        ~/sr_accel/srcdst/sr_accel.bit ~/sr_accel/weights forks.npz

``prep`` makes a list of LM jobs ``(src, dst, char)`` and each job's expected int8 scores:
first a hand-made tree (``THE`` forks into ``THEM`` and ``THEY`` in slots 200 and 255, the
parent is stepped again afterwards, then an in-place step), then random jobs that read any
live slot and write any of the 256. A later job reads what earlier jobs wrote, so a wrong
h or c anywhere shows up in the scores downstream. ``run`` replays the jobs on the fabric.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from shared import vocab

N_SLOTS = 256


def _c(ch: str) -> int:
    return vocab.encode(ch)[0]


def prep(images: Path, out: Path, n_random: int, seed: int) -> None:
    """Jobs and expected scores, saved to ``out``."""
    import fixed

    ints = fixed.read_ints(images / "cm_fabric.mem")
    fmt = fixed.fabric_fmt(ints)
    cm = fixed.IntCharModel(ints, fmt)
    zero = np.zeros((cm.n_layers, cm.hidden), np.int64)
    states = {0: (zero, zero)}  # slot 0 is cleared on the fabric first

    jobs = [(0, 0, vocab.EOS_IDX), (0, 1, _c("T")), (1, 2, _c("H")), (2, 3, _c("E")),
            (3, 200, _c("M")), (3, 255, _c("Y")),  # fork: two children of THE
            (3, 4, _c(" ")),  # the parent again, after both children were written
            (200, 201, _c(" ")), (255, 254, _c(" ")), (3, 3, _c("N"))]  # in place: THEN
    hand = len(jobs)
    rng = np.random.default_rng(seed)
    chars = [i for i in range(vocab.VOCAB_SIZE) if i != vocab.BLANK_IDX]
    scores = []
    for k in range(hand + n_random):
        if k < hand:
            src, dst, ch = jobs[k]
        else:  # any live parent, any slot (often a live one, or src itself), any char
            src = int(rng.choice(sorted(states)))
            dst, ch = int(rng.integers(N_SLOTS)), int(rng.choice(chars))
            jobs.append((src, dst, ch))
        h, c = states[src]
        logits, h_new, c_new = cm._forward([ch], h[:, None], c[:, None])
        states[dst] = (h_new[:, 0], c_new[:, 0])
        scores.append(np.rint(logits[0] * 2**fmt.score_frac).astype(np.int8))

    np.savez(out, jobs=np.array(jobs, np.int64), scores=np.array(scores), hand=hand)
    print(f"{len(jobs)} jobs ({hand} hand-made), {len(states)} slots written, "
          f"{len({d for _, d, _ in jobs})} distinct destinations")


def run(bitfile: str, images: Path, npz: Path) -> None:
    """Replay the jobs on the fabric and compare every score vector."""
    from runtime.fabric import Fabric

    d = np.load(npz)
    jobs, want, hand = d["jobs"], d["scores"], int(d["hand"])
    fabric = Fabric(bitfile)
    fabric.load_images(images / "am_fabric.mem", images / "cm_fabric.mem")
    fabric.clear_slot(0)
    got = np.empty_like(want)
    t = time.perf_counter()
    for k, (src, dst, ch) in enumerate(jobs):
        got[k] = fabric.lm_step(int(src), int(dst), int(ch))
    dt = time.perf_counter() - t

    exact = (got == want).all(-1)
    names = ["EOS from zero", "T", "TH", "THE", "fork THEM -> 200", "fork THEY -> 255",
             "parent THE again -> 4", "THEM_ -> 201", "THEY_ -> 254", "in place THEN -> 3"]
    for k, name in enumerate(names[:hand]):
        print(f"  {name:24s} src {jobs[k][0]:3d} dst {jobs[k][1]:3d}  "
              f"{'exact' if exact[k] else 'MISMATCH'}")
    print(f"random jobs: {exact[hand:].sum()}/{len(jobs) - hand} exact")
    ms = dt / len(jobs) * 1e3
    print(f"all: {exact.sum()}/{len(jobs)} exact, {ms:.3f} ms per job incl. Python")
    if not exact.all():
        k = int(np.argmin(exact))
        print(f"first mismatch, job {k} {tuple(jobs[k])}:\n"
              f"    fabric {got[k].tolist()}\n    model  {want[k].tolist()}")


def main() -> None:
    """``prep`` or ``run``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep")
    s.add_argument("images", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--random", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("run")
    s.add_argument("bitfile")
    s.add_argument("images", type=Path)
    s.add_argument("npz", type=Path)
    args = p.parse_args()
    if args.cmd == "prep":
        prep(args.images, args.out, args.random, args.seed)
    else:
        run(args.bitfile, args.images, args.npz)


if __name__ == "__main__":
    sys.exit(main())
