r"""The whole decode with both LSTMs on the fabric: AM, CM and the beam search on the ARM.

Prototype, two steps. ``run`` needs root and PYNQ's venv, and takes the int8 features
``fabric_am.py feats`` saved, so it needs no audio or feature code. ``score`` runs in the
prototype venv and compares with a ``bench.py`` run (default: decode run 4's setup, AM run
3, unpruned trigram, gamma 1.0):

    sudo env PYTHONPATH=$HOME/sr_accel/runtime_src:$HOME/stt-proto/pr/shared/src \
        /usr/local/share/pynq-venv/bin/python3 fabric_decode.py run \
        ~/sr_accel/srcdst/sr_accel.bit ~/sr_accel/weights feats500.npz out.json \
        --arpa ~/stt-proto/data/3-gram.arpa.gz
    ~/stt-proto/venv/bin/python fabric_decode.py score out.json ref.json

``FabricCharModel`` is the CM behind the search's ``CharLM`` interface. A state is a slot
handle that puts its slot back on the free list when the search drops it. When the search
steps the CM, the live states are the last beam's (up to 128) and the new children (up to
128), so the 256 slots always suffice.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from shared import vocab
from standin import log_softmax

SCORE_STEP = 0.25  # the fabric's int8 scores are logits * 4 (fixed.FABRIC["score_frac"])


class _Slot:
    """A CM state: slot ``n`` of the context memory, freed when the search drops it."""

    __slots__ = ("free", "n")

    def __init__(self, n: int, free: list[int]) -> None:
        self.n, self.free = n, free

    def __del__(self) -> None:
        self.free.append(self.n)


class FabricCharModel:
    """The CM on the fabric, with ``fixed.IntCharModel``'s ``start`` and ``step``."""

    def __init__(self, accel, n_slots: int = 256) -> None:
        """Every slot starts free."""
        self.accel = accel
        self.free = list(range(n_slots))
        self.n_slots = n_slots
        self.fabric_s = 0.0
        self.steps = self.calls = self.peak = 0

    def _job(self, src: int, char: int) -> tuple[_Slot, np.ndarray]:
        from predict_one import JOB_LM

        if not self.free:
            raise RuntimeError(f"all {self.n_slots} CM slots are live")
        dst = _Slot(self.free.pop(), self.free)
        self.peak = max(self.peak, self.n_slots - len(self.free))
        scores = self.accel.run(JOB_LM, beam=dst.n, char=char, src=src)
        return dst, scores

    def start(self):
        """Feed EOS from the zero state, as at a sentence boundary in training."""
        from predict_one import JOB_CLEAR_BEAM

        slot = self.free[-1]
        self.accel.run(JOB_CLEAR_BEAM, beam=slot)
        state, scores = self._job(slot, vocab.EOS_IDX)  # pops that slot: an in-place step
        return state, log_softmax(scores * SCORE_STEP)

    def step(self, states, chars):
        """Feed ``chars[n]`` to ``states[n]``, one fabric job each."""
        t0 = time.perf_counter()
        new, scores = [], np.empty((len(chars), vocab.VOCAB_SIZE), np.int8)
        for k, (s, ch) in enumerate(zip(states, chars, strict=True)):
            state, scores[k] = self._job(s.n, ch)
            new.append(state)
        self.fabric_s += time.perf_counter() - t0
        self.steps += len(chars)
        self.calls += 1
        return new, log_softmax(scores * SCORE_STEP)


def run(args: argparse.Namespace) -> None:
    """Decode the saved utterances with the AM and CM on the fabric, save hypotheses and times."""
    from predict_one import JOB_AM, JOB_CLEAR_AM, Accel
    from shared.decode import beam_cpp
    from shared.decode.beam import Config

    feats = np.load(args.feats)
    utts = list(feats.files)[: args.n]
    want_am = np.load(args.am_scores) if args.am_scores else None
    cfg = Config(beam=args.beam, alpha=0.5, beta=0.5, gamma=args.gamma)
    t0 = time.perf_counter()
    accel = Accel(args.bitfile, args.images)
    cm = FabricCharModel(accel)
    wlm = beam_cpp.WordLM(str(args.arpa))
    print(f"bitstream, weights and KenLM loaded in {time.perf_counter() - t0:.1f} s", flush=True)

    rows = []
    for i, utt in enumerate(utts):
        q = feats[utt]
        t0 = time.perf_counter()
        accel.run(JOB_CLEAR_AM)
        scores = np.empty((len(q), vocab.VOCAB_SIZE), np.int8)
        for k, frame in enumerate(q):
            accel.load_frame(frame)
            scores[k] = accel.run(JOB_AM)
        t1 = time.perf_counter()
        logp = log_softmax(scores * SCORE_STEP)
        cm.fabric_s, cm.steps, cm.calls, cm.peak = 0.0, 0, 0, 0
        hyp = beam_cpp.beam_search(logp, cfg, cm, wlm)[0][0]
        t2 = time.perf_counter()
        am_bad = None if want_am is None else int((scores != want_am[utt]).any(-1).sum())
        rows.append({"utt": utt, "frames": len(q), "am_s": t1 - t0, "search_s": t2 - t1,
                     "cm_s": cm.fabric_s, "cm_steps": cm.steps, "cm_calls": cm.calls,
                     "peak_slots": cm.peak, "am_mismatched_frames": am_bad, "hyp": hyp,
                     "greedy": vocab.collapse(scores.argmax(-1))})
        print(f"[{i + 1}/{len(utts)}] {utt} {len(q) / 100:.1f} s | AM {t1 - t0:.2f} s, "
              f"CM {cm.fabric_s:.2f} s over {cm.steps} steps (peak {cm.peak} slots), "
              f"search on ARM {t2 - t1 - cm.fabric_s:.2f} s | {hyp}", flush=True)
        if am_bad:
            print(f"    AM MISMATCH: {am_bad} of {len(q)} frames", flush=True)
        if len(cm.free) != cm.n_slots:
            print(f"    {cm.n_slots - len(cm.free)} CM slots still live after the search")
        if (i + 1) % 25 == 0 or i + 1 == len(utts):
            args.out.write_text(json.dumps({"config": {**vars(cfg), "arpa": str(args.arpa)},
                                            "rows": rows}, indent=1))
    print(f"saved {args.out}")


def score(args: argparse.Namespace) -> None:
    """WER and CER of a ``run``, its transcripts against a ``bench.py`` run, and the times."""
    import jiwer

    out, ref = json.loads(args.out.read_text()), json.loads(args.ref.read_text())
    ref_rows = {r["utt"]: (r, text, hyp) for r, text, hyp in
                zip(ref["rows"], ref["refs"], ref["beam"], strict=True)}
    rows = out["rows"]
    texts = [ref_rows[r["utt"]][1] for r in rows]
    hyps = [r["hyp"] for r in rows]
    same = [r["hyp"] == ref_rows[r["utt"]][2] for r in rows]
    audio = sum(ref_rows[r["utt"]][0]["audio_s"] for r in rows)
    print(f"{len(rows)} utterances, {audio:.1f} s of audio")
    print(f"  beam + CM + WLM: WER {jiwer.wer(texts, hyps):.4f} CER {jiwer.cer(texts, hyps):.4f}")
    greedy = [r["greedy"] for r in rows]
    print(f"  greedy:          WER {jiwer.wer(texts, greedy):.4f} "
          f"CER {jiwer.cer(texts, greedy):.4f}")
    print(f"  transcripts equal to {args.ref.name}: {sum(same)}/{len(rows)}")
    for r, ok in zip(rows, same, strict=True):
        if not ok:
            want = ref_rows[r["utt"]][2]
            print(f"    {r['utt']}\n      fabric: {r['hyp']}\n      ref:    {want}")
    bad_am = [r["utt"] for r in rows if r["am_mismatched_frames"]]
    if rows[0]["am_mismatched_frames"] is not None:
        print(f"  AM scores equal to the saved ones: {len(rows) - len(bad_am)}/{len(rows)}")
    am = sum(r["am_s"] for r in rows)
    cm = sum(r["cm_s"] for r in rows)
    arm = sum(r["search_s"] for r in rows) - cm
    steps = sum(r["cm_steps"] for r in rows)
    for name, s in (("AM on fabric", am), ("CM on fabric", cm), ("search on ARM", arm),
                    ("total", am + cm + arm)):
        print(f"  {name:14s} {s:8.1f} s  {s / audio:6.3f}x real time (incl. Python)")
    print(f"  CM: {steps / audio:.0f} steps per s of audio, {cm / steps * 1e3:.3f} ms per step, "
          f"peak {max(r['peak_slots'] for r in rows)} slots")


def main() -> None:
    """``run`` or ``score``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("bitfile")
    s.add_argument("images", type=Path)
    s.add_argument("feats", type=Path, help="fabric_am.py feats output")
    s.add_argument("out", type=Path)
    s.add_argument("--am-scores", type=Path, help="check every AM frame against these")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--beam", type=int, default=128)
    s.add_argument("--gamma", type=float, default=1.0)
    s.add_argument("--arpa", type=Path, required=True, help="e.g. 3-gram.arpa.gz (sudo: no ~)")
    s = sub.add_parser("score")
    s.add_argument("out", type=Path)
    s.add_argument("ref", type=Path, help="a bench.py run's JSON")
    args = p.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        score(args)


if __name__ == "__main__":
    sys.exit(main())
