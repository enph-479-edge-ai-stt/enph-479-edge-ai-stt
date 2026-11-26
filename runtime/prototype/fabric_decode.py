r"""The 500-clip check of ``runtime.transcribe``: every transcript against a ``bench.py`` run.

Prototype, two steps. ``run`` transcribes decode.ipynb's test-clean utterances from their
FLACs with ``runtime.transcribe.Transcriber`` (root, PYNQ's venv). ``score`` runs in the
prototype venv (jiwer) and compares with a ``bench.py`` run, by default decode run 4's
setup (AM run 3, unpruned trigram, gamma 1.0). From this folder:

    sudo /usr/local/share/pynq-venv/bin/python3 fabric_decode.py run \
        ~/sr_accel/srcdst/sr_accel.bit ~/sr_accel/weights ~/stt-proto/data/3-gram.arpa.gz \
        ~/stt-proto/data/LibriSpeech out.json --am-scores scores500-am.npz
    ~/stt-proto/venv/bin/python fabric_decode.py score out.json ref.json

Under sudo ``~`` is root's home, so give the paths in full.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def run(args: argparse.Namespace) -> None:
    """Transcribe the utterances on the fabric, save the transcripts and times."""
    from bench import eval_sample, list_utterances

    from runtime.transcribe import Transcriber

    items = eval_sample(list_utterances(args.librispeech / "test-clean"))[: args.n]
    want_am = np.load(args.am_scores) if args.am_scores else None
    t0 = time.perf_counter()
    tr = Transcriber(args.bitfile, args.images, args.arpa)
    print(f"fabric, weights and word LM loaded in {time.perf_counter() - t0:.1f} s", flush=True)

    rows = []
    for i, (flac, _) in enumerate(items):
        utt = flac.stem
        r = tr.transcribe_file(flac)
        s = r.seconds
        am_bad = None if want_am is None else int((r.scores != want_am[utt]).any(-1).sum())
        rows.append({"utt": utt, "frames": len(r.scores), "features_s": s["features"],
                     "am_s": s["am"], "cm_s": s["cm"], "search_s": s["cm"] + s["search"],
                     "cm_steps": r.cm_steps, "peak_slots": r.peak_slots,
                     "am_mismatched_frames": am_bad, "hyp": r.text, "greedy": r.greedy})
        print(f"[{i + 1}/{len(items)}] {utt} {len(r.scores) / 100:.1f} s | features "
              f"{s['features']:.2f}, AM {s['am']:.2f}, CM {s['cm']:.2f} s over {r.cm_steps} "
              f"steps (peak {r.peak_slots} slots), search on ARM {s['search']:.2f} s | {r.text}",
              flush=True)
        if am_bad:
            print(f"    AM MISMATCH: {am_bad} of {len(r.scores)} frames", flush=True)
        if tr.cm.pool.live:
            print(f"    {tr.cm.pool.live} CM slots still live after the search", flush=True)
        if (i + 1) % 25 == 0 or i + 1 == len(items):
            args.out.write_text(json.dumps({"config": {**asdict(tr.cfg), "arpa": str(args.arpa)},
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
    feats = sum(r.get("features_s", 0.0) for r in rows)
    am = sum(r["am_s"] for r in rows)
    cm = sum(r["cm_s"] for r in rows)
    arm = sum(r["search_s"] for r in rows) - cm
    steps = sum(r["cm_steps"] for r in rows)
    for name, s in (("features (ARM)", feats), ("AM on fabric", am), ("CM on fabric", cm),
                    ("search on ARM", arm), ("total", feats + am + cm + arm)):
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
    s.add_argument("arpa", type=Path, help="e.g. 3-gram.arpa.gz")
    s.add_argument("librispeech", type=Path, help="the folder holding test-clean/")
    s.add_argument("out", type=Path)
    s.add_argument("--am-scores", type=Path, help="check every AM frame against these")
    s.add_argument("--n", type=int, default=500)
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
