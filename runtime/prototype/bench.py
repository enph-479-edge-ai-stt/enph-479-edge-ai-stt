"""Run the runtime pipeline on the board's CPU and time each stage.

FLAC -> features -> int8 -> AM (numpy stand-in) -> log-softmax -> beam search
(``shared.decode.beam_cpp``) with the CM (numpy stand-in) and KenLM -> text. Takes the
first ``--n`` of the 500 test-clean utterances ``decode.ipynb`` evaluated on (same seed),
so the error rates are comparable with its run, ``runs/20261102-2019-decode``. The
stand-ins' time is reported apart from the ARM's: on the board the LSTMs run on the fabric.

    python bench.py --n 10

With ``--images DIR`` the models come from the weight images (``am_fabric.mem``,
``cm_fabric.mem``) instead of the ``.npz`` files, in float, or with ``--fabric`` as the
fabric's integer arithmetic computes them (``fixed.py``, formats ``fixed.FABRIC``).
Decode run 4's setup (AM run 3, unpruned trigram):
``--images DIR --gamma 1.0 --arpa 3-gram.arpa.gz``.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import time
from pathlib import Path

import fixed
import jiwer
import numpy as np
import soundfile as sf
from runtime.features import Fbank
from shared import vocab
from shared.decode import beam_cpp
from shared.decode.beam import Config
from standin import AcousticModel, CharModel, log_softmax

# decode.ipynb drew its 100 tuning utterances from dev-clean (2,703 of them) before the
# 500 test-clean ones, from the same Random(0).
DEV_CLEAN_UTTS = 2703
TUNE_UTTS = 100
EVAL_UTTS = 500


def list_utterances(subset_dir: Path) -> list[tuple[Path, str]]:
    """(flac path, transcript) per utterance, sorted, as ``training.am.dataset`` lists them."""
    items = []
    for trans in sorted(subset_dir.glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, _, text = line.partition(" ")
            items.append((trans.parent / f"{utt_id}.flac", text))
    return items


def eval_sample(test: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """decode.ipynb's 500 test-clean utterances, in its order."""
    rng = random.Random(0)
    rng.sample(range(DEV_CLEAN_UTTS), TUNE_UTTS)
    return rng.sample(test, EVAL_UTTS)


def rss_mb() -> float:
    """Peak resident memory of this process so far."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    """Decode ``--n`` utterances, print per-utterance and total timings, save them as JSON."""
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--beam", type=int, default=128)
    p.add_argument("--root", type=Path, default=Path.home() / "stt-proto")
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--arpa", default="3-gram.pruned.1e-7.arpa.gz", help="in <root>/data")
    p.add_argument("--images", type=Path, help="folder with am_fabric.* and cm_fabric.*")
    p.add_argument("--fabric", action="store_true", help="with --images: integer arithmetic")
    p.add_argument(
        "--am-scores",
        type=Path,
        help="with --fabric: decode with the fabric's AM scores (fabric_am.py) and check "
        "every frame against the integer model",
    )
    args = p.parse_args()
    fabric_scores = np.load(args.am_scores) if args.am_scores else None
    cfg = Config(beam=args.beam, alpha=0.5, beta=0.5, gamma=args.gamma)  # decode runs' tuning

    x_scale = 2.0**-5  # the float AM takes the int8 features at their real scale
    if args.images:
        am_ints = fixed.read_ints(args.images / "am_fabric.mem")
        cm_ints = fixed.read_ints(args.images / "cm_fabric.mem")
        m = am_ints["m"]
        fb = Fbank(np.array(m["cmvn_mean"]), np.array(m["cmvn_std"]))
        if args.fabric:
            am = fixed.IntAcousticModel(am_ints, fixed.fabric_fmt(am_ints))
            cm = fixed.IntCharModel(cm_ints, fixed.fabric_fmt(cm_ints))
            x_scale = 1  # the fabric takes the int8 values themselves
        else:
            am = AcousticModel(fixed.float_npz(am_ints))
            cm = CharModel(fixed.float_npz(cm_ints))
    else:
        am_npz = dict(np.load(args.root / "models/am.npz"))
        fb = Fbank(am_npz["cmvn_mean"], am_npz["cmvn_std"])
        am = AcousticModel(am_npz)
        cm = CharModel(dict(np.load(args.root / "models/cm.npz")))
    arpa = args.root / "data" / args.arpa
    t0 = time.perf_counter()
    wlm = beam_cpp.WordLM(str(arpa))
    print(f"KenLM loaded in {time.perf_counter() - t0:.1f} s, peak RSS {rss_mb():.0f} MB")

    items = eval_sample(list_utterances(args.root / "data/LibriSpeech/test-clean"))[: args.n]
    stages = ("read", "features", "int8", "am_standin", "log_softmax", "search_arm", "cm_standin")
    rows, refs, greedy_hyps, beam_hyps = [], [], [], []
    for i, (flac, ref) in enumerate(items):
        t = {}
        t0 = time.perf_counter()
        wav, _ = sf.read(str(flac), dtype="float32")
        t1 = time.perf_counter()
        feats = fb.extract(wav)
        t2 = time.perf_counter()
        q = fb.int8(feats)
        t3 = time.perf_counter()
        logits = am.logits(q * x_scale)
        t4 = time.perf_counter()
        mismatched = None
        if fabric_scores is not None:
            got = fabric_scores[flac.stem].astype(np.int64)
            want = np.rint(logits * 2 ** fixed.FABRIC["score_frac"]).astype(np.int64)
            mismatched = int((got != want).any(-1).sum()) if got.shape == want.shape else -1
            logits = got * 2.0 ** -fixed.FABRIC["score_frac"]
        logp = log_softmax(logits)
        t5 = time.perf_counter()
        cm.standin_s = 0.0
        cm.steps = cm.calls = 0
        hyp = beam_cpp.beam_search(logp, cfg, cm, wlm)[0][0]
        t6 = time.perf_counter()
        t["read"], t["features"], t["int8"] = t1 - t0, t2 - t1, t3 - t2
        t["am_standin"], t["log_softmax"] = t4 - t3, t5 - t4
        t["cm_standin"] = cm.standin_s
        t["search_arm"] = t6 - t5 - cm.standin_s  # includes the CM's log-softmax and KenLM
        dur = len(wav) / 16_000
        greedy = vocab.collapse(logp.argmax(-1))
        rows.append(
            {
                "utt": flac.stem,
                "audio_s": dur,
                "frames": len(q),
                "cm_steps": cm.steps,
                "cm_calls": cm.calls,
                "fabric_mismatched_frames": mismatched,
                **t,
            }
        )
        refs.append(ref)
        greedy_hyps.append(greedy)
        beam_hyps.append(hyp)
        arm = t["features"] + t["int8"] + t["log_softmax"] + t["search_arm"]
        print(
            f"[{i + 1}/{len(items)}] {flac.stem} {dur:.1f} s audio | ARM {arm:.2f} s "
            f"(features {t['features']:.3f}, search {t['search_arm']:.2f}) | stand-ins: "
            f"AM {t['am_standin']:.2f}, CM {t['cm_standin']:.2f} s over {cm.steps} steps "
            f"in {cm.calls} calls",
            flush=True,
        )
        print(f"    ref: {ref}\n    hyp: {hyp}", flush=True)
        if mismatched:
            print(f"    FABRIC MISMATCH: {mismatched} of {len(q)} frames", flush=True)

    audio = sum(r["audio_s"] for r in rows)
    total = {s: sum(r[s] for r in rows) for s in stages}
    print(f"\n{len(rows)} utterances, {audio:.1f} s of audio, beam {cfg.beam}")
    print(f"  peak RSS {rss_mb():.0f} MB")
    for s in stages:
        print(f"  {s:12s} {total[s]:8.2f} s   {total[s] / audio:7.3f}x real time")
    arm = sum(total[s] for s in ("features", "int8", "log_softmax", "search_arm"))
    print(f"  {'ARM total':12s} {arm:8.2f} s   {arm / audio:7.3f}x real time (excl. read)")
    steps = sum(r["cm_steps"] for r in rows)
    calls = sum(r["cm_calls"] for r in rows)
    print(
        f"  CM: {steps / audio:.0f} steps per s of audio, {calls / audio:.0f} batched calls "
        f"per s, {steps / calls:.0f} per call"
    )
    errs = {
        "greedy": (jiwer.wer(refs, greedy_hyps), jiwer.cer(refs, greedy_hyps)),
        "beam + CM + WLM": (jiwer.wer(refs, beam_hyps), jiwer.cer(refs, beam_hyps)),
    }
    for name, (wer, cer) in errs.items():
        print(f"  {name}: WER {wer:.4f} CER {cer:.4f}")
    if fabric_scores is not None:
        bad = [r["utt"] for r in rows if r["fabric_mismatched_frames"]]
        frames = sum(r["frames"] for r in rows)
        print(f"  fabric AM vs integer model: {len(rows) - len(bad)}/{len(rows)} utterances "
              f"exact over {frames} frames; mismatched: {bad[:10]}")

    out = args.root / "runs" / time.strftime("%Y%m%d-%H%M%S")
    out.parent.mkdir(exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "config": {
                    **vars(cfg),
                    "arpa": args.arpa,
                    "images": str(args.images),
                    "fabric": args.fabric and fixed.FABRIC,
                    "am_scores": args.am_scores and str(args.am_scores),
                },
                "rows": rows,
                "total": total,
                "audio_s": audio,
                "errors": errs,
                "peak_rss_mb": rss_mb(),
                "refs": refs,
                "beam": beam_hyps,
                "greedy": greedy_hyps,
            },
            indent=1,
        )
    )
    print(f"saved {out}.json")


if __name__ == "__main__":
    main()
