r"""The CM on the fabric, checked score for score against the integer model.

Prototype. Feeds the transcripts of decode.ipynb's test-clean utterances through the
fabric's char-LM a character at a time, each sentence ``EOS`` then its characters, as
training scores a sentence. Every step's 30 int8 scores must equal the integer model's
(``fixed.IntCharModel``, formats ``fixed.FABRIC``). Two steps, as ``fabric_am.py`` (the
fabric needs root and PYNQ's venv):

    # ~/stt-proto/pr/runtime/prototype, PYTHONPATH=~/stt-proto/pr/shared/src
    ~/stt-proto/venv/bin/python fabric_cm.py prep ~/sr_accel/weights cm_want.npz --n 500
    sudo env PYTHONPATH=$HOME/sr_accel/runtime_src:$HOME/stt-proto/pr/shared/src \
        /usr/local/share/pynq-venv/bin/python3 fabric_cm.py run \
        ~/sr_accel/epu50/sr_accel.bit ~/sr_accel/weights cm_want.npz cm_got.npz

``run`` checks what the beam search will lean on, not only the arithmetic: sentences go
through ``--group`` beam slots at once, interleaved a character at a time, each sentence in
its own slot (sentence k in slot k mod 128, so every slot is used), and an AM job runs
after every ``--am-every`` CM steps, so the model switch and the AM's context are
exercised between CM steps. Any slot a CM step disturbs, or an AM job, shows up as a
mismatch in another sentence.

``show`` prints the fabric's guesses from ``run``'s saved scores (no root needed): for a
sentence, its likeliest next characters at each step against the one that came next.

    ~/stt-proto/venv/bin/python fabric_cm.py show cm_want.npz cm_got.npz --sentences 0 3
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from shared import vocab

N_SLOTS = 128  # the fabric's CM contexts, fsm_controller.v's N_BEAMS


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def bits_per_char(scores: np.ndarray, inputs: np.ndarray, offsets: np.ndarray) -> float:
    """Next-character cross-entropy in bits, as training's ``bits_per_char``: each
    sentence's inputs are ``EOS`` then its characters, the targets its characters then
    ``EOS``. ``scores`` are int8 at 2**-2."""
    logp = _log_softmax(scores.astype(np.float64) / 4)
    targets = np.empty_like(inputs)
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        targets[a : b - 1] = inputs[a + 1 : b]
        targets[b - 1] = vocab.EOS_IDX
    return float(-logp[np.arange(len(inputs)), targets].mean() / math.log(2))


def prep(images: Path, out: Path, n: int) -> None:
    """The first ``n`` transcripts' input characters and the integer model's scores."""
    import fixed
    from bench import eval_sample, list_utterances

    ints = fixed.read_ints(images / "cm_fabric.mem")
    fmt = fixed.fabric_fmt(ints)
    cm = fixed.IntCharModel(ints, fmt)
    test = list_utterances(Path.home() / "stt-proto/data/LibriSpeech/test-clean")
    sents = [[vocab.EOS_IDX, *vocab.encode(ref)] for _, ref in eval_sample(test)[:n]]

    # All sentences at once, a step at a time: row k of h, c is sentence k.
    lens = np.array([len(s) for s in sents])
    h = np.zeros((cm.n_layers, len(sents), cm.hidden), np.int64)
    c = np.zeros_like(h)
    want = [np.empty((n_k, vocab.VOCAB_SIZE), np.int8) for n_k in lens]
    for t in range(lens.max()):
        live = np.flatnonzero(lens > t)
        logits, h[:, live], c[:, live] = cm._forward(
            [sents[k][t] for k in live], h[:, live], c[:, live]
        )
        scores = np.rint(logits * 2**fmt.score_frac).astype(np.int8)
        for row, k in enumerate(live):
            want[k][t] = scores[row]

    inputs = np.concatenate([np.array(s, np.int64) for s in sents])
    offsets = np.concatenate([[0], np.cumsum(lens)])
    scores = np.concatenate(want)
    np.savez(out, inputs=inputs, offsets=offsets, scores=scores)
    print(f"{len(sents)} sentences, {len(inputs)} CM steps -> {out}")
    print(f"integer model: {bits_per_char(scores, inputs, offsets):.4f} bits per character")


def run(
    bitfile: str, images: Path, want_npz: Path, out: Path, group: int, am_every: int
) -> None:
    """Every step of the prepared sentences on the fabric, compared with ``prep``'s scores."""
    from predict_one import JOB_AM, JOB_CLEAR_AM, JOB_CLEAR_BEAM, JOB_LM, N_IN, Accel

    if not 1 <= group <= N_SLOTS:
        raise ValueError(f"--group must be 1..{N_SLOTS}: each sentence in a group needs a slot")
    d = np.load(want_npz)
    inputs, offsets, want = d["inputs"], d["offsets"], d["scores"]
    n_sents = len(offsets) - 1
    t = time.perf_counter()
    accel = Accel(bitfile, images)
    print(f"bitstream and weights loaded in {time.perf_counter() - t:.2f} s")

    # The AM jobs in between run on one fixed frame; their scores aren't checked here.
    accel.run(JOB_CLEAR_AM)
    accel.load_frame(np.arange(N_IN, dtype=np.int64).astype(np.int8))
    got = np.empty_like(want)
    cm_steps = am_jobs = 0
    t = time.perf_counter()
    for g in range(0, n_sents, group):
        ks = range(g, min(g + group, n_sents))
        for k in ks:
            accel.run(JOB_CLEAR_BEAM, beam=k % N_SLOTS)
        for step in range(max(offsets[k + 1] - offsets[k] for k in ks)):
            for k in ks:
                i = offsets[k] + step
                if i >= offsets[k + 1]:
                    continue
                got[i] = accel.run(JOB_LM, beam=k % N_SLOTS, char=int(inputs[i]))
                cm_steps += 1
                if am_every and cm_steps % am_every == 0:
                    accel.run(JOB_AM)
                    am_jobs += 1
        if (g // group + 1) % 10 == 0:
            print(f"[{ks.stop}/{n_sents}] {cm_steps} CM steps, {time.perf_counter() - t:.0f} s",
                  flush=True)
    dt = time.perf_counter() - t
    np.savez(out, scores=got)

    exact = (got == want).all(-1)
    print(f"{cm_steps} CM steps and {am_jobs} AM jobs in {dt:.1f} s "
          f"({dt / (cm_steps + am_jobs) * 1e3:.2f} ms per job incl. Python) -> {out}")
    print(f"CM scores: {exact.sum()}/{len(exact)} steps exact, "
          f"max |diff| {np.abs(got.astype(int) - want).max()}")
    bad = np.flatnonzero(~exact)
    if len(bad):
        i = int(bad[0])
        k = int(np.searchsorted(offsets, i, side="right") - 1)
        sent_bad = np.unique(np.searchsorted(offsets, bad, side="right") - 1)
        print(f"  {len(sent_bad)} sentences affected; first mismatch: sentence {k} (slot "
              f"{k % N_SLOTS}), step {i - offsets[k]}, input {vocab.CHARS[inputs[i]]!r}")
        print(f"    fabric {got[i].tolist()}\n    model  {want[i].tolist()}")
    print(f"bits per character: fabric {bits_per_char(got, inputs, offsets):.4f}, "
          f"integer model {bits_per_char(want, inputs, offsets):.4f}")
    print(f"STATUS 0x{accel.regs.read(0x10):08x}")


def _sym(i: int) -> str:
    """A vocab index as one printable character: EOS as ``|``, the blank as ``_``."""
    return {vocab.EOS_IDX: "|", vocab.BLANK_IDX: "_"}.get(int(i), vocab.CHARS[int(i)])


def show(want_npz: Path, got_npz: Path, sentences: list[int], top: int) -> None:
    """The fabric's guesses: per step, its ``top`` likeliest next characters given the
    sentence so far, against the one that came next. Reads ``run``'s saved scores, so it
    needs no fabric and no root."""
    d = np.load(want_npz)
    inputs, offsets = d["inputs"], d["offsets"]
    got = np.load(got_npz)["scores"]
    probs = np.exp(_log_softmax(got.astype(np.float64) / 4))
    targets = np.empty_like(inputs)
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        targets[a : b - 1] = inputs[a + 1 : b]
        targets[b - 1] = vocab.EOS_IDX
    right = probs.argmax(-1) == targets
    print(f"top-1 next-character accuracy over all {len(inputs)} steps: {right.mean():.1%}")

    for k in sentences:
        a, b = offsets[k], offsets[k + 1]
        actual = "".join(_sym(i) for i in targets[a:b])
        guess = "".join(_sym(i) for i in probs[a:b].argmax(-1))
        marks = "".join(" " if r else "^" for r in right[a:b])
        print(f"\nsentence {k} (slot {k % N_SLOTS}), {right[a:b].mean():.0%} guessed right")
        print(f"  actual: {actual}\n  guess:  {guess}\n          {marks}")
        print(f"  {'so far':>24}  next  {'top guesses':<{top * 11}}")
        for i in range(a, b):
            seen = "".join(_sym(x) for x in inputs[a + 1 : i + 1])[-24:]
            best = np.argsort(-probs[i])[:top]
            guesses = "  ".join(f"{_sym(c)!r} {probs[i, c]:4.0%}" for c in best)
            ok = "" if right[i] else f"   ({_sym(targets[i])!r} at {probs[i, targets[i]]:.0%})"
            print(f"  {seen:>24}  {_sym(targets[i])!r:4}  {guesses}{ok}")


def main() -> None:
    """``prep``, ``run`` or ``show``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prep")
    s.add_argument("images", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--n", type=int, default=500)
    s = sub.add_parser("run")
    s.add_argument("bitfile")
    s.add_argument("images", type=Path)
    s.add_argument("want", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--group", type=int, default=8, help="sentences interleaved at once")
    s.add_argument("--am-every", type=int, default=4, help="CM steps per AM job; 0 for none")
    s = sub.add_parser("show")
    s.add_argument("want", type=Path)
    s.add_argument("got", type=Path)
    s.add_argument("--sentences", type=int, nargs="+", default=[0])
    s.add_argument("--top", type=int, default=3)
    args = p.parse_args()
    if args.cmd == "prep":
        prep(args.images, args.out, args.n)
    elif args.cmd == "show":
        show(args.want, args.got, args.sentences, args.top)
    else:
        run(args.bitfile, args.images, args.want, args.out, args.group, args.am_every)


if __name__ == "__main__":
    main()
