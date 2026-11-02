"""Check the C++ search decodes real utterances exactly as the Python search does.

``shared/tests/test_beam_cpp.py`` holds it to the golden case on synthetic input; this
runs the first ``--n`` of ``bench.py``'s utterances through both, with the numpy CM and
KenLM, and compares the whole N-best lists.

    python check_cpp.py --n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from bench import eval_sample, list_utterances
from features import Fbank
from standin import AcousticModel, CharModel, log_softmax

from shared.decode import beam_cpp
from shared.decode.beam import Config, beam_search
from shared.decode.wlm import KenLM


def main() -> None:
    """Print, per utterance, whether the two N-best lists match and the largest score gap."""
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--root", type=Path, default=Path.home() / "stt-proto")
    args = p.parse_args()

    am_npz = dict(np.load(args.root / "models/am.npz"))
    fb = Fbank(am_npz["cmvn_mean"], am_npz["cmvn_std"])
    am = AcousticModel(am_npz)
    cm = CharModel(dict(np.load(args.root / "models/cm.npz")))
    arpa = args.root / "data/3-gram.pruned.1e-7.arpa.gz"
    wlm_py, wlm_cpp = KenLM(arpa), beam_cpp.WordLM(str(arpa))
    cfg = Config(beam=128, alpha=0.5, beta=0.5, gamma=2.0)
    items = eval_sample(list_utterances(args.root / "data/LibriSpeech/test-clean"))[: args.n]
    for flac, _ in items:
        wav, _ = sf.read(str(flac), dtype="float32")
        logp = log_softmax(am.logits(fb.int8(fb.extract(wav)) * 2.0**-5))
        py = beam_search(logp, cfg, cm, wlm_py)
        cpp = beam_cpp.beam_search(logp, cfg, cm, wlm_cpp)
        same = [t for t, _ in py] == [t for t, _ in cpp]
        gap = max(abs(a - b) for (_, a), (_, b) in zip(py, cpp, strict=False))
        print(
            f"{flac.stem}: {len(py)} and {len(cpp)} hypotheses, N-best "
            f"{'same' if same else 'DIFFERENT'}, largest score difference {gap:.1e}"
        )


if __name__ == "__main__":
    main()
