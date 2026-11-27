r"""Audio files to text on the KV260, with both LSTMs on the fabric (board only, as root).

FLAC -> features -> int8 (``features.py``) -> AM on the fabric -> log-softmax -> beam
search on the ARM (``shared.decode.beam_cpp``) with the char LM on the fabric and the
word LM -> text. From the repo on the board, in PYNQ's venv (see ``runtime/README.md``):

    sudo /usr/local/share/pynq-venv/bin/python3 -m runtime.transcribe sr_accel.bit \
        weights/ 3-gram.arpa.gz clip.flac [clip2.flac ...] [--json out.json]

``weights/`` holds ``am_fabric.mem`` and ``cm_fabric.mem`` with their manifests. The
search settings default to decode run 4's (``DECODE_RUN_4``), which the unpruned trigram
was tuned with.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from shared import vocab
from shared.decode import beam_cpp
from shared.decode.beam import Config

from runtime.fabric import Fabric
from runtime.features import SAMPLE_RATE, Fbank
from runtime.models import FabricAcousticModel, FabricCharModel, log_softmax, logits

DECODE_RUN_4 = Config(beam=128, alpha=0.5, beta=0.5, gamma=1.0)


@dataclass
class Result:
    """One clip's transcript, with what the checks and the timing need."""

    text: str
    greedy: str  # the AM's best path, collapsed, without the search
    scores: np.ndarray = field(repr=False)  # the AM's int8 scores [T, V]
    seconds: dict[str, float]  # features, am (fabric), cm (fabric), search (ARM)
    cm_steps: int
    peak_slots: int


class Transcriber:
    """The bitstream, both weight images and the word LM, loaded once."""

    def __init__(
        self, bitfile: str, images: Path, arpa: str | Path, cfg: Config = DECODE_RUN_4
    ) -> None:
        """Load the fabric and its weights, then the word LM (minutes for the unpruned one)."""
        self.fabric = Fabric(bitfile)
        am, _ = self.fabric.load_images(images / "am_fabric.mem", images / "cm_fabric.mem")
        self.fbank = Fbank(np.array(am["cmvn_mean"]), np.array(am["cmvn_std"]))
        self.am = FabricAcousticModel(self.fabric)
        self.cm = FabricCharModel(self.fabric)
        self.wlm = beam_cpp.WordLM(str(arpa))
        self.cfg = cfg

    def transcribe(self, wav: np.ndarray) -> Result:
        """Decode one clip of 16 kHz mono audio."""
        t0 = time.perf_counter()
        q = self.fbank.int8(self.fbank.extract(wav))
        t1 = time.perf_counter()
        scores = self.am.scores(q)
        t2 = time.perf_counter()
        logp = log_softmax(logits(scores))
        self.cm.seconds, self.cm.steps, self.cm.pool.peak = 0.0, 0, 0
        text = beam_cpp.beam_search(logp, self.cfg, self.cm, self.wlm)[0][0]
        t3 = time.perf_counter()
        seconds = {
            "features": t1 - t0,
            "am": t2 - t1,
            "cm": self.cm.seconds,
            "search": t3 - t2 - self.cm.seconds,
        }
        greedy = vocab.collapse(scores.argmax(-1))
        return Result(text, greedy, scores, seconds, self.cm.steps, self.cm.pool.peak)

    def transcribe_file(self, path: str | Path) -> Result:
        """Decode one audio file."""
        wav, sr = sf.read(str(path), dtype="float32")
        if sr != SAMPLE_RATE or wav.ndim != 1:
            raise ValueError(f"{path}: need {SAMPLE_RATE} Hz mono, got {sr} Hz, shape {wav.shape}")
        return self.transcribe(wav)


def main() -> None:
    """Transcribe each file, print the text and times, optionally save them as JSON."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("bitfile")
    p.add_argument("images", type=Path, help="folder with am_fabric.* and cm_fabric.*")
    p.add_argument("arpa", type=Path, help="the word LM, e.g. 3-gram.arpa.gz")
    p.add_argument("audio", type=Path, nargs="+")
    p.add_argument("--json", type=Path, help="save each clip's text and times here")
    args = p.parse_args()

    t = time.perf_counter()
    tr = Transcriber(args.bitfile, args.images, args.arpa)
    print(f"fabric, weights and word LM loaded in {time.perf_counter() - t:.1f} s", flush=True)
    rows = []
    for path in args.audio:
        r = tr.transcribe_file(path)
        audio_s = sf.info(str(path)).duration
        s = r.seconds
        print(
            f"{path.name} {audio_s:.1f} s | features {s['features']:.2f}, AM {s['am']:.2f}, "
            f"CM {s['cm']:.2f} over {r.cm_steps} steps (peak {r.peak_slots} slots), "
            f"search {s['search']:.2f} s | {r.text}",
            flush=True,
        )
        row = asdict(r)
        del row["scores"]
        rows.append({"file": str(path), "audio_s": audio_s, **row})
    if args.json:
        args.json.write_text(json.dumps({"config": asdict(tr.cfg), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
