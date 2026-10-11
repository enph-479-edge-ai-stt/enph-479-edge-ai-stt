"""Synthetic data the test modules share: a tiny LibriSpeech subset and a tiny text corpus.

Nothing here imports torch at module level, so CI (which has no torch) can still load it.
"""

from __future__ import annotations

TEXTS = ["HELLO WORLD", "CAT", "IT'S FINE", "DOG"]
SENTENCES = ["HELLO WORLD", "THE CAT SAT", "IT'S FINE", "A DOG RAN HOME"]


def write_subset(root):
    """Four noise FLACs (0.5-1.1 s) with ``TEXTS`` as transcripts, in LibriSpeech's layout."""
    import soundfile as sf
    import torch

    chapter = root / "1" / "2"
    chapter.mkdir(parents=True)
    lines = []
    for i, text in enumerate(TEXTS):
        wav = (torch.randn(8000 + 3000 * i) * 0.1).numpy()
        sf.write(str(chapter / f"1-2-{i:04d}.flac"), wav, 16000, format="FLAC")
        lines.append(f"1-2-{i:04d} {text}")
    (chapter / "1-2.trans.txt").write_text("\n".join(lines), encoding="utf-8")
    return root
