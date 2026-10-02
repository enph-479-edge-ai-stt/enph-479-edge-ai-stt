"""Download LibriSpeech audio subsets and LM text from OpenSLR into a local dir (e.g. /content)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

OPENSLR_12 = "https://www.openslr.org/resources/12"
OPENSLR_11 = "https://www.openslr.org/resources/11"
LM_TEXT = "librispeech-lm-norm.txt.gz"

# Audio subsets from https://www.openslr.org/resources/12/md5sum.txt. OpenSLR 11
# publishes no checksums; the LM text's was computed from a download on 2026-10-01.
MD5 = {
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
    LM_TEXT: "c83c64c726a1aedfe65f80aa311de402",
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_subset(subset: str, dest: str | Path) -> Path:
    """Download, md5-check and extract one subset; return ``<dest>/LibriSpeech/<subset>``.

    Skips everything if the subset is already extracted. Extracts into a temp dir
    and renames it into place, so an interrupted extract never looks finished.
    """
    dest = Path(dest)
    out = dest / "LibriSpeech" / subset
    if out.is_dir():
        return out

    t0 = time.perf_counter()
    dest.mkdir(parents=True, exist_ok=True)
    tar = dest / f"{subset}.tar.gz"
    url = f"{OPENSLR_12}/{subset}.tar.gz"
    print(f"downloading {url}", flush=True)
    subprocess.run(["wget", "-nv", "-c", "-O", str(tar), url], check=True)
    if _md5(tar) != MD5[subset]:
        tar.unlink()
        raise RuntimeError(f"md5 mismatch for {tar.name}; deleted it, re-run to download again")

    tmp = dest / f".extract-{subset}"
    with tarfile.open(tar) as t:
        t.extractall(tmp, filter="data")
    out.parent.mkdir(parents=True, exist_ok=True)
    (tmp / "LibriSpeech" / subset).rename(out)
    shutil.rmtree(tmp)
    tar.unlink()
    print(f"{subset} ready at {out} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return out


def download_lm_text(dest: str | Path) -> Path:
    """Download and md5-check the normalized LM text (1.5 GB gzip); return its path.

    One upper-case sentence per line, from ~14.5K books with the dev/test books
    excluded. Stays gzipped: it is ~4.3 GB extracted and is only ever streamed.
    Skips the download if the file is already there and checks out.
    """
    dest = Path(dest)
    gz = dest / LM_TEXT
    if gz.is_file() and _md5(gz) == MD5[LM_TEXT]:
        return gz

    t0 = time.perf_counter()
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{OPENSLR_11}/{LM_TEXT}"
    print(f"downloading {url}", flush=True)
    subprocess.run(["wget", "-nv", "-c", "-O", str(gz), url], check=True)
    if _md5(gz) != MD5[LM_TEXT]:
        gz.unlink()
        raise RuntimeError(f"md5 mismatch for {gz.name}; deleted it, re-run to download again")
    print(f"{gz.name} ready at {gz} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return gz
