"""Get LibriSpeech audio subsets, LM text and word LM into a local dir (e.g. /content).

Each archive comes from a cache dir on Google Drive if it is there. Otherwise it is
downloaded from OpenSLR and copied into the cache, so only the first session downloads.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

# OpenSLR's EU mirror for the audio: from Colab the main server ran at 0.8 MB/s on
# 2026-11-03 (8 hours for train-clean-360), the mirror at 55 MB/s.
OPENSLR_12 = "https://openslr.elda.org/resources/12"
# OpenSLR's EU mirror for the LM files: the main server ran at about 0.1 MB/s on
# 2026-11-03, the mirror at 10 MB/s.
OPENSLR_11 = "https://openslr.elda.org/resources/11"
LM_TEXT = "librispeech-lm-norm.txt.gz"
WORD_LM = "3-gram.arpa.gz"

# Audio subsets from https://www.openslr.org/resources/12/md5sum.txt. OpenSLR 11
# publishes no checksums; the LM text's was computed from a download on 2026-10-01,
# the word LM's on 2026-11-03.
MD5 = {
    "dev-clean.tar.gz": "42e2234ba48799c1f50f24a7926300a1",
    "test-clean.tar.gz": "32fa31d27d2e1cad72775fee3f4849a9",
    "train-clean-100.tar.gz": "2a93770f6d5c6c964bc36631d331a522",
    "train-clean-360.tar.gz": "c0e676e450a7ff2f54aeade5171606fa",
    LM_TEXT: "c83c64c726a1aedfe65f80aa311de402",
    WORD_LM: "74cbb837e582378c26c602e1c8cbcb9c",
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(base_url: str, name: str, dest: Path, cache: Path) -> Path:
    """Put the archive ``name`` in ``dest`` and md5-check it; return its path.

    Copies it from ``cache`` if it is there. Otherwise downloads it from ``base_url``
    and copies it into ``cache``. The cache holds the archives as downloaded, not
    extracted: Drive is fast with a few big files and slow with thousands of small ones.
    """
    local, cached = dest / name, cache / name
    in_cache = cached.is_file()
    dest.mkdir(parents=True, exist_ok=True)
    if in_cache:
        print(f"copying {cached}", flush=True)
        shutil.copy(cached, local)
    else:
        url = f"{base_url}/{name}"
        print(f"downloading {url}", flush=True)
        subprocess.run(["wget", "-nv", "-c", "-O", str(local), url], check=True)
    if _md5(local) != MD5[name]:
        local.unlink()
        cached.unlink(missing_ok=True)
        raise RuntimeError(f"md5 mismatch for {name}; deleted it, re-run to download again")
    if not in_cache:
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, cached)
    return local


def download_subset(subset: str, dest: str | Path, cache: str | Path) -> Path:
    """Fetch, md5-check and extract one subset; return ``<dest>/LibriSpeech/<subset>``.

    Skips everything if the subset is already extracted. Extracts into a temp dir
    and renames it into place, so an interrupted extract never looks finished.
    """
    dest = Path(dest)
    out = dest / "LibriSpeech" / subset
    if out.is_dir():
        return out

    t0 = time.perf_counter()
    tar = _fetch(OPENSLR_12, f"{subset}.tar.gz", dest, Path(cache))
    tmp = dest / f".extract-{subset}"
    with tarfile.open(tar) as t:
        t.extractall(tmp, filter="data")
    out.parent.mkdir(parents=True, exist_ok=True)
    (tmp / "LibriSpeech" / subset).rename(out)
    shutil.rmtree(tmp)
    tar.unlink()
    print(f"{subset} ready at {out} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return out


def _download_file(name: str, dest: str | Path, cache: str | Path) -> Path:
    """Fetch and md5-check one OpenSLR 11 file into ``dest``; return its path.

    Skips everything if the file is already in ``dest`` and checks out.
    """
    dest = Path(dest)
    path = dest / name
    if path.is_file() and _md5(path) == MD5[name]:
        return path

    t0 = time.perf_counter()
    _fetch(OPENSLR_11, name, dest, Path(cache))
    print(f"{name} ready at {path} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return path


def download_lm_text(dest: str | Path, cache: str | Path) -> Path:
    """Fetch and md5-check the normalized LM text (1.5 GB gzip); return its path.

    One upper-case sentence per line, from ~14.5K books with the dev/test books
    excluded. Stays gzipped: it is ~4.3 GB extracted and is only ever streamed.
    """
    return _download_file(LM_TEXT, dest, cache)


def download_word_lm(dest: str | Path, cache: str | Path) -> Path:
    """Fetch and md5-check the word LM (759 MB gzip ARPA); return its path.

    OpenSLR's unpruned trigram over the same LM text (so dev/test books excluded):
    200K words, 38.2M bigrams, 49.7M trigrams, upper case like the transcripts. KenLM
    reads the gzipped ARPA directly, into about 1.8 GB of RAM (about 3 minutes on the
    KV260).
    """
    return _download_file(WORD_LM, dest, cache)
