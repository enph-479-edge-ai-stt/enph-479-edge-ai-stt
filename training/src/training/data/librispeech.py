"""Download LibriSpeech audio subsets (and optional LM text) into a local dir.

Built for Colab: downloads go to the VM's local scratch disk (fast, wiped at
session end), are resumable across dropped sessions (wget -c), and audio
archives are md5-verified against OpenSLR's official checksums before extract.

Audio subsets (OpenSLR resource 12) extract to  <dest>/LibriSpeech/<subset>/.
LM files (OpenSLR resource 11) have no published md5, so they get a gzip
integrity check instead.

Example:
    from training.data.librispeech import download
    download(["train-clean-100", "dev-clean", "test-clean"], dest="/content/data")
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tarfile
from pathlib import Path

OPENSLR_12 = "https://www.openslr.org/resources/12"  # audio
OPENSLR_11 = "https://www.openslr.org/resources/11"  # language model

# md5 checksums from https://www.openslr.org/resources/12/md5sum.txt
AUDIO_SUBSETS = {
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other": "fb5a50374b501bb3bac4815ee91d3135",
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
    "train-clean-360": "c0e676e450a7ff2f54aeade5171606fa",
    "train-other-500": "d1a0fd59409feb2c614ce4d30c387708",
}


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _wget_resume(url: str, out: Path) -> None:
    """Resumable download. wget -c completes a partial file, or confirms a
    complete one, or starts fresh if none exists. wget ships on Colab/Ubuntu."""
    subprocess.run(["wget", "-c", "-O", str(out), url], check=True)


def download_subset(subset: str, dest: str | Path, base_url: str = OPENSLR_12,
                    keep_tar: bool = False) -> Path:
    """Download, verify, and extract one LibriSpeech audio subset.

    Returns the extracted subset dir. No-ops if it is already extracted.
    """
    if subset not in AUDIO_SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; known: {sorted(AUDIO_SUBSETS)}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    extracted = dest / "LibriSpeech" / subset
    if extracted.is_dir():
        print(f"[skip] {subset}: already extracted at {extracted}")
        return extracted

    md5 = AUDIO_SUBSETS[subset]
    tar = dest / f"{subset}.tar.gz"
    url = f"{base_url}/{subset}.tar.gz"

    attempts = 0
    while True:
        print(f"[get ] {url}")
        _wget_resume(url, tar)  # resumes a partial from a dropped session
        print(f"[md5 ] verifying {tar.name} ...")
        if _md5(tar) == md5:
            break
        attempts += 1
        if attempts >= 2:
            raise RuntimeError(f"md5 mismatch for {subset} after re-download")
        print(f"[warn] md5 mismatch, deleting and re-downloading {tar.name}")
        tar.unlink()

    print(f"[ok  ] md5 verified, extracting {tar.name} ...")
    with tarfile.open(tar) as t:
        t.extractall(dest)  # official, md5-verified archive; creates LibriSpeech/<subset>
    if not keep_tar:
        tar.unlink()
    print(f"[done] {extracted}")
    return extracted


def download(subsets, dest: str | Path = "data", base_url: str = OPENSLR_12,
             keep_tar: bool = False):
    """Download several audio subsets. Returns the list of extracted dirs."""
    return [download_subset(s, dest, base_url, keep_tar) for s in subsets]


def download_lm(files, dest: str | Path = "data/lm", base_url: str = OPENSLR_11):
    """Download LM resources (resource 11). No published md5; .gz files get a
    read-through gzip integrity check. Files: e.g. librispeech-lm-norm.txt.gz,
    3-gram.arpa.gz, 3-gram.pruned.3e-7.arpa.gz, librispeech-vocab.txt.
    """
    import gzip

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name in files:
        path = dest / name
        if path.exists():
            print(f"[skip] {name}: already present")
            out.append(path)
            continue
        url = f"{base_url}/{name}"
        print(f"[get ] {url}")
        _wget_resume(url, path)
        if name.endswith(".gz"):
            print(f"[chk ] gzip integrity {name} ...")
            with gzip.open(path, "rb") as f:
                while f.read(1 << 20):
                    pass
            print(f"[ok  ] {name}")
        out.append(path)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download LibriSpeech into a local dir.")
    p.add_argument("--subsets", nargs="+",
                   default=["train-clean-100", "dev-clean", "test-clean"],
                   choices=sorted(AUDIO_SUBSETS),
                   help="audio subsets to fetch")
    p.add_argument("--dest", default="data", help="destination dir (use /content/data on Colab)")
    p.add_argument("--keep-tar", action="store_true", help="keep the .tar.gz after extract")
    args = p.parse_args()
    download(args.subsets, dest=args.dest, keep_tar=args.keep_tar)
