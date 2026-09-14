"""Download LibriSpeech audio subsets (and optional LM text) into a local dir.

Built for Colab: downloads go to the VM's local scratch disk (fast, wiped at
session end) and are resumable across dropped sessions (``wget -c`` on Colab
and Ubuntu, ``curl -C -`` on a dev box without wget). Audio archives are
md5-verified against OpenSLR's official checksums before extract, and
extraction is atomic (unpack to a scratch dir, then rename into place), so a
session that dies mid-extract can never leave a half-populated subset dir that
a later run mistakes for complete.

Audio subsets (OpenSLR resource 12) extract to ``<dest>/LibriSpeech/<subset>/``
with the corpus metadata files (SPEAKERS.TXT, CHAPTERS.TXT, ...) next to them
in ``<dest>/LibriSpeech/``. LM files (OpenSLR resource 11) have no published
md5, so they get a gzip integrity check instead.

Example:
    from training.data.librispeech import download, summarize
    dirs = download(["dev-clean", "test-clean", "train-clean-100"], dest="/content/data")
    for d in dirs:
        print(summarize(d))
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import string
import subprocess
import tarfile
import time
from pathlib import Path

OPENSLR_12 = "https://www.openslr.org/resources/12"  # audio
OPENSLR_11 = "https://www.openslr.org/resources/11"  # language model

# md5 checksums from https://www.openslr.org/resources/12/md5sum.txt (re-checked 2026-09-14)
AUDIO_SUBSETS = {
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other": "fb5a50374b501bb3bac4815ee91d3135",
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
    "train-clean-360": "c0e676e450a7ff2f54aeade5171606fa",
    "train-other-500": "d1a0fd59409feb2c614ce4d30c387708",
}

# Utterance count per subset, from the corpus statistics (also listed per split
# in the openslr/librispeech_asr dataset card). summarize() compares against
# these to flag a truncated or partial extract.
EXPECTED_UTTERANCES = {
    "dev-clean": 2703,
    "dev-other": 2864,
    "test-clean": 2620,
    "test-other": 2939,
    "train-clean-100": 28539,
    "train-clean-360": 104014,
    "train-other-500": 148688,
}

# Official hours per subset (LibriSpeech paper, Panayotov et al. 2015, Table 2).
# Shown next to the measured duration in the summary table; not a hard check.
EXPECTED_HOURS = {
    "dev-clean": 5.4,
    "dev-other": 5.3,
    "test-clean": 5.4,
    "test-other": 5.1,
    "train-clean-100": 100.6,
    "train-clean-360": 363.6,
    "train-other-500": 496.7,
}

# LibriSpeech transcripts are upper-case letters, space and apostrophe.
# summarize() reports anything outside this set so the vocab build can decide
# what to do with it.
TRANSCRIPT_CHARS = frozenset(string.ascii_uppercase + " '")


def _log(msg: str) -> None:
    print(msg, flush=True)  # keep ordering sane next to wget/curl output


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_resumable(url: str, out: Path) -> None:
    """Resumable download: completes a partial file, confirms a complete one,
    or starts fresh. Prefers wget (ships on Colab/Ubuntu), falls back to curl
    (ships on Windows 10+ and macOS) so this also runs on a dev box.

    A non-zero exit with the file already present is tolerated (curl reports
    resuming an already-complete file as an error); the caller's integrity
    check is the real verdict on the bytes.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        cmd = ["wget", "-c", "-O", str(out), url]
    elif shutil.which("curl"):
        cmd = ["curl", "-L", "-C", "-", "--retry", "3", "-o", str(out), url]
    else:
        raise RuntimeError(f"need wget or curl on PATH to download {url}")
    rc = subprocess.run(cmd).returncode
    if rc != 0 and not out.exists():
        raise RuntimeError(f"download failed (exit {rc}): {url}")


def _extract_subset(tar: Path, subset: str, dest: Path) -> Path:
    """Unpack ``tar`` into a scratch dir under ``dest``, then rename the finished
    subset dir into ``dest/LibriSpeech/<subset>``. Corpus-level metadata files
    shipped in every archive (SPEAKERS.TXT, CHAPTERS.TXT, ...) are moved next
    to it if not already there. The scratch dir is removed at the end, so a
    session that dies mid-extract leaves no final dir behind."""
    root = dest / "LibriSpeech"
    final = root / subset
    scratch = dest / f".extract-{subset}"
    if scratch.exists():
        shutil.rmtree(scratch)
    with tarfile.open(tar) as t:
        t.extractall(scratch, filter="data")  # archive layout: LibriSpeech/<subset>/...
    unpacked = scratch / "LibriSpeech"
    if not (unpacked / subset).is_dir():
        raise RuntimeError(f"{tar.name} did not contain LibriSpeech/{subset}/")
    root.mkdir(parents=True, exist_ok=True)
    for meta in unpacked.iterdir():
        if meta.is_file() and not (root / meta.name).exists():
            meta.rename(root / meta.name)
    (unpacked / subset).rename(final)
    shutil.rmtree(scratch)
    return final


def download_subset(
    subset: str, dest: str | Path, base_url: str = OPENSLR_12, keep_tar: bool = False
) -> Path:
    """Download, verify, and extract one LibriSpeech audio subset.

    Returns the extracted subset dir. No-ops if it is already extracted.
    """
    if subset not in AUDIO_SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; known: {sorted(AUDIO_SUBSETS)}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    extracted = dest / "LibriSpeech" / subset
    if extracted.is_dir():
        _log(f"[skip] {subset}: already extracted at {extracted}")
        return extracted

    t0 = time.perf_counter()
    md5 = AUDIO_SUBSETS[subset]
    tar = dest / f"{subset}.tar.gz"
    url = f"{base_url}/{subset}.tar.gz"

    for attempt in (1, 2):
        _log(f"[get ] {url}")
        _download_resumable(url, tar)  # resumes a partial from a dropped session
        _log(f"[md5 ] verifying {tar.name} ({tar.stat().st_size / 1e9:.2f} GB) ...")
        if _md5(tar) == md5:
            break
        if attempt == 2:
            raise RuntimeError(f"md5 mismatch for {subset} after re-download")
        _log(f"[warn] md5 mismatch, deleting and re-downloading {tar.name}")
        tar.unlink()

    _log(f"[ok  ] md5 verified, extracting {tar.name} ...")
    _extract_subset(tar, subset, dest)
    if not keep_tar:
        tar.unlink()
    _log(f"[done] {extracted} ({time.perf_counter() - t0:.0f} s)")
    return extracted


def download(
    subsets, dest: str | Path = "data", base_url: str = OPENSLR_12, keep_tar: bool = False
) -> list[Path]:
    """Download several audio subsets. Returns the list of extracted dirs."""
    return [download_subset(s, dest, base_url, keep_tar) for s in subsets]


def download_lm(files, dest: str | Path = "data/lm", base_url: str = OPENSLR_11) -> list[Path]:
    """Download LM resources (resource 11). No published md5; .gz files get a
    read-through gzip integrity check. Files: e.g. librispeech-lm-norm.txt.gz,
    3-gram.arpa.gz, 3-gram.pruned.3e-7.arpa.gz, librispeech-vocab.txt.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name in files:
        path = dest / name
        if path.exists():
            _log(f"[skip] {name}: already present")
            out.append(path)
            continue
        url = f"{base_url}/{name}"
        _log(f"[get ] {url}")
        _download_resumable(url, path)
        if name.endswith(".gz"):
            _log(f"[chk ] gzip integrity {name} ...")
            with gzip.open(path, "rb") as f:
                while f.read(1 << 20):
                    pass
            _log(f"[ok  ] {name}")
        out.append(path)
    return out


def _flac_streaminfo(path: Path) -> tuple[int, int, int, int] | None:
    """(sample_rate, channels, bits_per_sample, total_samples) from a FLAC file's
    STREAMINFO block, which is always the first metadata block after the "fLaC"
    marker. Reads 26 bytes, no decoder needed. None if the file is not FLAC."""
    with open(path, "rb") as f:
        head = f.read(26)
    if len(head) < 26 or head[:4] != b"fLaC" or head[4] & 0x7F != 0:
        return None
    # bytes 18..25: sample rate (20 bits) | channels-1 (3) | bps-1 (5) | total samples (36)
    v = int.from_bytes(head[18:26], "big")
    return v >> 44, ((v >> 41) & 0x7) + 1, ((v >> 36) & 0x1F) + 1, v & ((1 << 36) - 1)


def summarize(subset_dir: str | Path) -> dict:
    """Stdlib-only sanity check of one extracted subset.

    Walks every ``*.trans.txt`` (one per speaker/chapter dir), counts speakers,
    chapters and utterances, confirms each transcript line has its ``.flac``,
    collects any transcript characters outside ``TRANSCRIPT_CHARS``, and reads
    every FLAC header for sample rate / channels / duration. ``ok`` is True when
    nothing is missing, empty or malformed, every file is 16 kHz mono, and the
    utterance count matches ``EXPECTED_UTTERANCES`` (subsets not in that table
    only get the completeness and format checks).
    """
    subset_dir = Path(subset_dir)
    subset = subset_dir.name
    speakers: set[str] = set()
    chapters = 0
    utterances = 0
    empty = 0
    missing: list[str] = []
    other_chars: set[str] = set()
    for trans in sorted(subset_dir.glob("*/*/*.trans.txt")):
        chapters += 1
        speakers.add(trans.parent.parent.name)
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, _, text = line.partition(" ")
            utterances += 1
            if not text.strip():
                empty += 1
            other_chars.update(set(text) - TRANSCRIPT_CHARS)
            if not (trans.parent / f"{utt_id}.flac").is_file():
                missing.append(utt_id)

    flac_files = 0
    bad_flac = 0
    seconds = 0.0
    formats: set[tuple[int, int, int]] = set()
    for flac in subset_dir.rglob("*.flac"):
        flac_files += 1
        info = _flac_streaminfo(flac)
        if info is None:
            bad_flac += 1
            continue
        rate, channels, bits, total_samples = info
        formats.add((rate, channels, bits))
        seconds += total_samples / rate

    expected = EXPECTED_UTTERANCES.get(subset)
    return {
        "subset": subset,
        "speakers": len(speakers),
        "chapters": chapters,
        "utterances": utterances,
        "expected_utterances": expected,
        "flac_files": flac_files,
        "missing_flac": len(missing),
        "empty_transcripts": empty,
        "bad_flac": bad_flac,
        "hours": round(seconds / 3600, 2),
        "audio_formats": sorted(formats),
        "other_chars": sorted(other_chars),
        "ok": (
            not missing
            and empty == 0
            and bad_flac == 0
            and flac_files == utterances
            and all(rate == 16000 and channels == 1 for rate, channels, _ in formats)
            and (expected is None or utterances == expected)
        ),
    }


def format_summary(rows: list[dict]) -> str:
    """Render summarize() dicts as a fixed-width table for notebook output."""
    lines = [
        f"{'subset':<16}{'spk':>5}{'chap':>6}{'utts':>8}{'expected':>9}"
        f"{'flac':>8}{'missing':>8}{'hours':>8}{'official':>9}  ok"
    ]
    for r in rows:
        expected = r["expected_utterances"]
        official_hours = EXPECTED_HOURS.get(r["subset"])
        lines.append(
            f"{r['subset']:<16}{r['speakers']:>5}{r['chapters']:>6}{r['utterances']:>8}"
            f"{(expected if expected is not None else '-'):>9}{r['flac_files']:>8}"
            f"{r['missing_flac']:>8}{r['hours']:>8.2f}"
            f"{(official_hours if official_hours is not None else '-'):>9}"
            f"  {'OK' if r['ok'] else 'FAIL'}"
        )
    chars = sorted({c for r in rows for c in r["other_chars"]})
    formats = sorted({f for r in rows for f in r["audio_formats"]})
    lines.append(f"transcript chars outside A-Z / space / apostrophe: {chars or 'none'}")
    lines.append(
        "audio formats (Hz/channels/bits): "
        + (", ".join(f"{rate}/{ch}/{bits}" for rate, ch, bits in formats) or "none")
    )
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download LibriSpeech into a local dir.")
    p.add_argument(
        "--subsets",
        nargs="+",
        default=["dev-clean", "test-clean", "train-clean-100"],
        choices=sorted(AUDIO_SUBSETS),
        help="audio subsets to fetch",
    )
    p.add_argument("--dest", default="data", help="destination dir (use /content/data on Colab)")
    p.add_argument("--keep-tar", action="store_true", help="keep the .tar.gz after extract")
    args = p.parse_args()
    dirs = download(args.subsets, dest=args.dest, keep_tar=args.keep_tar)
    print(format_summary([summarize(d) for d in dirs]))
