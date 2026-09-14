"""Offline tests for training.data.librispeech (no network access)."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
from types import SimpleNamespace

import pytest

from training.data import librispeech as ls
from training.data.librispeech import (
    AUDIO_SUBSETS,
    EXPECTED_HOURS,
    EXPECTED_UTTERANCES,
    _download_resumable,
    _extract_subset,
    _flac_streaminfo,
    _md5,
    download_subset,
    format_summary,
    summarize,
)

# "speaker/chapter" -> [(utterance id, transcript, has its .flac)]
MINI = {
    "1/2": [("1-2-0000", "HELLO WORLD", True), ("1-2-0001", "IT'S FINE", True)],
    "3/4": [("3-4-0000", "ANOTHER ONE", True)],
}
TEN_MINUTES = 16000 * 600  # samples at 16 kHz


def _flac_bytes(rate=16000, channels=1, bits=16, total_samples=TEN_MINUTES) -> bytes:
    """Return a FLAC file's first 26 bytes (marker + STREAMINFO block header).

    Packs the rate/channels/bits/total-samples field the way _flac_streaminfo
    reads it, which is all the header parsing under test needs.
    """
    packed = (rate << 44) | ((channels - 1) << 41) | ((bits - 1) << 36) | total_samples
    return b"fLaC" + bytes([0x80]) + (34).to_bytes(3, "big") + bytes(10) + packed.to_bytes(8, "big")


def _make_subset_tree(root, subset, chapters):
    """Write a fake LibriSpeech/<subset>/<spk>/<chap>/ tree. Returns the subset dir."""
    subset_dir = root / "LibriSpeech" / subset
    for chap, utts in chapters.items():
        spk, ch = chap.split("/")
        d = subset_dir / spk / ch
        d.mkdir(parents=True)
        (d / f"{spk}-{ch}.trans.txt").write_text(
            "".join(f"{utt} {text}\n" for utt, text, _ in utts), encoding="utf-8"
        )
        for utt, _, has_flac in utts:
            if has_flac:
                (d / f"{utt}.flac").write_bytes(_flac_bytes())
    return subset_dir


def _make_subset_tar(build_dir, subset, chapters, meta=("SPEAKERS.TXT",)):
    """Build <subset>.tar.gz with the real archive layout.

    Lays out LibriSpeech/<subset>/... plus corpus-level metadata files directly
    under LibriSpeech/, matching the OpenSLR archives.
    """
    src = build_dir / "src"
    _make_subset_tree(src, subset, chapters)
    for name in meta:
        (src / "LibriSpeech" / name).write_text(f"{name}\n")
    tar = build_dir / f"{subset}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(src / "LibriSpeech", arcname="LibriSpeech")
    return tar


# --- checksums and constants --------------------------------------------------


def test_audio_subset_checksums_are_md5_hex():
    """Every pinned checksum is a 32-char hex md5 digest."""
    for subset, checksum in AUDIO_SUBSETS.items():
        assert len(checksum) == 32, subset
        int(checksum, 16)  # raises ValueError if not hex


def test_expected_tables_cover_every_subset():
    assert set(EXPECTED_UTTERANCES) == set(AUDIO_SUBSETS)
    assert set(EXPECTED_HOURS) == set(AUDIO_SUBSETS)


def test_md5_matches_hashlib(tmp_path):
    """_md5 chunks the file but must equal a one-shot hashlib digest."""
    payload = b"the quick brown fox\n" * 1000
    blob = tmp_path / "blob.bin"
    blob.write_bytes(payload)
    assert _md5(blob) == hashlib.md5(payload).hexdigest()


# --- downloader command selection --------------------------------------------


def test_downloader_prefers_wget_then_curl(monkeypatch, tmp_path):
    seen = []

    def fake_run(cmd, **_):
        seen.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ls.subprocess, "run", fake_run)
    out = tmp_path / "x.tar.gz"

    monkeypatch.setattr(ls.shutil, "which", lambda n: "/usr/bin/wget" if n == "wget" else None)
    _download_resumable("http://x/x.tar.gz", out)
    assert seen[-1][0] == "wget" and "-c" in seen[-1]

    monkeypatch.setattr(ls.shutil, "which", lambda n: "/usr/bin/curl" if n == "curl" else None)
    _download_resumable("http://x/x.tar.gz", out)
    assert seen[-1][0] == "curl"
    assert "-L" in seen[-1] and "-C" in seen[-1]  # follow the openslr mirror redirect, resume

    monkeypatch.setattr(ls.shutil, "which", lambda n: None)
    with pytest.raises(RuntimeError, match="wget or curl"):
        _download_resumable("http://x/x.tar.gz", out)


def test_downloader_nonzero_exit_only_fatal_when_nothing_landed(monkeypatch, tmp_path):
    monkeypatch.setattr(ls.shutil, "which", lambda n: "/usr/bin/curl" if n == "curl" else None)
    monkeypatch.setattr(ls.subprocess, "run", lambda cmd, **_: SimpleNamespace(returncode=33))
    out = tmp_path / "x.tar.gz"

    with pytest.raises(RuntimeError, match="download failed"):
        _download_resumable("http://x/x.tar.gz", out)

    out.write_bytes(b"already complete")  # curl's resume-of-a-complete-file case
    _download_resumable("http://x/x.tar.gz", out)  # no raise; md5 decides


# --- extraction ---------------------------------------------------------------


def test_extract_subset_is_atomic_and_keeps_metadata(tmp_path):
    tar = _make_subset_tar(tmp_path / "build", "mini", MINI, meta=("SPEAKERS.TXT", "README.TXT"))
    dest = tmp_path / "dest"

    out = _extract_subset(tar, "mini", dest)

    assert out == dest / "LibriSpeech" / "mini"
    assert (dest / "LibriSpeech" / "SPEAKERS.TXT").read_text() == "SPEAKERS.TXT\n"
    assert (dest / "LibriSpeech" / "README.TXT").is_file()
    assert not (dest / ".extract-mini").exists(), "scratch dir must not survive"
    assert summarize(out)["ok"]


def test_extract_subset_rejects_archive_for_another_subset(tmp_path):
    tar = _make_subset_tar(tmp_path / "build", "mini", MINI)
    with pytest.raises(RuntimeError, match="did not contain"):
        _extract_subset(tar, "other", tmp_path / "dest")
    assert not (tmp_path / "dest" / "LibriSpeech" / "other").exists()


# --- download_subset orchestration (network replaced by a fake) --------------


def test_download_subset_rejects_unknown(tmp_path):
    """An unknown subset fails fast, before any download is attempted."""
    with pytest.raises(ValueError, match="unknown subset"):
        download_subset("no-such-subset", dest=tmp_path)


def test_download_subset_skips_when_already_extracted(tmp_path):
    """A pre-existing extracted dir short-circuits before any network call."""
    extracted = tmp_path / "LibriSpeech" / "dev-clean"
    extracted.mkdir(parents=True)
    assert download_subset("dev-clean", dest=tmp_path) == extracted


def test_download_subset_end_to_end_offline(tmp_path, monkeypatch):
    """Get -> md5 -> extract -> delete tar, then a second call is a no-op."""
    tar = _make_subset_tar(tmp_path / "build", "mini", MINI)
    monkeypatch.setitem(AUDIO_SUBSETS, "mini", _md5(tar))
    calls = []

    def fake_get(url, out):
        calls.append(url)
        shutil.copy(tar, out)

    monkeypatch.setattr(ls, "_download_resumable", fake_get)
    dest = tmp_path / "dest"

    out = download_subset("mini", dest)

    assert out == dest / "LibriSpeech" / "mini"
    assert calls == [f"{ls.OPENSLR_12}/mini.tar.gz"]
    assert not (dest / "mini.tar.gz").exists(), "tar is deleted after extract by default"
    assert summarize(out)["utterances"] == 3

    assert download_subset("mini", dest) == out
    assert len(calls) == 1, "already-extracted subset must not re-download"


def test_download_subset_keep_tar(tmp_path, monkeypatch):
    tar = _make_subset_tar(tmp_path / "build", "mini", MINI)
    monkeypatch.setitem(AUDIO_SUBSETS, "mini", _md5(tar))
    monkeypatch.setattr(ls, "_download_resumable", lambda url, out: shutil.copy(tar, out))
    dest = tmp_path / "dest"
    download_subset("mini", dest, keep_tar=True)
    assert (dest / "mini.tar.gz").is_file()


def test_download_subset_md5_mismatch_retries_once_then_fails(tmp_path, monkeypatch):
    monkeypatch.setitem(AUDIO_SUBSETS, "mini", "0" * 32)
    calls = []

    def fake_get(url, out):
        calls.append(url)
        out.write_bytes(b"not the archive")

    monkeypatch.setattr(ls, "_download_resumable", fake_get)

    with pytest.raises(RuntimeError, match="md5 mismatch"):
        download_subset("mini", tmp_path / "dest")
    assert len(calls) == 2
    assert not (tmp_path / "dest" / "LibriSpeech" / "mini").exists()


# --- FLAC header ----------------------------------------------------------------


def test_flac_streaminfo_roundtrip(tmp_path):
    f = tmp_path / "a.flac"
    f.write_bytes(_flac_bytes(rate=44100, channels=2, bits=24, total_samples=123456789))
    assert _flac_streaminfo(f) == (44100, 2, 24, 123456789)


def test_flac_streaminfo_rejects_non_flac(tmp_path):
    f = tmp_path / "a.flac"
    f.write_bytes(b"RIFF" + bytes(40))
    assert _flac_streaminfo(f) is None
    f.write_bytes(b"fLaC")  # truncated
    assert _flac_streaminfo(f) is None


# --- summarize / format_summary -----------------------------------------------


def test_summarize_complete_tree(tmp_path):
    out = summarize(_make_subset_tree(tmp_path, "mini", MINI))
    assert out["speakers"] == 2
    assert out["chapters"] == 2
    assert out["utterances"] == 3
    assert out["flac_files"] == 3
    assert out["missing_flac"] == 0
    assert out["empty_transcripts"] == 0
    assert out["bad_flac"] == 0
    assert out["hours"] == 0.5  # 3 x 10 min
    assert out["audio_formats"] == [(16000, 1, 16)]
    assert out["other_chars"] == []  # apostrophe in "IT'S" is in-vocab
    assert out["expected_utterances"] is None  # "mini" is not a real subset
    assert out["ok"]


def test_summarize_flags_missing_flac_and_odd_chars(tmp_path):
    chapters = {"1/2": [("1-2-0000", "HELLO, WORLD", False), ("1-2-0001", "OK", True)]}
    out = summarize(_make_subset_tree(tmp_path, "mini", chapters))
    assert out["missing_flac"] == 1
    assert out["other_chars"] == [","]
    assert not out["ok"]


def test_summarize_flags_empty_transcript(tmp_path):
    chapters = {"1/2": [("1-2-0000", "", True)]}
    out = summarize(_make_subset_tree(tmp_path, "mini", chapters))
    assert out["empty_transcripts"] == 1
    assert not out["ok"]


def test_summarize_flags_wrong_audio_format_and_bad_flac(tmp_path):
    subset_dir = _make_subset_tree(tmp_path, "mini", MINI)
    (subset_dir / "1" / "2" / "1-2-0000.flac").write_bytes(_flac_bytes(rate=44100, channels=2))
    out = summarize(subset_dir)
    assert (44100, 2, 16) in out["audio_formats"]
    assert not out["ok"]

    (subset_dir / "1" / "2" / "1-2-0000.flac").write_bytes(b"garbage")
    out = summarize(subset_dir)
    assert out["bad_flac"] == 1
    assert not out["ok"]


def test_summarize_checks_official_count_for_real_subsets(tmp_path):
    out = summarize(_make_subset_tree(tmp_path, "dev-clean", MINI))
    assert out["expected_utterances"] == 2703
    assert out["utterances"] == 3
    assert not out["ok"]


def test_format_summary_lists_every_subset_and_verdict(tmp_path):
    good = summarize(_make_subset_tree(tmp_path / "a", "mini", MINI))
    bad = summarize(_make_subset_tree(tmp_path / "b", "dev-clean", MINI))
    text = format_summary([good, bad])
    lines = text.splitlines()
    assert lines[0].startswith("subset")
    assert lines[1].startswith("mini") and lines[1].endswith("OK")
    assert lines[2].startswith("dev-clean") and lines[2].endswith("FAIL")
    assert "2703" in lines[2] and "5.4" in lines[2]  # official utterances and hours
    assert lines[-2].endswith("none")
    assert lines[-1].endswith("16000/1/16")
