"""Offline tests for training.data.librispeech (no network access)."""

from __future__ import annotations

import hashlib

import pytest

from training.data.librispeech import AUDIO_SUBSETS, _md5, download_subset


def test_audio_subset_checksums_are_md5_hex():
    """Every pinned checksum is a 32-char hex md5 digest."""
    for subset, checksum in AUDIO_SUBSETS.items():
        assert len(checksum) == 32, subset
        int(checksum, 16)  # raises ValueError if not hex


def test_md5_matches_hashlib(tmp_path):
    """_md5 chunks the file but must equal a one-shot hashlib digest."""
    payload = b"the quick brown fox\n" * 1000
    blob = tmp_path / "blob.bin"
    blob.write_bytes(payload)
    assert _md5(blob) == hashlib.md5(payload).hexdigest()


def test_download_subset_rejects_unknown(tmp_path):
    """An unknown subset fails fast, before any download is attempted."""
    with pytest.raises(ValueError, match="unknown subset"):
        download_subset("no-such-subset", dest=tmp_path)


def test_download_subset_skips_when_already_extracted(tmp_path):
    """A pre-existing extracted dir short-circuits before any network call."""
    extracted = tmp_path / "LibriSpeech" / "dev-clean"
    extracted.mkdir(parents=True)
    assert download_subset("dev-clean", dest=tmp_path) == extracted
