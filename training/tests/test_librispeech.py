"""Offline tests for training.data.librispeech (wget is faked, no network)."""

from __future__ import annotations

import shutil
import tarfile

import pytest

from training.data import librispeech as ls


@pytest.fixture
def fake_wget(tmp_path, monkeypatch):
    """Serve a tiny tarball in LibriSpeech's archive layout instead of downloading."""
    chapter = tmp_path / "src" / "LibriSpeech" / "mini" / "1" / "2"
    chapter.mkdir(parents=True)
    (chapter / "1-2.trans.txt").write_text("1-2-0000 HELLO\n")
    (chapter / "1-2-0000.flac").write_bytes(b"not really flac")
    tar = tmp_path / "mini.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(tmp_path / "src" / "LibriSpeech", arcname="LibriSpeech")

    calls = []

    def run(cmd, **_):
        calls.append(cmd)
        shutil.copy(tar, cmd[cmd.index("-O") + 1])

    monkeypatch.setattr(ls.subprocess, "run", run)
    monkeypatch.setitem(ls.MD5, "mini", ls._md5(tar))
    return calls


def test_download_extracts_then_skips(tmp_path, fake_wget):
    dest = tmp_path / "data"
    out = ls.download_subset("mini", dest)
    assert out == dest / "LibriSpeech" / "mini"
    assert (out / "1" / "2" / "1-2-0000.flac").is_file()
    assert not (dest / "mini.tar.gz").exists()
    assert not (dest / ".extract-mini").exists()

    assert ls.download_subset("mini", dest) == out
    assert len(fake_wget) == 1  # already extracted: no second download


def test_md5_mismatch_raises_and_deletes_tar(tmp_path, fake_wget, monkeypatch):
    monkeypatch.setitem(ls.MD5, "mini", "0" * 32)
    dest = tmp_path / "data"
    with pytest.raises(RuntimeError, match="md5"):
        ls.download_subset("mini", dest)
    assert not (dest / "mini.tar.gz").exists()
    assert not (dest / "LibriSpeech" / "mini").exists()
