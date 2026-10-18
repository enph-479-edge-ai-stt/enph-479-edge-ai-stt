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
    monkeypatch.setitem(ls.MD5, "mini.tar.gz", ls._md5(tar))
    return calls


def test_download_extracts_then_skips(tmp_path, fake_wget):
    dest, cache = tmp_path / "data", tmp_path / "cache"
    out = ls.download_subset("mini", dest, cache)
    assert out == dest / "LibriSpeech" / "mini"
    assert (out / "1" / "2" / "1-2-0000.flac").is_file()
    assert not (dest / "mini.tar.gz").exists()
    assert not (dest / ".extract-mini").exists()

    assert ls.download_subset("mini", dest, cache) == out
    assert len(fake_wget) == 1  # already extracted: no second download


def test_second_session_copies_from_the_cache(tmp_path, fake_wget):
    cache = tmp_path / "cache"
    ls.download_subset("mini", tmp_path / "session1", cache)
    assert (cache / "mini.tar.gz").is_file()

    out = ls.download_subset("mini", tmp_path / "session2", cache)
    assert (out / "1" / "2" / "1-2-0000.flac").is_file()
    assert len(fake_wget) == 1  # the archive was in the cache: no second download


def test_md5_mismatch_raises_and_deletes_tar(tmp_path, fake_wget, monkeypatch):
    monkeypatch.setitem(ls.MD5, "mini.tar.gz", "0" * 32)
    dest, cache = tmp_path / "data", tmp_path / "cache"
    with pytest.raises(RuntimeError, match="md5"):
        ls.download_subset("mini", dest, cache)
    assert not (dest / "mini.tar.gz").exists()
    assert not (cache / "mini.tar.gz").exists()
    assert not (dest / "LibriSpeech" / "mini").exists()


@pytest.fixture
def fake_wget_file(tmp_path, monkeypatch):
    """Serve a small file instead of downloading one from OpenSLR 11."""
    src = tmp_path / "src.gz"
    src.write_bytes(b"a word lm")
    calls = []

    def run(cmd, **_):
        calls.append(cmd)
        shutil.copy(src, cmd[cmd.index("-O") + 1])

    monkeypatch.setattr(ls.subprocess, "run", run)
    monkeypatch.setitem(ls.MD5, ls.WORD_LM, ls._md5(src))
    return calls


def test_download_word_lm_downloads_then_skips(tmp_path, fake_wget_file):
    dest, cache = tmp_path / "data", tmp_path / "cache"
    path = ls.download_word_lm(dest, cache)
    assert path == dest / ls.WORD_LM
    assert path.read_bytes() == b"a word lm"
    assert (cache / ls.WORD_LM).is_file()
    assert fake_wget_file[0][-1] == f"{ls.OPENSLR_11}/{ls.WORD_LM}"

    assert ls.download_word_lm(dest, cache) == path
    assert len(fake_wget_file) == 1  # already there and checks out: no second download


def test_download_word_lm_md5_mismatch_raises_and_deletes(tmp_path, fake_wget_file, monkeypatch):
    monkeypatch.setitem(ls.MD5, ls.WORD_LM, "0" * 32)
    dest, cache = tmp_path / "data", tmp_path / "cache"
    with pytest.raises(RuntimeError, match="md5"):
        ls.download_word_lm(dest, cache)
    assert not (dest / ls.WORD_LM).exists()
    assert not (cache / ls.WORD_LM).exists()
