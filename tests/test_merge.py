"""merge_audio: guards, the ffmpeg invocation shape, concat-list lifecycle, and natural
sort of the fallback glob path. subprocess is always mocked — ffmpeg is never run."""

from __future__ import annotations

import subprocess

import pytest
from pdf2audio import merge as merge_mod
from pdf2audio.errors import MergeError
from pdf2audio.merge import merge_audio


class _FakeCompleted:
    def __init__(self, returncode=0, stderr="") -> None:
        self.returncode = returncode
        self.stderr = stderr


@pytest.fixture
def spy_subprocess(monkeypatch):
    """Capture the command passed to subprocess.run and return success by default."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_missing_directory_raises_without_ffmpeg(tmp_path, spy_subprocess):
    with pytest.raises(MergeError):
        merge_audio(str(tmp_path / "no_such_dir"), "mp3", valid_files=["a.wav"])
    assert spy_subprocess == []  # ffmpeg was never invoked


def test_no_files_raises_without_ffmpeg(tmp_path, spy_subprocess):
    with pytest.raises(MergeError):
        merge_audio(str(tmp_path), "mp3", valid_files=[])
    assert spy_subprocess == []


def test_valid_files_invokes_ffmpeg_and_writes_concat_list(tmp_path, monkeypatch):
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    files = [str(book_dir / "chunk_0001.wav"), str(book_dir / "chunk_0002.wav")]
    for f in files:
        open(f, "wb").close()

    captured = {}

    def fake_run(command, **kwargs):
        # Capture the concat-list contents while the file still exists (before finally cleanup).
        list_path = book_dir / "concat_list.txt"
        captured["command"] = command
        captured["list_exists_during_run"] = list_path.exists()
        captured["list_contents"] = list_path.read_text(encoding="utf-8")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    merge_audio(str(book_dir), "mp3", valid_files=files)

    assert captured["command"][0] == "ffmpeg"
    assert "concat" in captured["command"]
    assert captured["command"][-1].endswith("book_full.mp3")
    assert "-c:a" in captured["command"]  # mp3 -> libmp3lame added
    assert "-af" in captured["command"]  # loudness normalization applied
    assert any("loudnorm" in arg for arg in captured["command"])
    assert captured["list_exists_during_run"] is True
    assert "chunk_0001.wav" in captured["list_contents"]
    assert "chunk_0002.wav" in captured["list_contents"]

    # The concat list must be cleaned up in the finally block.
    assert not (book_dir / "concat_list.txt").exists()


def test_concat_list_cleaned_up_on_ffmpeg_failure(tmp_path, monkeypatch):
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    f = book_dir / "chunk_0001.wav"
    f.touch()

    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: _FakeCompleted(returncode=1, stderr="boom")
    )

    with pytest.raises(MergeError, match="ffmpeg merge failed"):
        merge_audio(str(book_dir), "mp3", valid_files=[str(f)])
    assert not (book_dir / "concat_list.txt").exists()  # cleaned up despite the failure


def test_concat_list_cleaned_up_on_subprocess_error(tmp_path, monkeypatch):
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    f = book_dir / "chunk_0001.wav"
    f.touch()

    def raising_run(command, **kwargs):
        raise subprocess.SubprocessError("spawn failed")

    monkeypatch.setattr(subprocess, "run", raising_run)

    with pytest.raises(MergeError):
        merge_audio(str(book_dir), "mp3", valid_files=[str(f)])
    assert not (book_dir / "concat_list.txt").exists()


def test_fallback_glob_natural_sort_order(tmp_path, monkeypatch):
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    # Create in scrambled order; natural sort must produce 1, 2, 10.
    for name in ["chunk_10.wav", "chunk_1.wav", "chunk_2.wav"]:
        (book_dir / name).touch()

    captured = {}

    def fake_run(command, **kwargs):
        captured["contents"] = (book_dir / "concat_list.txt").read_text(encoding="utf-8")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    # No valid_files -> exercises the legacy glob + natural_sort_key path.
    merge_audio(str(book_dir), "wav")

    lines = [ln for ln in captured["contents"].splitlines() if ln]
    order = [ln for ln in lines]
    assert order[0].endswith("chunk_1.wav'")
    assert order[1].endswith("chunk_2.wav'")
    assert order[2].endswith("chunk_10.wav'")


def test_m4a_uses_aac_codec(tmp_path, monkeypatch):
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    f = book_dir / "chunk_0001.wav"
    f.touch()

    captured = {}

    def fake_run(command, **kwargs):
        captured["cmd"] = command
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    merge_audio(str(book_dir), "m4a", valid_files=[str(f)])
    assert "aac" in captured["cmd"]
    assert captured["cmd"][-1].endswith("book_full.m4a")


def test_merge_module_importable_without_running_ffmpeg():
    # Sanity: the module exposes merge_audio and did not attempt any I/O on import.
    assert callable(merge_mod.merge_audio)


def test_timeout_scales_with_input_size():
    floor = merge_mod._MERGE_TIMEOUT_FLOOR
    rate = merge_mod._LOUDNORM_BYTES_PER_SEC
    # Small books stay at the floor; a multi-GB book gets a proportionally larger budget.
    assert merge_mod._timeout_for(0, rate) == floor
    assert merge_mod._timeout_for(1024, rate) == floor
    big = 3237 * 1024**2  # ~3.2 GB (the book that timed out at the old fixed 600 s)
    assert merge_mod._timeout_for(big, rate) > floor
    assert merge_mod._timeout_for(big, rate) == -(-big // rate)  # ceil division


def test_falls_back_to_plain_merge_when_loudnorm_times_out(tmp_path, monkeypatch):
    # A book too large to loudness-normalize in time must still produce an audiobook (un-normalized)
    # rather than failing outright and discarding hours of TTS work.
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    f = book_dir / "chunk_0001.wav"
    f.write_bytes(b"\0" * 1024)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout"))
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    merge_audio(str(book_dir), "mp3", valid_files=[str(f)])

    assert len(calls) == 2
    # First attempt is loudness-normalized; the fallback drops the filter but still encodes to mp3.
    assert "-af" in calls[0] and any("loudnorm" in a for a in calls[0])
    assert "-af" not in calls[1] and not any("loudnorm" in a for a in calls[1])
    assert "-c:a" in calls[1] and calls[1][-1].endswith("book_full.mp3")
    assert not (book_dir / "concat_list.txt").exists()  # still cleaned up


def test_raises_when_even_the_plain_fallback_times_out(tmp_path, monkeypatch):
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    f = book_dir / "chunk_0001.wav"
    f.write_bytes(b"\0" * 1024)

    def always_timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", always_timeout)
    with pytest.raises(MergeError):
        merge_audio(str(book_dir), "mp3", valid_files=[str(f)])
    assert not (book_dir / "concat_list.txt").exists()


def test_merge_all_raises_when_a_document_fails(tmp_path, monkeypatch):
    # A failed per-document merge must surface as MergeError so `pdf2audio merge` exits non-zero
    # instead of falsely reporting success.
    book_dir = tmp_path / "book"
    book_dir.mkdir()

    monkeypatch.setattr(merge_mod.documents, "discover_documents", lambda src: [tmp_path / "book"])
    monkeypatch.setattr(
        merge_mod, "merge_audio", lambda *a, **k: (_ for _ in ()).throw(MergeError("ffmpeg died"))
    )

    class _Cfg:
        source_path = tmp_path
        out_audio_dir = tmp_path
        audio_format = "mp3"

    with pytest.raises(MergeError, match="failed to merge"):
        merge_mod.merge_all(_Cfg())
