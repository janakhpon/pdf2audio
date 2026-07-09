"""pipeline.process_document — the real orchestration + resume path, with the heavy
components (extractor, editor, audio, merge) faked so the test is offline and fast.

This is the integration test that was previously deferred because the logic lived as
closures inside a 297-line function; it now exercises the real state transitions,
resume-skip, per-chunk failure isolation, and the low-disk abort.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pytest
from pdf2audio import pipeline
from pdf2audio.config import Config
from pdf2audio.errors import AudioError, PDF2AudioError
from pdf2audio.state import ChunkStateStore, ChunkStatus

_Usage = namedtuple("_Usage", "total used free")


def _make_config(tmp_path: Path, doc: Path, **overrides) -> Config:
    base = dict(
        source_path=doc,
        chunk_size=5,
        editor_enabled=False,
        editor_model="gemma3:27b",
        editor_mode="full",
        editor_preserve_context=True,
        editor_url="http://localhost:11434",
        editor_timeout=600,
        audio_model_path="model.onnx",
        audio_voices_path="voices.bin",
        audio_voice="af_heart",
        audio_speed=1.0,
        audio_format="mp3",
        out_audio_dir=tmp_path / "audio",
        out_transcripts_dir=tmp_path / "transcripts",
        save_transcripts=True,
    )
    base.update(overrides)
    return Config(**base)


class _FakeExtractor:
    chunks = ["chunk one", "chunk two"]

    def __init__(self, chunk_size: int = 5) -> None:
        pass

    def process_file(self, doc_path: Path):
        yield from _FakeExtractor.chunks


class _FakeEditor:
    def __init__(self, config: Config) -> None:
        pass

    def process_transcript(self, text: str) -> str:
        return text

    def load_saved_context(self, text: str) -> None:
        pass


def _install_fakes(monkeypatch, *, generate=None, extractor=_FakeExtractor, editor=_FakeEditor):
    calls: list[str] = []

    class _FakeAudio:
        def __init__(self, config: Config) -> None:
            pass

        def generate(self, text: str, output_path: Path) -> None:
            calls.append(text)
            if generate is not None:
                generate(text, output_path)
            output_path.with_suffix(".wav").write_bytes(b"RIFFfake")

    merges: list[tuple] = []

    def _fake_merge(directory, output_format="mp3", valid_files=None):
        merges.append((directory, output_format, valid_files))

    monkeypatch.setattr(pipeline, "DocumentExtractor", extractor)
    monkeypatch.setattr(pipeline, "SmartEditor", editor)
    monkeypatch.setattr(pipeline, "AudioEngine", _FakeAudio)
    monkeypatch.setattr(pipeline, "merge_audio", _fake_merge)
    return calls, merges


def _make_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "book.pdf"
    doc.write_bytes(b"pdf bytes")
    return doc


def test_happy_path_marks_all_done_and_merges(tmp_path, monkeypatch):
    doc = _make_doc(tmp_path)
    config = _make_config(tmp_path, doc)
    calls, merges = _install_fakes(monkeypatch)

    pipeline.process_document(doc, config)

    assert calls == ["chunk one", "chunk two"]  # both synthesized
    assert len(merges) == 1  # merge fired once
    _dir, _fmt, valid_files = merges[0]
    assert valid_files is not None and len(valid_files) == 2

    db = config.out_audio_dir / next(p.name for p in config.out_audio_dir.glob("*.db"))
    from pdf2audio.documents import document_hash

    with ChunkStateStore(db) as store:
        assert store.pending_count(document_hash(doc, config)) == 0


def test_resume_skips_completed_chunks(tmp_path, monkeypatch):
    doc = _make_doc(tmp_path)
    config = _make_config(tmp_path, doc)
    from pdf2audio.documents import document_hash

    doc_hash = document_hash(doc, config)

    # Pre-stage chunk 1 as DONE with its audio + transcript already on disk.
    audio_dir = config.out_audio_dir / doc.stem
    tx_dir = config.out_transcripts_dir / doc.stem
    audio_dir.mkdir(parents=True)
    tx_dir.mkdir(parents=True)
    (audio_dir / "chunk_0001.wav").write_bytes(b"RIFFdone")
    (tx_dir / "chunk_0001.txt").write_text("chunk one")
    with ChunkStateStore(config.out_audio_dir / f"pdf2audio_state_{doc_hash}.db") as store:
        store.mark(doc_hash, 1, ChunkStatus.DONE)

    calls, merges = _install_fakes(monkeypatch)
    pipeline.process_document(doc, config)

    assert calls == ["chunk two"]  # chunk one skipped, only chunk two synthesized
    assert len(merges) == 1


def test_chunk_failure_is_isolated_and_blocks_merge(tmp_path, monkeypatch):
    doc = _make_doc(tmp_path)
    config = _make_config(tmp_path, doc)

    class _FailingEditor(_FakeEditor):
        def process_transcript(self, text: str) -> str:
            if text == "chunk two":
                raise AudioError("synthetic failure")
            return text

    calls, merges = _install_fakes(monkeypatch, editor=_FailingEditor)
    pipeline.process_document(doc, config)

    assert calls == ["chunk one"]  # chunk two never reached synthesis
    assert merges == []  # a FAILED chunk blocks the merge

    from pdf2audio.documents import document_hash

    with ChunkStateStore(
        config.out_audio_dir / f"pdf2audio_state_{document_hash(doc, config)}.db"
    ) as s:
        assert s.pending_count(document_hash(doc, config)) == 1


def test_low_disk_aborts_the_run(tmp_path, monkeypatch):
    doc = _make_doc(tmp_path)
    config = _make_config(tmp_path, doc)
    _install_fakes(monkeypatch)
    monkeypatch.setattr(
        pipeline.shutil, "disk_usage", lambda _p: _Usage(1_000, 999, pipeline.MIN_FREE_BYTES - 1)
    )

    with pytest.raises(PDF2AudioError, match="disk space"):
        pipeline.process_document(doc, config)


def test_unsupported_file_is_a_noop(tmp_path, monkeypatch):
    doc = tmp_path / "notes.txt"
    doc.write_text("hi")
    config = _make_config(tmp_path, doc)
    calls, merges = _install_fakes(monkeypatch)

    pipeline.process_document(doc, config)

    assert calls == [] and merges == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("**bold** text", "bold text"),
        ("# Heading\ntext", "Heading text"),
        ("use `code` here", "use code here"),
        ("see [link](http://x.com)", "see link"),
        ("a -- b", "a , b"),
        ("multi   space", "multi space"),
    ],
)
def test_sanitize_for_tts(raw, expected):
    assert pipeline.sanitize_for_tts(raw) == expected
