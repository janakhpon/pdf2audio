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
        self.last_degraded = False

    def ensure_ready(self) -> bool:
        return True

    def process_transcript(self, text: str) -> str:
        self.last_degraded = False
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


class _TocExtractor:
    # First chunk is a dense table-of-contents page; second is real prose.
    chunks = ["Intro..... 1 Methods..... 12 Results..... 34 Refs..... 88", "real prose here"]

    def __init__(self, chunk_size: int = 5) -> None:
        pass

    def process_file(self, doc_path: Path):
        yield from _TocExtractor.chunks


def test_toc_chunk_is_skipped_not_narrated(tmp_path, monkeypatch):
    seen: list[str] = []

    class _SpyEditor(_FakeEditor):
        def process_transcript(self, text: str) -> str:
            seen.append(text)
            return super().process_transcript(text)

    doc = _make_doc(tmp_path)
    config = _make_config(tmp_path, doc)
    calls, merges = _install_fakes(monkeypatch, extractor=_TocExtractor, editor=_SpyEditor)

    pipeline.process_document(doc, config)

    # The TOC chunk is skipped: synthesized as empty text, and the editor is never asked to polish
    # it. The prose chunk is narrated normally, and the merge still fires.
    assert calls == ["", "real prose here"]
    assert seen == ["real prose here"]
    assert len(merges) == 1


def test_degraded_chunk_counted_and_summary_warned(tmp_path, monkeypatch, caplog):
    import logging

    class _DegradingEditor(_FakeEditor):
        def process_transcript(self, text: str) -> str:
            self.last_degraded = True  # every chunk falls back to raw
            return text

    doc = _make_doc(tmp_path)
    config = _make_config(tmp_path, doc)
    _install_fakes(monkeypatch, editor=_DegradingEditor)

    with caplog.at_level(logging.WARNING):
        pipeline.process_document(doc, config)

    assert any("raw text" in r.getMessage().lower() for r in caplog.records)


def test_worker_survives_a_throwing_failed_mark(tmp_path, monkeypatch):
    # Even if the FAILED-mark itself throws (e.g. a locked SQLite), the worker must not die before
    # calling task_done — otherwise the main thread would block forever on the bounded queue.
    monkeypatch.setattr(pipeline, "_MAX_TTS_RETRIES", 1)  # no backoff sleeps in the test

    def boom(text, output_path):
        raise RuntimeError("tts blew up")

    _install_fakes(monkeypatch, generate=boom)

    real_mark = ChunkStateStore.mark

    def flaky_mark(self, doc_hash, idx, status):
        if status is ChunkStatus.FAILED:
            raise RuntimeError("db locked")
        return real_mark(self, doc_hash, idx, status)

    monkeypatch.setattr(ChunkStateStore, "mark", flaky_mark)

    doc = _make_doc(tmp_path)
    config = _make_config(tmp_path, doc)
    # Must return (not hang, not raise) despite every chunk failing AND the FAILED-mark throwing.
    pipeline.process_document(doc, config)


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
        ("a -- b", "a, b"),  # em-dash -> natural comma pause, no orphan space
        ("multi   space", "multi space"),
        # docling image placeholder is removed cleanly, NOT mangled into "<!, image ,>"
        ("before <!-- image --> after", "before after"),
        ("<!-- formula-not-decoded -->", ""),
        ("![Image](data:image/png;base64,zzz) caption", "caption"),  # image markdown dropped
        ("visit https://example.com/path now", "visit now"),  # bare URL dropped
        ("done. -- Next point", "done. Next point"),  # no orphan comma after a sentence end
        ("the ratio \\( n \\) to \\( k \\)", "the ratio n to k"),  # LaTeX math delimiters dropped
        ("Hash Tables.......... 12", "Hash Tables"),  # TOC dot leaders + page number dropped
        ("wait... really?", "wait... really?"),  # a normal ellipsis (3 dots) is preserved
    ],
)
def test_sanitize_for_tts(raw, expected):
    assert pipeline.sanitize_for_tts(raw) == expected


def test_is_structural_noise_detects_toc_but_not_prose():
    toc = "Intro..... 1, Hashing..... 12, Bloom Filters..... 30, Trees..... 47"
    prose = "Hash tables store key-value pairs. They offer constant-time lookups on average."
    assert pipeline.is_structural_noise(toc) is True
    assert pipeline.is_structural_noise(prose) is False
    assert pipeline.is_structural_noise("a normal sentence... with an ellipsis.") is False


def test_sanitize_for_tts_strips_citation_markers_but_keeps_indices():
    assert pipeline.sanitize_for_tts("proven by Smith [12].") == "proven by Smith."
    # attached array indices and non-numeric brackets are technical content, kept as-is
    assert "a[0]" in pipeline.sanitize_for_tts("the array a[0] holds the value")
    assert "[i]" in pipeline.sanitize_for_tts("the element [i] is next")


def test_sanitize_for_tts_drops_fenced_code_but_keeps_surrounding_prose():
    out = pipeline.sanitize_for_tts(
        "Consider the code ```c\nint32_t a[16];\nfor (int i=0;i<16;i++){}\n``` then continue."
    )
    assert "int32_t" not in out and "```" not in out  # verbatim code block dropped
    assert "c\n" not in out  # the fence language tag does not leak as bare text
    assert "Consider the code" in out and "then continue." in out  # prose kept
    # single-backtick inline identifiers are kept as listenable words, not dropped
    assert "resize()" in pipeline.sanitize_for_tts("Call `resize()` on overflow.")


def test_sanitize_for_tts_drops_bare_hex_literals():
    out = pipeline.sanitize_for_tts("The value 0x162f5a33 hashes to 0xDB608.")
    assert "0x" not in out.lower()  # no hex read aloud
    assert "hashes to" in out  # surrounding prose intact


def test_sanitize_for_tts_keeps_inline_pipe_but_despipes_table_rows():
    # Pipes in prose/math mean "given" / set-builder / absolute value, NOT a table cell divider.
    assert "P(A|B)" in pipeline.sanitize_for_tts("the conditional P(A|B) is read given.")
    assert "{x | x>0}" in pipeline.sanitize_for_tts("the set {x | x>0} of positives")
    assert "|x|" in pipeline.sanitize_for_tts("the absolute value |x| is non-negative")  # 2 pipes!
    # A real markdown row (leading pipe, 2+ cells) still becomes a spoken comma list.
    row = pipeline.sanitize_for_tts("| Alice | 30 | NYC |")
    assert "|" not in row and "Alice" in row and "30" in row


def test_is_structural_noise_density_not_absolute_count():
    # A pure TOC page (dot-leaders dense vs words) is flagged.
    toc = "Intro..... 1 Methods..... 12 Results..... 34 Refs..... 88"
    assert pipeline.is_structural_noise(toc) is True
    # A mostly-prose chunk that merely straddles a few stray leaders is NOT dropped whole.
    prose = ("This chapter explains the method in careful detail. " * 40) + "See TOC..... 3"
    assert pipeline.is_structural_noise(prose) is False


def test_sanitize_for_tts_descaffolds_pipe_table():
    out = pipeline.sanitize_for_tts("| Name | Age |\n|------|-----|\n| Bob | 30 |")
    assert "|" not in out  # no cell dividers voiced as "vertical bar"
    assert "---" not in out and "<!" not in out
    assert "Bob" in out and "30" in out  # the actual data is preserved
