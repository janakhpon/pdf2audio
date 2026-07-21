"""documents.py — natural sort, source discovery, and content+config hashing."""

from __future__ import annotations

from pathlib import Path

from pdf2audio.config import Config
from pdf2audio.documents import discover_documents, document_hash, natural_sort_key


def _make_config(**overrides) -> Config:
    base = dict(
        source_path=Path("books"),
        chunk_size=5,
        editor_enabled=False,
        editor_model="gemma3:27b",
        editor_mode="full",
        editor_preserve_context=True,
        editor_url="http://localhost:11434",
        editor_timeout=600,
        audio_model_path="assets/models/kokoro-v1.0.onnx",
        audio_voices_path="assets/models/voices-v1.0.bin",
        audio_voice="af_heart",
        audio_speed=1.0,
        audio_format="mp3",
        out_audio_dir=Path("output/audio"),
        out_transcripts_dir=Path("output/transcripts"),
        save_transcripts=True,
    )
    base.update(overrides)
    return Config(**base)


def test_natural_sort_orders_numeric_runs_as_numbers():
    paths = [Path("chapter_10.html"), Path("chapter_2.html"), Path("chapter_1.html")]
    assert [p.name for p in sorted(paths, key=natural_sort_key)] == [
        "chapter_1.html",
        "chapter_2.html",
        "chapter_10.html",
    ]


def test_discover_pdfs_and_epubs_sorted(tmp_path):
    (tmp_path / "b.pdf").touch()
    (tmp_path / "a.pdf").touch()
    (tmp_path / "c.epub").touch()
    (tmp_path / "notes.txt").touch()  # ignored
    found = discover_documents(tmp_path)
    assert [p.name for p in found] == ["a.pdf", "b.pdf", "c.epub"]


def test_discover_html_dir_returns_the_directory(tmp_path):
    (tmp_path / "1.html").touch()
    (tmp_path / "2.html").touch()
    (tmp_path / "ignored.pdf").touch()  # html presence wins — dir is one document
    found = discover_documents(tmp_path)
    assert found == [tmp_path]


def test_discover_single_file(tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.touch()
    assert discover_documents(pdf) == [pdf]


def test_discover_unsupported_file_is_skipped(tmp_path):
    txt = tmp_path / "book.txt"
    txt.touch()
    assert discover_documents(txt) == []


def test_discover_missing_path_is_skipped(tmp_path):
    assert discover_documents(tmp_path / "nope.pdf") == []


def test_discover_accepts_a_list_of_sources(tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.epub"
    a.touch()
    b.touch()
    assert set(discover_documents([a, b])) == {a, b}


def test_document_hash_is_stable_for_same_content_and_config(tmp_path):
    doc = tmp_path / "book.pdf"
    doc.write_bytes(b"same bytes")
    cfg = _make_config()
    assert document_hash(doc, cfg) == document_hash(doc, cfg)


def test_document_hash_changes_with_content(tmp_path):
    doc = tmp_path / "book.pdf"
    cfg = _make_config()
    doc.write_bytes(b"one")
    first = document_hash(doc, cfg)
    doc.write_bytes(b"two")
    assert document_hash(doc, cfg) != first


def test_document_hash_changes_with_config(tmp_path):
    doc = tmp_path / "book.pdf"
    doc.write_bytes(b"same bytes")
    # A different voice must restart work -> a different key.
    assert document_hash(doc, _make_config(audio_voice="af_heart")) != document_hash(
        doc, _make_config(audio_voice="am_adam")
    )


def test_document_hash_changes_with_chunk_size(tmp_path):
    # chunk_size moves the chunk boundaries, so it MUST key the state: a re-run after changing it
    # gets a fresh DB + audio dir instead of silently reusing the old chunking's audio.
    doc = tmp_path / "book.pdf"
    doc.write_bytes(b"same bytes")
    assert document_hash(doc, _make_config(chunk_size=5)) != document_hash(
        doc, _make_config(chunk_size=6)
    )


def test_document_hash_of_html_dir_uses_names(tmp_path):
    (tmp_path / "1.html").write_text("<p>one</p>")
    (tmp_path / "2.html").write_text("<p>two</p>")
    cfg = _make_config()
    assert document_hash(tmp_path, cfg) == document_hash(tmp_path, cfg)
