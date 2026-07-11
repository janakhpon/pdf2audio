"""DocumentExtractor: text cleaning, the HTML-directory path (natural sort, tag stripping,
chunking), and the typed-error surface for bad inputs. No real PDF/EPUB/docling calls."""

from __future__ import annotations

import pytest
from ebooklib import epub
from pdf2audio.errors import ExtractionError
from pdf2audio.extractor import DocumentExtractor


def _make_epub(path, *, spine, chapters, add_nav=True):
    """Write a minimal real EPUB. `chapters` are (id, filename, body_html) added in that
    (manifest) order; `spine` is a list of ids / the 'nav' token defining reading order."""
    book = epub.EpubBook()
    book.set_identifier("test-id")
    book.set_title("Test Book")
    book.set_language("en")
    items = {}
    for cid, fname, body in chapters:
        item = epub.EpubHtml(uid=cid, file_name=fname, lang="en")
        item.content = body
        book.add_item(item)
        items[cid] = item
    book.add_item(epub.EpubNcx())
    if add_nav:
        book.add_item(epub.EpubNav())
    book.spine = [items.get(s, s) for s in spine]
    epub.write_epub(str(path), book)
    return path


# --------------------------------------------------------------------------- _clean_text


def test_clean_text_collapses_whitespace():
    extractor = DocumentExtractor()
    assert extractor._clean_text("hello   \n\t  world\n\n") == "hello world"


def test_clean_text_empty():
    extractor = DocumentExtractor()
    assert extractor._clean_text("") == ""
    assert extractor._clean_text("   \n  ") == ""


# --------------------------------------------------------------------------- HTML directory path


def _write_html(dir_path, name, body):
    (dir_path / name).write_text(body, encoding="utf-8")


def test_html_dir_natural_sort_and_tag_stripping(tmp_path):
    html_dir = tmp_path / "book"
    html_dir.mkdir()
    # Deliberately create out-of-order filenames with two- and one-digit numbers.
    _write_html(
        html_dir,
        "chapter_2.html",
        "<html><body><nav>SKIP NAV TWO</nav><p>content two</p></body></html>",
    )
    _write_html(
        html_dir,
        "chapter_10.html",
        "<html><body><script>SKIP SCRIPT TEN</script><p>content ten</p></body></html>",
    )
    _write_html(
        html_dir,
        "chapter_1.html",
        "<html><body><nav>SKIP NAV ONE</nav><p>content one</p></body></html>",
    )

    extractor = DocumentExtractor(chunk_size=10)  # one chunk (only 3 files, remainder yielded)
    chunks = list(extractor.process_file(html_dir))

    assert len(chunks) == 1
    combined = chunks[0]

    # Natural sort: 1 then 2 then 10 (not lexical 1, 10, 2).
    assert combined.index("content one") < combined.index("content two")
    assert combined.index("content two") < combined.index("content ten")

    # Nav/script content must be stripped.
    assert "SKIP NAV" not in combined
    assert "SKIP SCRIPT" not in combined


def test_html_dir_chunking_by_chunk_size(tmp_path):
    html_dir = tmp_path / "book"
    html_dir.mkdir()
    for i in range(1, 7):  # six files
        _write_html(html_dir, f"chapter_{i}.html", f"<p>chapter {i} text</p>")

    # chunk_size=2 over 6 files -> exactly 3 chunks.
    chunks = list(DocumentExtractor(chunk_size=2).process_file(html_dir))
    assert len(chunks) == 3
    assert "chapter 1 text" in chunks[0]
    assert "chapter 2 text" in chunks[0]
    assert "chapter 3 text" in chunks[1]


def test_html_dir_remainder_chunk_is_yielded(tmp_path):
    html_dir = tmp_path / "book"
    html_dir.mkdir()
    for i in range(1, 6):  # five files
        _write_html(html_dir, f"chapter_{i}.html", f"<p>chapter {i}</p>")

    # chunk_size=2 over 5 files -> 2 full chunks + 1 remainder = 3 chunks.
    chunks = list(DocumentExtractor(chunk_size=2).process_file(html_dir))
    assert len(chunks) == 3
    assert "chapter 5" in chunks[-1]


def test_dir_without_html_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "notes.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(ExtractionError, match="does not contain HTML"):
        list(DocumentExtractor().process_file(empty_dir))


# --------------------------------------------------------------------------- EPUB path


def test_epub_reads_in_spine_order_skips_nav_and_strips_style(tmp_path):
    path = _make_epub(
        tmp_path / "book.epub",
        # manifest (add) order is c1 then c2 ...
        chapters=[
            (
                "c1",
                "c1.xhtml",
                "<html><body><h1>Chapter One</h1><p>alpha content</p>"
                "<style>.hidden{display:none}</style></body></html>",
            ),
            ("c2", "c2.xhtml", "<html><body><p>beta content</p></body></html>"),
        ],
        # ... but reading order (spine) is nav, then c2, then c1
        spine=["nav", "c2", "c1"],
    )
    # chunk_size=1 -> one chunk per yielded document, so the count proves nav was excluded.
    chunks = list(DocumentExtractor(chunk_size=1).process_file(path))

    assert len(chunks) == 2  # only the two real chapters; nav document excluded
    assert "beta content" in chunks[0]  # spine reading order (c2 first), not manifest order
    assert "alpha content" in chunks[1]
    assert "Chapter One" in chunks[1]  # a chapter <h1> heading is kept
    assert "display:none" not in " ".join(chunks)  # <style> stripped, never narrated


def test_epub_falls_back_to_manifest_order_when_spine_has_no_documents(tmp_path):
    # Spine references only the nav; the real chapters live in the manifest only.
    path = _make_epub(
        tmp_path / "book.epub",
        chapters=[
            ("c1", "c1.xhtml", "<html><body><p>alpha content</p></body></html>"),
            ("c2", "c2.xhtml", "<html><body><p>beta content</p></body></html>"),
        ],
        spine=["nav"],
    )
    chunks = list(DocumentExtractor(chunk_size=1).process_file(path))

    assert len(chunks) == 2  # fell back to manifest documents, nav still excluded
    assert "alpha content" in chunks[0] and "beta content" in chunks[1]  # manifest order


# --------------------------------------------------------------------------- error surface


def test_nonexistent_path_raises(tmp_path):
    with pytest.raises(ExtractionError, match="File not found"):
        list(DocumentExtractor().process_file(tmp_path / "nope.pdf"))


def test_unsupported_suffix_raises(tmp_path):
    txt = tmp_path / "doc.txt"
    txt.write_text("plain text", encoding="utf-8")
    with pytest.raises(ExtractionError, match="Unsupported file format"):
        list(DocumentExtractor().process_file(txt))


def test_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(ExtractionError, match="File is empty"):
        list(DocumentExtractor().process_file(empty))


def test_corrupt_epub_raises_extraction_error(tmp_path):
    # A tiny non-zip blob: ebooklib cannot parse it, and _process_epub wraps the
    # untyped failure in our typed ExtractionError.
    bad = tmp_path / "broken.epub"
    bad.write_bytes(b"not really an epub")
    with pytest.raises(ExtractionError, match="Could not read EPUB"):
        list(DocumentExtractor().process_file(bad))
