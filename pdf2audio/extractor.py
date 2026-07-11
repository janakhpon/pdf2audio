from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from pdf2audio.documents import natural_sort_key
from pdf2audio.errors import ExtractionError
from pdf2audio.logger import logger

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

# Reject implausibly large inputs early rather than OOM-ing deep inside docling/ebooklib.
_MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB

# Tags to drop before reading text. Loose HTML files carry site chrome, so strip broadly; EPUB
# chapter bodies are already just the chapter, so strip only what is never reading content (a
# chapter's <header>/<aside> can hold its title or a pull-quote, so those are kept for EPUBs).
_HTML_DIR_DROP_TAGS = ["nav", "header", "footer", "script", "style", "aside", "noscript", "form"]
_EPUB_DROP_TAGS = ["script", "style", "noscript"]


def _soup_to_text(content: str | bytes, drop_tags: list[str]) -> str:
    """Parse HTML, remove non-content tags, and return whitespace-joined visible text."""
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(drop_tags):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


class DocumentExtractor:
    def __init__(self, chunk_size: int = 5) -> None:
        self.chunk_size = chunk_size
        self._docling_converter: DocumentConverter | None = None

    def _get_docling_converter(self) -> DocumentConverter:
        if self._docling_converter is None:
            # Lazy load the (heavy) Docling model only when a PDF is actually processed.
            logger.info("Initializing Docling DocumentConverter...")
            from docling.document_converter import DocumentConverter

            self._docling_converter = DocumentConverter()
        return self._docling_converter

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Collapse consecutive whitespace, including newlines.
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _check_size(self, file_path: Path) -> None:
        size = file_path.stat().st_size
        if size == 0:
            raise ExtractionError(f"File is empty: {file_path}")
        if size > _MAX_FILE_BYTES:
            raise ExtractionError(
                f"File too large ({size / 1024**2:.0f} MB > "
                f"{_MAX_FILE_BYTES // 1024**2} MB): {file_path}"
            )

    def process_file(self, file_path: Path) -> Iterator[str]:
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        logger.info(f"Extracting: {file_path.name}")

        if file_path.is_dir():
            if list(file_path.glob("*.html")):
                yield from self._process_html_dir(file_path)
                return
            raise ExtractionError(f"Directory {file_path} does not contain HTML files.")

        self._check_size(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            yield from self._process_pdf(file_path)
        elif suffix == ".epub":
            yield from self._process_epub(file_path)
        else:
            raise ExtractionError(f"Unsupported file format: {suffix}")

    def _process_pdf(self, file_path: Path) -> Iterator[str]:
        # Docling extracts natively as Markdown.
        converter = self._get_docling_converter()
        try:
            result = converter.convert(str(file_path))
            markdown_text = result.document.export_to_markdown()
        except Exception as exc:  # docling raises a variety of untyped errors
            raise ExtractionError(f"Could not extract PDF {file_path.name}: {exc}") from exc

        blocks = [block.strip() for block in markdown_text.split("\n\n") if block.strip()]

        current_chunk: list[str] = []
        count = 0

        for i, block in enumerate(blocks):
            current_chunk.append(self._clean_text(block))

            # Docling exports a single string, so we treat logical blocks as page analogues,
            # scaled up since a markdown block is usually smaller than a PDF page.
            if (i + 1) % (self.chunk_size * 10) == 0 or (i + 1) == len(blocks):
                if chunk_text := " ".join(current_chunk).strip():
                    count += 1
                    yield chunk_text
                current_chunk.clear()

        logger.info(f"Extracted {count} chunks from PDF.")

    def _process_epub(self, file_path: Path) -> Iterator[str]:
        try:
            book = epub.read_epub(str(file_path))
        except Exception as exc:  # ebooklib raises untyped errors on malformed EPUBs
            raise ExtractionError(f"Could not read EPUB {file_path.name}: {exc}") from exc

        current_chunk: list[str] = []
        count = 0

        def documents_in_reading_order() -> Iterator[epub.EpubHtml]:
            # The spine is the reading order; book.get_items() is manifest order and can scramble
            # chapters. EpubNav (the table of contents) and EpubCoverHtml both report ITEM_DOCUMENT,
            # so skip them explicitly — they are navigation/cover, not reading content.
            yielded = False
            for idref, _linear in book.spine:
                item = book.get_item_with_id(idref)
                if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
                    continue
                if isinstance(item, (epub.EpubNav, epub.EpubCoverHtml)):
                    continue
                yielded = True
                yield item
            if not yielded:
                # Empty or unreadable spine: fall back to manifest order so nothing is lost.
                logger.warning("EPUB spine empty/unreadable; falling back to manifest order.")
                for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                    if not isinstance(item, (epub.EpubNav, epub.EpubCoverHtml)):
                        yield item

        def extract_epub_text() -> Iterator[str]:
            for item in documents_in_reading_order():
                text = _soup_to_text(item.get_body_content(), _EPUB_DROP_TAGS)
                if text:
                    yield text

        for chapter_idx, chapter_text in enumerate(extract_epub_text(), start=1):
            current_chunk.append(self._clean_text(chapter_text))

            if chapter_idx % self.chunk_size == 0:
                if chunk_text := " ".join(current_chunk).strip():
                    count += 1
                    yield chunk_text
                current_chunk.clear()

        # Yield the remainder.
        if current_chunk and (chunk_text := " ".join(current_chunk).strip()):
            count += 1
            yield chunk_text

        logger.info(f"Extracted {count} chunks from EPUB.")

    def _process_html_dir(self, dir_path: Path) -> Iterator[str]:
        html_files = sorted(dir_path.glob("*.html"), key=natural_sort_key)
        logger.info(f"HTML directory: found {len(html_files)} file(s) in sorted order.")

        current_chunk: list[str] = []
        count = 0
        file_idx = 0

        def extract_html_text() -> Iterator[str]:
            for i, html_file in enumerate(html_files):
                logger.debug(f"  Reading [{i + 1}/{len(html_files)}]: {html_file.name}")
                with open(html_file, encoding="utf-8", errors="replace") as f:
                    text = _soup_to_text(f.read(), _HTML_DIR_DROP_TAGS)
                if text:
                    yield text

        for chapter_text in extract_html_text():
            file_idx += 1
            current_chunk.append(self._clean_text(chapter_text))

            if file_idx % self.chunk_size == 0:
                if chunk_text := " ".join(current_chunk).strip():
                    count += 1
                    yield chunk_text
                current_chunk.clear()

        # Yield the remainder.
        if current_chunk and (chunk_text := " ".join(current_chunk).strip()):
            count += 1
            yield chunk_text

        logger.info(f"Extracted {count} chunks from {file_idx}/{len(html_files)} HTML files.")
