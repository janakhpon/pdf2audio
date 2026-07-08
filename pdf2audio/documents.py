"""Document-set resolution and identity.

Turns the configured source (a file, a directory, or a list) into the ordered list of
documents to process, and computes the content+config hash that keys each document's
resumable state. Centralizes the natural-sort, discovery, and hashing logic that was
previously duplicated across ``__main__``, ``extractor``, and ``merge``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pdf2audio.config import Config
from pdf2audio.logger import logger

# A directory of these is processed as a single document; a directory of *.html is too.
SUPPORTED_DOC_SUFFIXES = {".pdf", ".epub"}

_READ_CHUNK = 65536


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort key so ``chapter_2`` precedes ``chapter_10`` (numeric runs compare as numbers)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", path.name)]


def discover_documents(source: Path | list[Path]) -> list[Path]:
    """Resolve the configured source(s) into an ordered list of documents to process.

    A directory containing ``*.html`` is treated as one document (a book split across
    files); otherwise its ``*.pdf``/``*.epub`` files are each a document. Missing paths
    are logged and skipped rather than aborting the whole run.
    """
    sources = source if isinstance(source, list) else [source]
    documents: list[Path] = []
    for sp in sources:
        if not sp.exists():
            logger.warning(f"Source path does not exist: {sp}")
            continue
        if sp.is_dir():
            if list(sp.glob("*.html")):
                documents.append(sp)
            else:
                documents.extend(sorted(sp.glob("*.pdf")))
                documents.extend(sorted(sp.glob("*.epub")))
        elif sp.is_file() and sp.suffix.lower() in SUPPORTED_DOC_SUFFIXES:
            documents.append(sp)
    return documents


def document_hash(doc_path: Path, config: Config) -> str:
    """Content+config hash keying the resumable state DB (changing either restarts work).

    Uses MD5 purely as a fast cache key — not for security. For an HTML directory the
    filenames + mtimes stand in for content so a large book isn't fully re-read each run.
    """
    hasher = hashlib.md5()
    config_state = (
        f"{config.audio_voice}_{config.audio_speed}_{config.editor_model}_"
        f"{config.editor_mode}_{config.editor_enabled}"
    )
    hasher.update(config_state.encode("utf-8"))

    if doc_path.is_dir():
        for html_file in sorted(doc_path.glob("*.html"), key=natural_sort_key):
            hasher.update(html_file.name.encode("utf-8"))
            hasher.update(str(html_file.stat().st_mtime).encode("utf-8"))
    else:
        with open(doc_path, "rb") as fh:
            while buf := fh.read(_READ_CHUNK):
                hasher.update(buf)
    return hasher.hexdigest()
