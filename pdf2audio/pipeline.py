"""The core pipeline: extract -> polish -> synthesize -> merge, for one document.

Orchestrates the modules around the resumable :class:`ChunkStateStore`. Extraction and
LLM polish run on the calling thread; a single daemon worker does TTS, fed through a
bounded queue so audio synthesis overlaps with reading the next chunk without letting
polished chunks pile up in memory.

This is library code: it takes a ``Config`` and raises typed errors. It never calls
``sys.exit`` or touches argv — mapping outcomes to exit codes is the CLI's job.
"""

from __future__ import annotations

import queue
import re
import shutil
import threading
import time
from pathlib import Path

from pdf2audio import documents
from pdf2audio.audio import AudioEngine
from pdf2audio.config import Config
from pdf2audio.editor import SmartEditor
from pdf2audio.errors import PDF2AudioError
from pdf2audio.extractor import DocumentExtractor
from pdf2audio.logger import logger
from pdf2audio.merge import merge_audio
from pdf2audio.state import ChunkStateStore, ChunkStatus

# Halt extraction if free disk drops below this mid-run (audio output can be large).
MIN_FREE_BYTES = 500 * 1024 * 1024
_MAX_TTS_RETRIES = 3
_QUEUE_MAXSIZE = 3  # cap buffered chunks awaiting TTS


def is_structural_noise(text: str) -> bool:
    """True for a table-of-contents / index / list-of-figures page: dominated by dot leaders and
    page numbers, which have no spoken value. Such chunks are skipped (not narrated). Ordinary
    prose never has several dot-leader runs, so this does not fire on real content."""
    return len(re.findall(r"\.{4,}", text)) >= 3


def sanitize_for_tts(text: str) -> str:
    """Turn residual visual/markup cruft into clean spoken text before synthesis.

    This is the last line of defence for both the polished output and the raw-text fallback
    (see SmartEditor). It only removes things that are unambiguously non-content on a page —
    placeholders, markup symbols, table scaffolding, bare URLs. Context-dependent choices
    (figure/page references, citation numbers, table meaning) are left to the LLM.
    """
    # Order matters: strip docling's HTML-comment placeholders (<!-- image -->,
    # <!-- formula-not-decoded -->, <!-- missing-* -->) BEFORE the dash rule, or their "--" gets
    # turned into commas and voiced as garbage.
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    # Fenced code blocks (```lang ... ```): verbatim source is unlistenable, so drop the whole
    # block. Must run BEFORE the lone-backtick rule below, which would otherwise strip the fences
    # and leave the code body (and its language tag) as bare spoken text.
    text = re.sub(r"`{3,}[^\n]*\n?.*?`{3,}", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # ![alt](url) image -> drop (incl. the !)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [link](url) -> link text
    # Bare citation/footnote markers ("Smith [12]" -> "Smith"). Require a leading space so attached
    # array indices (a[0], arr[5]) and non-numeric brackets ([i]) are left untouched.
    text = re.sub(r"\s+\[\d+\]", "", text)
    text = re.sub(r"https?://\S+", " ", text)  # bare URL -> drop (annoying to hear spelled out)
    # Bare hex literals (0x162f5a33) have no spoken value.
    text = re.sub(r"\b0x[0-9a-fA-F]+\b", " ", text)
    # Table-of-contents / index dot leaders and their page numbers ("Title..... 80"). 4+ dots so a
    # normal "..." ellipsis is left alone.
    text = re.sub(r"\.{4,}\s*\d*", " ", text)
    # Pipe tables: drop separator rows, then read remaining cells as a short spoken list.
    text = re.sub(r"(?m)^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$", " ", text)  # |---|:--:| separator rows
    text = re.sub(r"\s*\|\s*", ", ", text)  # cell dividers -> comma pause
    text = re.sub(r"\*+", "", text)  # bold/italic asterisks
    text = re.sub(r"#+\s*", "", text)  # headings
    text = re.sub(r"`+", "", text)  # code ticks
    text = re.sub(r"~~", "", text)  # strikethrough
    text = re.sub(r"\${1,2}", "", text)  # $ / $$ formula delimiters
    text = re.sub(r"\\[()\[\]]", "", text)  # LaTeX \( \) \[ \] math delimiters (keep the content)
    text = re.sub(r"_{2,}", "", text)  # double underscores
    text = re.sub(r"[ \t]*(?:[—–]|-{2,})[ \t]*", ", ", text)  # em/en dash or -- -> comma pause
    text = re.sub(r"\s+", " ", text)
    # Tidy punctuation the substitutions can leave behind (e.g. a stray " , " where a rule/dash
    # between blocks was dropped).
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)  # no space before punctuation
    text = re.sub(r"([.!?])[,;:]+", r"\1", text)  # drop a separator orphaned after sentence-end
    text = re.sub(r"[,;:](?=[,;:])", "", text)  # collapse runs of separators
    return text.strip(" ,;:")


def process_document(doc_path: Path, config: Config) -> None:
    """Run the resumable pipeline for one document (file or HTML directory).

    Per-chunk failures are isolated: the chunk is marked FAILED and the run continues.
    Running low on disk mid-run aborts the whole run — raised as ``PDF2AudioError`` so the
    caller (CLI) can stop and exit non-zero, after the TTS worker has drained cleanly.
    """
    if doc_path.is_file() and doc_path.suffix.lower() not in documents.SUPPORTED_DOC_SUFFIXES:
        return

    logger.info(f"Processing: {doc_path.name}")
    doc_hash = documents.document_hash(doc_path, config)
    db_path = config.out_audio_dir / f"pdf2audio_state_{doc_hash}.db"

    extractor = DocumentExtractor(chunk_size=config.chunk_size)
    editor = SmartEditor(config)
    audio_engine = AudioEngine(config)

    # Validate the editor before the (slow) extraction so a down/misconfigured Ollama surfaces
    # immediately instead of after a multi-minute PDF conversion. Degrades, never crashes.
    if config.editor_enabled and not editor.ensure_ready():
        logger.warning("Editor unavailable; this book will be narrated from unpolished text.")

    book_audio_dir = config.out_audio_dir / doc_path.stem
    book_transcripts_dir = config.out_transcripts_dir / doc_path.stem
    book_audio_dir.mkdir(parents=True, exist_ok=True)
    if config.save_transcripts:
        book_transcripts_dir.mkdir(parents=True, exist_ok=True)

    chunks_processed = 0
    degraded_chunks = 0
    skipped_chunks = 0
    halted_low_disk = False
    job_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)

    with ChunkStateStore(db_path) as store:
        store.reset_stale(doc_hash)

        def tts_worker() -> None:
            while True:
                job = job_queue.get()
                if job is None:
                    break
                chunk_idx, text, audio_out = job
                for attempt in range(_MAX_TTS_RETRIES):
                    try:
                        audio_engine.generate(text, audio_out)
                        store.mark(doc_hash, chunk_idx, ChunkStatus.DONE)
                        break
                    except Exception:
                        # A background worker must never die silently; retry then mark FAILED.
                        if attempt == _MAX_TTS_RETRIES - 1:
                            store.mark(doc_hash, chunk_idx, ChunkStatus.FAILED)
                            logger.exception(
                                f"TTS failed after {_MAX_TTS_RETRIES} attempts for chunk "
                                f"{chunk_idx}; marking FAILED"
                            )
                        else:
                            delay = 2**attempt
                            logger.warning(
                                f"TTS error (attempt {attempt + 1}/{_MAX_TTS_RETRIES}); "
                                f"retrying in {delay}s"
                            )
                            time.sleep(delay)
                job_queue.task_done()

        worker_thread = threading.Thread(target=tts_worker, name="tts-worker", daemon=True)
        worker_thread.start()

        try:
            for raw_text in extractor.process_file(doc_path):
                free_bytes = shutil.disk_usage(config.out_audio_dir).free
                if free_bytes < MIN_FREE_BYTES:
                    logger.error(
                        f"Disk space below {MIN_FREE_BYTES // 1024**2} MB "
                        f"({free_bytes / 1024**2:.0f} MB free); halting extraction."
                    )
                    halted_low_disk = True
                    break

                chunks_processed += 1
                status = store.status(doc_hash, chunks_processed)

                transcript_out = book_transcripts_dir / f"chunk_{chunks_processed:04d}.txt"
                audio_out = book_audio_dir / f"chunk_{chunks_processed:04d}"
                final_audio_path = audio_out.with_suffix(".wav")

                already_done = (
                    status == ChunkStatus.DONE
                    and final_audio_path.exists()
                    and (not config.save_transcripts or transcript_out.exists())
                )
                if already_done:
                    if config.save_transcripts and config.editor_preserve_context:
                        try:
                            editor.load_saved_context(
                                transcript_out.read_text(encoding="utf-8").strip()
                            )
                        except OSError as exc:
                            logger.warning(
                                f"Could not restore context from {transcript_out.name}: {exc}"
                            )
                    continue

                try:
                    store.mark(doc_hash, chunks_processed, ChunkStatus.PROCESSING)
                    if is_structural_noise(raw_text):
                        # A table-of-contents / index page: not narratable, skip it (empty audio).
                        logger.info(
                            f"Chunk {chunks_processed}: table-of-contents/index page; not narrated."
                        )
                        skipped_chunks += 1
                        clean_text = ""
                    else:
                        polished_text = editor.process_transcript(raw_text)
                        if editor.last_degraded:
                            degraded_chunks += 1
                        clean_text = sanitize_for_tts(polished_text)
                    if config.save_transcripts:
                        transcript_out.write_text(clean_text, encoding="utf-8")
                    job_queue.put((chunks_processed, clean_text, audio_out))
                except (PDF2AudioError, OSError):
                    store.mark(doc_hash, chunks_processed, ChunkStatus.FAILED)
                    logger.exception(
                        f"Failed to prepare chunk {chunks_processed}; marking FAILED and continuing"
                    )
        finally:
            # Always stop the worker cleanly so in-flight audio finishes and state is flushed.
            job_queue.put(None)
            worker_thread.join()

        if halted_low_disk:
            raise PDF2AudioError(f"Halted: free disk space below {MIN_FREE_BYTES // 1024**2} MB.")

        if chunks_processed == 0:
            logger.error(f"Failed to extract text from {doc_path.name}.")
            return

        if degraded_chunks:
            logger.warning(
                f"{degraded_chunks} of {chunks_processed} chunk(s) used the complete raw text "
                f"(editor unavailable, or its polish dropped/truncated content) to keep the "
                f"narration complete. Re-run once Ollama is healthy to re-polish them."
            )
        if skipped_chunks:
            logger.info(
                f"{skipped_chunks} of {chunks_processed} chunk(s) were table-of-contents/index "
                f"pages and were not narrated."
            )

        pending = store.pending_count(doc_hash)
        if pending == 0:
            logger.info(f"All chunks processed for {doc_path.name}; merging.")
            valid_files = [
                str(book_audio_dir / f"chunk_{idx:04d}.wav") for idx in store.done_indices(doc_hash)
            ]
            merge_audio(str(book_audio_dir), config.audio_format, valid_files=valid_files)
        else:
            logger.warning(
                f"{pending} chunk(s) failed/pending for {doc_path.name}; skipping merge."
            )

    logger.info(f"Completed: {doc_path.name}")
