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


def sanitize_for_tts(text: str) -> str:
    """Strip residual markdown the LLM may emit despite instructions, before synthesis."""
    text = re.sub(r"\*+", "", text)  # asterisks
    text = re.sub(r"#+\s*", "", text)  # headings
    text = re.sub(r"`+", "", text)  # code ticks
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [link](url) -> link text
    text = re.sub(r"-{2,}", ",", text)  # em-dashes -> comma pause
    text = re.sub(r"_{2,}", "", text)  # double underscores
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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

    book_audio_dir = config.out_audio_dir / doc_path.stem
    book_transcripts_dir = config.out_transcripts_dir / doc_path.stem
    book_audio_dir.mkdir(parents=True, exist_ok=True)
    if config.save_transcripts:
        book_transcripts_dir.mkdir(parents=True, exist_ok=True)

    chunks_processed = 0
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
                    polished_text = editor.process_transcript(raw_text)
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
