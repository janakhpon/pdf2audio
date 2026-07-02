import hashlib
import queue
import re
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path

from src.audio import AudioEngine
from src.config import Config, load_config
from src.editor import SmartEditor
from src.errors import PDF2AudioError
from src.extractor import DocumentExtractor
from src.logger import logger
from src.merge import merge_audio

_MIN_FREE_BYTES = 500 * 1024 * 1024  # halt extraction if free disk drops below this
_LOW_DISK_WARN_GB = 5.0
_MAX_TTS_RETRIES = 3
_SUPPORTED_DOC_SUFFIXES = {".pdf", ".epub"}


def _sanitize_for_tts(text: str) -> str:
    """Strip residual markdown the LLM may emit despite instructions."""
    text = re.sub(r"\*+", "", text)  # asterisks
    text = re.sub(r"#+\s*", "", text)  # headings
    text = re.sub(r"`+", "", text)  # code ticks
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [link](url) -> link text
    text = re.sub(r"-{2,}", ",", text)  # em-dashes -> comma pause
    text = re.sub(r"_{2,}", "", text)  # double underscores
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _natural_sort_key(path: Path) -> list[int | str]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", path.name)]


def get_document_hash(doc_path: Path, config: Config) -> str:
    """Content+config hash that keys the resumable state DB (changing either restarts work)."""
    hasher = hashlib.md5()
    config_state = (
        f"{config.audio_voice}_{config.audio_speed}_{config.editor_model}_"
        f"{config.editor_mode}_{config.editor_enabled}"
    )
    hasher.update(config_state.encode("utf-8"))

    if doc_path.is_dir():
        for html_file in sorted(doc_path.glob("*.html"), key=_natural_sort_key):
            hasher.update(html_file.name.encode("utf-8"))
            hasher.update(str(html_file.stat().st_mtime).encode("utf-8"))
    else:
        with open(doc_path, "rb") as fh:
            while buf := fh.read(65536):
                hasher.update(buf)
    return hasher.hexdigest()


def process_single_document(doc_path: Path, config: Config) -> None:
    if doc_path.is_file() and doc_path.suffix.lower() not in _SUPPORTED_DOC_SUFFIXES:
        return

    logger.info(f"Processing: {doc_path.name}")
    doc_hash = get_document_hash(doc_path, config)
    db_path = config.out_audio_dir / f"pdf2audio_state_{doc_hash}.db"
    db_conn = sqlite3.connect(db_path, check_same_thread=False)
    db_conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
    db_conn.execute("PRAGMA synchronous=NORMAL")  # balance durability vs. speed

    # The connection is shared by the main (extract/edit) thread and the TTS worker. WAL protects
    # the file, but a Python sqlite3 connection/cursor is not safe for concurrent use — serialize
    # every access through one lock (se-brain: concurrency / double-write prevention).
    db_lock = threading.Lock()

    def db_write(sql: str, params: tuple = ()) -> None:
        with db_lock:
            db_conn.execute(sql, params)
            db_conn.commit()

    def db_query(sql: str, params: tuple = ()) -> list:
        with db_lock:
            return db_conn.execute(sql, params).fetchall()

    db_write(
        "CREATE TABLE IF NOT EXISTS chunks "
        "(pdf_hash TEXT, chunk_idx INTEGER, status TEXT, PRIMARY KEY(pdf_hash, chunk_idx))"
    )
    db_write(
        "UPDATE chunks SET status='PENDING' WHERE pdf_hash=? AND status='PROCESSING'",
        (doc_hash,),
    )

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
    job_queue: queue.Queue = queue.Queue(maxsize=3)  # cap buffered chunks awaiting TTS

    def tts_worker() -> None:
        while True:
            job = job_queue.get()
            if job is None:
                break
            chunk_idx, text, audio_out = job
            for attempt in range(_MAX_TTS_RETRIES):
                try:
                    audio_engine.generate(text, audio_out)
                    db_write(
                        "UPDATE chunks SET status='DONE' WHERE pdf_hash=? AND chunk_idx=?",
                        (doc_hash, chunk_idx),
                    )
                    break
                except Exception:
                    if attempt == _MAX_TTS_RETRIES - 1:
                        db_write(
                            "UPDATE chunks SET status='FAILED' WHERE pdf_hash=? AND chunk_idx=?",
                            (doc_hash, chunk_idx),
                        )
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
            if free_bytes < _MIN_FREE_BYTES:
                logger.error(
                    f"Disk space below {_MIN_FREE_BYTES // 1024**2} MB "
                    f"({free_bytes / 1024**2:.0f} MB free); halting extraction."
                )
                halted_low_disk = True
                break

            chunks_processed += 1
            rows = db_query(
                "SELECT status FROM chunks WHERE pdf_hash=? AND chunk_idx=?",
                (doc_hash, chunks_processed),
            )
            status = rows[0][0] if rows else None

            transcript_out = book_transcripts_dir / f"chunk_{chunks_processed:04d}.txt"
            audio_out = book_audio_dir / f"chunk_{chunks_processed:04d}"
            final_audio_path = audio_out.with_suffix(".wav")

            already_done = (
                status == "DONE"
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
                db_write(
                    "INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)",
                    (doc_hash, chunks_processed, "PROCESSING"),
                )
                polished_text = editor.process_transcript(raw_text)
                clean_text = _sanitize_for_tts(polished_text)
                if config.save_transcripts:
                    transcript_out.write_text(clean_text, encoding="utf-8")
                job_queue.put((chunks_processed, clean_text, audio_out))
            except (PDF2AudioError, OSError):
                db_write(
                    "UPDATE chunks SET status='FAILED' WHERE pdf_hash=? AND chunk_idx=?",
                    (doc_hash, chunks_processed),
                )
                logger.exception(
                    f"Failed to prepare chunk {chunks_processed}; marking FAILED and continuing"
                )
    finally:
        # Always stop the worker cleanly so in-flight audio finishes and state is flushed.
        job_queue.put(None)
        worker_thread.join()

    if halted_low_disk:
        db_conn.close()
        sys.exit(1)

    if chunks_processed == 0:
        logger.error(f"Failed to extract text from {doc_path.name}.")
    else:
        pending = db_query(
            "SELECT COUNT(*) FROM chunks WHERE pdf_hash=? AND status!='DONE'",
            (doc_hash,),
        )[0][0]
        if pending == 0:
            logger.info(f"All chunks processed for {doc_path.name}; merging.")
            valid_indices = [
                r[0]
                for r in db_query(
                    "SELECT chunk_idx FROM chunks WHERE pdf_hash=? AND status='DONE' "
                    "ORDER BY chunk_idx",
                    (doc_hash,),
                )
            ]
            valid_files = [str(book_audio_dir / f"chunk_{idx:04d}.wav") for idx in valid_indices]
            merge_audio(str(book_audio_dir), config.audio_format, valid_files=valid_files)
        else:
            logger.warning(
                f"{pending} chunk(s) failed/pending for {doc_path.name}; skipping merge."
            )

    db_conn.close()
    logger.info(f"Completed: {doc_path.name}")


def _discover_documents(config: Config) -> list[Path]:
    source_input = config.source_path
    source_paths = source_input if isinstance(source_input, list) else [source_input]
    doc_files: list[Path] = []
    for sp in source_paths:
        if not sp.exists():
            logger.warning(f"Source path does not exist: {sp}")
            continue
        if sp.is_dir():
            if list(sp.glob("*.html")):
                doc_files.append(sp)
            else:
                doc_files.extend(sorted(sp.glob("*.pdf")))
                doc_files.extend(sorted(sp.glob("*.epub")))
        elif sp.is_file() and sp.suffix.lower() in _SUPPORTED_DOC_SUFFIXES:
            doc_files.append(sp)
    return doc_files


def main() -> None:
    if not shutil.which("ffmpeg"):
        logger.error(
            "FFmpeg is required but not installed. Run `brew install ffmpeg` or see ffmpeg.org."
        )
        sys.exit(1)

    try:
        config = load_config("config.yaml")
        logger.info("Configuration loaded.")
        config.out_audio_dir.mkdir(parents=True, exist_ok=True)
    except PDF2AudioError as exc:
        logger.error(f"Config error: {exc}")
        sys.exit(1)
    except OSError as exc:
        logger.error(f"Could not create output directory: {exc}")
        sys.exit(1)

    free_gb = shutil.disk_usage(config.out_audio_dir).free / (1024**3)
    if free_gb < _LOW_DISK_WARN_GB:
        logger.warning(f"Low disk space: {free_gb:.2f} GB free. Extraction/merge may fail.")
    else:
        logger.info(f"Disk check passed: {free_gb:.2f} GB available.")

    doc_files = _discover_documents(config)
    if not doc_files:
        logger.error("No valid PDF, EPUB, or HTML directory found.")
        sys.exit(1)

    logger.info(f"Discovered {len(doc_files)} document(s).")
    try:
        for doc_file in doc_files:
            process_single_document(doc_file, config)
    except KeyboardInterrupt:
        logger.warning("Interrupted. Progress is saved; re-run to resume where it stopped.")
        sys.exit(130)


if __name__ == "__main__":
    main()
