import sys
import re
import sqlite3
import hashlib
import shutil
import queue
import threading
import time
from pathlib import Path

from src.config import load_config
from src.logger import logger
from src.extractor import DocumentExtractor
from src.editor import SmartEditor
from src.audio import AudioEngine
from src.merge import merge_audio


def _sanitize_for_tts(text: str) -> str:
    """Strip residual markdown the LLM may emit despite instructions."""
    text = re.sub(r'\*+', '', text)                          # asterisks
    text = re.sub(r'#+\s*', '', text)                        # headings
    text = re.sub(r'`+', '', text)                           # code ticks
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)   # [link](url) → link text
    text = re.sub(r'-{2,}', ',', text)                       # em-dashes → comma pause
    text = re.sub(r'_{2,}', '', text)                        # double underscores
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_document_hash(doc_path: Path, config):
    hasher = hashlib.md5()
    
    config_state = f"{config.audio_voice}_{config.audio_speed}_{config.editor_model}_{config.editor_mode}_{config.editor_enabled}"
    hasher.update(config_state.encode('utf-8'))
    
    if doc_path.is_dir():
        import re
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s.name)]
        for f in sorted(list(doc_path.glob("*.html")), key=natural_sort_key):
            hasher.update(f.name.encode('utf-8'))
            hasher.update(str(f.stat().st_mtime).encode('utf-8'))
    else:
        with open(doc_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
    return hasher.hexdigest()

def process_single_document(doc_path: Path, config):
    if doc_path.is_file() and doc_path.suffix.lower() not in [".pdf", ".epub"]:
        return

    logger.info(f"Processing: {doc_path.name}")
    doc_hash = get_document_hash(doc_path, config)
    db_conn = sqlite3.connect(config.out_audio_dir / f"pdf2audio_state_{doc_hash}.db", check_same_thread=False)
    db_conn.execute("PRAGMA journal_mode=WAL")      # Safe for multi-thread reader/writer
    db_conn.execute("PRAGMA synchronous=NORMAL")    # Balance durability vs. speed

    c = db_conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chunks
                 (pdf_hash TEXT, chunk_idx INTEGER, status TEXT, PRIMARY KEY(pdf_hash, chunk_idx))''')
    c.execute("UPDATE chunks SET status='PENDING' WHERE pdf_hash=? AND status='PROCESSING'", (doc_hash,))
    db_conn.commit()
    
    extractor = DocumentExtractor(chunk_size=config.chunk_size)
    editor = SmartEditor(config)
    audio_engine = AudioEngine(config)
    
    book_audio_dir = config.out_audio_dir / doc_path.stem
    book_transcripts_dir = config.out_transcripts_dir / doc_path.stem
    
    book_audio_dir.mkdir(parents=True, exist_ok=True)
    if config.save_transcripts:
        book_transcripts_dir.mkdir(parents=True, exist_ok=True)

    chunks_processed = 0
    job_queue = queue.Queue(maxsize=3) # Limit memory by capping buffered LLM chunks waiting for TTS
    
    def tts_worker():
        c = db_conn.cursor()
        while True:
            job = job_queue.get()
            if job is None:
                break

            chunk_idx, text, audio_out = job
            max_tts_retries = 3
            for attempt in range(max_tts_retries):
                try:
                    audio_engine.generate(text, audio_out)
                    c.execute("UPDATE chunks SET status='DONE' WHERE pdf_hash=? AND chunk_idx=?", (doc_hash, chunk_idx))
                    db_conn.commit()
                    break
                except Exception as e:
                    if attempt == max_tts_retries - 1:
                        c.execute("UPDATE chunks SET status='FAILED' WHERE pdf_hash=? AND chunk_idx=?", (doc_hash, chunk_idx))
                        db_conn.commit()
                        logger.error(f"TTS failed after {max_tts_retries} attempts for chunk {chunk_idx}: {e}. Marking as FAILED.")
                    else:
                        delay = 2 ** attempt
                        logger.warning(f"TTS error (Attempt {attempt + 1}/{max_tts_retries}): {e}. Retrying in {delay}s...")
                        time.sleep(delay)
            job_queue.task_done()

    worker_thread = threading.Thread(target=tts_worker, daemon=True)
    worker_thread.start()
    
    c = db_conn.cursor()
    for raw_text in extractor.process_file(doc_path):
        free_bytes = shutil.disk_usage(config.out_audio_dir).free
        if free_bytes < (500 * 1024 * 1024):
            logger.error(f"CRITICAL: Disk space below 500MB ({free_bytes / 1024**2:.2f}MB). Halting extraction gracefully.")
            # Clear queue and kill gracefully
            while not job_queue.empty():
                job_queue.get()
                job_queue.task_done()
            job_queue.put(None)
            worker_thread.join()
            db_conn.close()
            sys.exit(1)
            
        chunks_processed += 1
        
        c.execute("SELECT status FROM chunks WHERE pdf_hash=? AND chunk_idx=?", (doc_hash, chunks_processed))
        row = c.fetchone()
        
        transcript_out = book_transcripts_dir / f"chunk_{chunks_processed:04d}.txt"
        audio_out = book_audio_dir / f"chunk_{chunks_processed:04d}"
        final_audio_path = audio_out.with_suffix(f".wav")

        # Check existing successful runs
        if row and row[0] == 'DONE' and final_audio_path.exists() and (not config.save_transcripts or transcript_out.exists()):
            if config.save_transcripts and config.editor_preserve_context:
                try:
                    with open(transcript_out, "r", encoding="utf-8") as f:
                        saved_text = f.read().strip()
                        editor._previous_context = saved_text if config.editor_mode in ("short", "medium") else editor._safe_slice_context(saved_text)
                except Exception as e:
                    logger.warning(f"Failed to read context: {e}")
            continue

        try:
            c.execute("INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)", 
                     (doc_hash, chunks_processed, 'PROCESSING'))
            db_conn.commit()
            
            polished_text = editor.process_transcript(raw_text)
            clean_text = _sanitize_for_tts(polished_text)

            if config.save_transcripts:
                with open(transcript_out, "w", encoding="utf-8") as f:
                    f.write(clean_text)

            job_queue.put((chunks_processed, clean_text, audio_out))

        except Exception as e:
            c.execute("UPDATE chunks SET status='FAILED' WHERE pdf_hash=? AND chunk_idx=?", (doc_hash, chunks_processed))
            db_conn.commit()
            logger.error(f"Error fetching transcript for chunk {chunks_processed}: {e}. Marking as FAILED and continuing.")
            
    # Drain the final TTS chunks
    job_queue.put(None)
    worker_thread.join()
        
    if chunks_processed == 0:
        logger.error(f"Failed to extract text from {doc_path.name}.")
    else:
        c.execute("SELECT COUNT(*) FROM chunks WHERE pdf_hash=? AND status!='DONE'", (doc_hash,))
        pending = c.fetchone()[0]
        if pending == 0:
            logger.info(f"All chunks processed for {doc_path.name}. Initiating auto-merge...")
            
            # Explicit chronological list extraction for Safe-Merge
            c.execute("SELECT chunk_idx FROM chunks WHERE pdf_hash=? AND status='DONE' ORDER BY chunk_idx", (doc_hash,))
            valid_indices = [row[0] for row in c.fetchall()]
            valid_files = [str(book_audio_dir / f"chunk_{idx:04d}.wav") for idx in valid_indices]
            
            merge_audio(str(book_audio_dir), config.audio_format, valid_files=valid_files)
        else:
            logger.warning(f"Some chunks failed for {doc_path.name} ({pending} failed/processing). Skipping automatic merge.")
            
    db_conn.close()
    logger.info(f"Completed: {doc_path.name}")

def main():
    if not shutil.which("ffmpeg"):
        logger.error("FFmpeg is required but not installed. Please run `brew install ffmpeg` or download it from ffmpeg.org.")
        sys.exit(1)
        
    try:
        config = load_config("config.yaml")
        logger.info("Configuration loaded.")
        config.out_audio_dir.mkdir(parents=True, exist_ok=True)
        
        free_bytes = shutil.disk_usage(config.out_audio_dir).free
        free_gb = free_bytes / (1024**3)
        if free_gb < 5.0:
            logger.warning(f"Low disk space detected: {free_gb:.2f} GB free. Audio extraction and merging may fail.")
        else:
            logger.info(f"Disk check passed: {free_gb:.2f} GB available.")
            
    except Exception as e:
        logger.error(f"Config error: {e}")
        sys.exit(1)

    source_input = config.source_path
    source_paths = source_input if isinstance(source_input, list) else [source_input]

    doc_files = []
    for sp in source_paths:
        if not sp.exists():
            continue
        if sp.is_dir():
            if list(sp.glob("*.html")):
                doc_files.append(sp)
            else:
                doc_files.extend(list(sp.glob("*.pdf")))
                doc_files.extend(list(sp.glob("*.epub")))
        elif sp.is_file() and sp.suffix.lower() in [".pdf", ".epub"]:
            doc_files.append(sp)

    if not doc_files:
        logger.error("No valid PDF, EPUB, or HTML directory found.")
        sys.exit(1)
        
    logger.info(f"Discovered {len(doc_files)} document(s).")
    for doc_file in doc_files:
        process_single_document(doc_file, config)

if __name__ == "__main__":
    main()
