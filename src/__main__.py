import sys
import gc
import sqlite3
import hashlib
from pathlib import Path
from src.config import load_config
from src.logger import logger
from src.extractor import PDFExtractor
from src.editor import SmartEditor
from src.audio import AudioEngine

def init_db(config, pdf_hash: str):
    db_path = config.out_audio_dir / f"pdf2audio_state_{pdf_hash}.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chunks 
                 (pdf_hash TEXT, chunk_idx INTEGER, status TEXT, PRIMARY KEY(pdf_hash, chunk_idx))''')
    
    # Auto-revert any dirty state from a previous crashed run
    c.execute("UPDATE chunks SET status='PENDING' WHERE pdf_hash=? AND status='PROCESSING'", (pdf_hash,))
    
    conn.commit()
    return conn

def get_pdf_hash(pdf_path: Path):
    hasher = hashlib.md5()
    with open(pdf_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def process_single_pdf(pdf_path: Path, config):
    if pdf_path.suffix.lower() != ".pdf":
        return

    logger.info(f"Processing: {pdf_path.name}")
    pdf_hash = get_pdf_hash(pdf_path)
    db_conn = init_db(config, pdf_hash)
    
    extractor = PDFExtractor(chunk_size=config.chunk_size)
    editor = SmartEditor(config)
    audio_engine = AudioEngine(config)
    
    book_audio_dir = config.out_audio_dir / pdf_path.stem
    book_transcripts_dir = config.out_transcripts_dir / pdf_path.stem
    
    book_audio_dir.mkdir(parents=True, exist_ok=True)
    if config.save_transcripts:
        book_transcripts_dir.mkdir(parents=True, exist_ok=True)

    chunks_processed = 0
    c = db_conn.cursor()
    
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=1) as tts_executor:
        tts_future = None
        
        for raw_text in extractor.process_file(pdf_path):
            chunks_processed += 1
            
            c.execute("SELECT status FROM chunks WHERE pdf_hash=? AND chunk_idx=?", (pdf_hash, chunks_processed))
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
                # 1. State: Mark intent
                c.execute("INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)", 
                         (pdf_hash, chunks_processed, 'PROCESSING'))
                db_conn.commit()
                
                # 2. Sync Network Call: Retrieve polished text (blocks the main thread)
                polished_text = editor.process_transcript(raw_text)
                
                if config.save_transcripts:
                    with open(transcript_out, "w", encoding="utf-8") as f:
                        f.write(polished_text)
                        
                # 3. Wait for the PREVIOUS async audio generation finish
                if tts_future:
                    tts_future.result()
                    
                # 4. Async CPU task: Start synthesizing audio for CURRENT chunk in thread pool
                # This unblocks the next loop iteration (and the next LLM call)
                tts_future = tts_executor.submit(audio_engine.generate, polished_text, audio_out)

                # 5. State: Mark success
                c.execute("UPDATE chunks SET status='DONE' WHERE pdf_hash=? AND chunk_idx=?", (pdf_hash, chunks_processed))
                db_conn.commit()

            except Exception as e:
                c.execute("UPDATE chunks SET status='FAILED' WHERE pdf_hash=? AND chunk_idx=?", (pdf_hash, chunks_processed))
                db_conn.commit()
                logger.error(f"Error processing chunk {chunks_processed}: {e}. Marking as FAILED and continuing.")
                
        # Drain the final TTS chunk
        if tts_future:
            try:
                tts_future.result()
            except Exception as e:
                logger.error(f"Final audio generation failed: {e}")
        
    if chunks_processed == 0:
        logger.error(f"Failed to extract text from {pdf_path.name}.")
    else:
        c.execute("SELECT COUNT(*) FROM chunks WHERE pdf_hash=? AND status!='DONE'", (pdf_hash,))
        pending = c.fetchone()[0]
        if pending == 0:
            logger.info(f"All chunks processed for {pdf_path.name}. Initiating auto-merge...")
            
            # Explicit chronological list extraction for Safe-Merge
            c.execute("SELECT chunk_idx FROM chunks WHERE pdf_hash=? AND status='DONE' ORDER BY chunk_idx", (pdf_hash,))
            valid_indices = [row[0] for row in c.fetchall()]
            valid_files = [str(book_audio_dir / f"chunk_{idx:04d}.wav") for idx in valid_indices]
            
            from src.merge import merge_audio
            merge_audio(str(book_audio_dir), config.audio_format, valid_files=valid_files)
        else:
            logger.warning(f"Some chunks failed for {pdf_path.name} ({pending} failed/processing). Skipping automatic merge.")
            
    db_conn.close()
    logger.info(f"Completed: {pdf_path.name}")

def main():
    import shutil
    import sys
    
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

    pdf_files = []
    for sp in source_paths:
        if not sp.exists():
            continue
        if sp.is_dir():
            pdf_files.extend(list(sp.glob("*.pdf")))
        elif sp.is_file() and sp.suffix.lower() == ".pdf":
            pdf_files.append(sp)

    if not pdf_files:
        logger.error("No valid PDF files found.")
        sys.exit(1)
        
    logger.info(f"Discovered {len(pdf_files)} PDF(s).")
    for pdf_file in pdf_files:
        process_single_pdf(pdf_file, config)

if __name__ == "__main__":
    main()
