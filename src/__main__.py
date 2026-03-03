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

def init_db(config):
    db_path = config.out_audio_dir / "pdf2audio_state.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chunks 
                 (pdf_hash TEXT, chunk_idx INTEGER, status TEXT, PRIMARY KEY(pdf_hash, chunk_idx))''')
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

def process_single_pdf(pdf_path: Path, config, db_conn):
    if pdf_path.suffix.lower() != ".pdf":
        return

    logger.info(f"Processing: {pdf_path.name}")
    pdf_hash = get_pdf_hash(pdf_path)
    
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
    
    for raw_text in extractor.process_file(pdf_path):
        chunks_processed += 1
        
        c.execute("SELECT status FROM chunks WHERE pdf_hash=? AND chunk_idx=?", (pdf_hash, chunks_processed))
        row = c.fetchone()
        
        transcript_out = book_transcripts_dir / f"chunk_{chunks_processed:03d}.txt"
        audio_out = book_audio_dir / f"chunk_{chunks_processed:03d}"
        final_audio_path = audio_out.with_suffix(f".{config.audio_format}")

        if row and row[0] == 'DONE' and final_audio_path.exists() and (not config.save_transcripts or transcript_out.exists()):
            if config.save_transcripts and config.editor_preserve_context:
                try:
                    with open(transcript_out, "r", encoding="utf-8") as f:
                        saved_text = f.read().strip()
                        editor._previous_context = saved_text if config.editor_mode in ("short", "medium") else saved_text[-800:]
                except Exception as e:
                    logger.warning(f"Failed to read context: {e}")
            continue

        try:
            c.execute("INSERT OR REPLACE INTO chunks (pdf_hash, chunk_idx, status) VALUES (?, ?, ?)", 
                     (pdf_hash, chunks_processed, 'PROCESSING'))
            db_conn.commit()
            
            polished_text = editor.process_transcript(raw_text)
            
            if config.save_transcripts:
                with open(transcript_out, "w", encoding="utf-8") as f:
                    f.write(polished_text)
                
            audio_engine.generate(polished_text, output_path=audio_out)

            c.execute("UPDATE chunks SET status='DONE' WHERE pdf_hash=? AND chunk_idx=?", (pdf_hash, chunks_processed))
            db_conn.commit()

        except Exception as e:
            c.execute("UPDATE chunks SET status='FAILED' WHERE pdf_hash=? AND chunk_idx=?", (pdf_hash, chunks_processed))
            db_conn.commit()
            logger.error(f"Critical error processing chunk {chunks_processed}: {e}. Halting pipeline to prevent data corruption.")
            raise
        
    if chunks_processed == 0:
        logger.error(f"Failed to extract text from {pdf_path.name}.")
        
    logger.info(f"Completed: {pdf_path.name}")

def main():
    try:
        config = load_config("config.yaml")
        logger.info("Configuration loaded.")
        config.out_audio_dir.mkdir(parents=True, exist_ok=True)
        db_conn = init_db(config)
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
        process_single_pdf(pdf_file, config, db_conn)
        
    db_conn.close()

if __name__ == "__main__":
    main()
