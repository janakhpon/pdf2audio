import sys
import gc
from pathlib import Path
from src.config import load_config
from src.logger import logger
from src.extractor import PDFExtractor
from src.editor import SmartEditor
from src.audio import AudioEngine

def process_single_pdf(pdf_path: Path, config):
    """Processes a single PDF file through the extraction, editing, and synthesis pipeline."""
    if pdf_path.suffix.lower() != ".pdf":
        logger.warning(f"Skipping non-PDF file: {pdf_path.name}")
        return

    logger.info(f"--- Starting Processing Pipeline for {pdf_path.name} ---")
    
    extractor = PDFExtractor(chunk_size=config.chunk_size)
    editor = SmartEditor(config)
    audio_engine = AudioEngine(config)
    
    # Establish output directories cleanly
    book_audio_dir = config.out_audio_dir / pdf_path.stem
    book_transcripts_dir = config.out_transcripts_dir / pdf_path.stem
    
    book_audio_dir.mkdir(parents=True, exist_ok=True)
    if config.save_transcripts:
        book_transcripts_dir.mkdir(parents=True, exist_ok=True)

    chunks_processed = 0
    for raw_text in extractor.process_file(pdf_path):
        chunks_processed += 1
        
        transcript_out = book_transcripts_dir / f"chunk_{chunks_processed:03d}.txt"
        audio_out = book_audio_dir / f"chunk_{chunks_processed:03d}"
        final_audio_path = audio_out.with_suffix(f".{config.audio_format}")

        if final_audio_path.exists() and (not config.save_transcripts or transcript_out.exists()):
            logger.info(f"Chunk {chunks_processed} already processed. Skipping.")
            if config.save_transcripts and config.editor_preserve_context:
                try:
                    with open(transcript_out, "r", encoding="utf-8") as f:
                        saved_text = f.read().strip()
                        if config.editor_mode in ("short", "medium"):
                            editor._previous_context = saved_text
                        else:
                            editor._previous_context = saved_text[-800:] if len(saved_text) > 800 else saved_text
                except Exception as e:
                    logger.warning(f"Failed to read context from {transcript_out.name}: {e}")
            if 'raw_text' in locals(): del raw_text
            gc.collect()
            continue

        logger.info(f"Processing chunk {chunks_processed}")
        try:
            # 1. Editor Step
            polished_text = editor.process_transcript(raw_text)
            
            # 2. Transcription Saving Step
            if config.save_transcripts:
                with open(transcript_out, "w", encoding="utf-8") as f:
                    f.write(polished_text)
                logger.info(f"Saved text transcript to: {transcript_out}")
                
            # 3. Audio Synthesis Step
            audio_engine.generate(polished_text, output_path=audio_out)

        except Exception as e:
            logger.error(f"Critical error processing chunk {chunks_processed}. Skipping. Details: {e}")
            
        finally:
            # 4. Explicit Garbage Collection for Massive PDFs
            # Deleting the variables ensures memory stays completely flat (O(1)) across thousands of pages.
            if 'raw_text' in locals(): del raw_text
            if 'polished_text' in locals(): del polished_text
            gc.collect()
        
    if chunks_processed == 0:
        logger.error(f"Failed to extract meaningful text from {pdf_path.name}.")
        
    logger.info(f"--- Completed Pipeline for {pdf_path.name} ---")

def main():
    try:
        config = load_config("config.yaml")
        logger.info("Successfully loaded system configuration.")
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {e}")
        sys.exit(1)

    source_input = config.source_path
    source_paths = source_input if isinstance(source_input, list) else [source_input]

    pdf_files_to_process = []
    for sp in source_paths:
        if not sp.exists():
            logger.error(f"Source path not found: {sp}")
            continue
            
        if sp.is_dir():
            pdf_files_to_process.extend(list(sp.glob("*.pdf")))
        elif sp.is_file() and sp.suffix.lower() == ".pdf":
            pdf_files_to_process.append(sp)
        else:
            logger.warning(f"Skipping invalid source path: {sp}")

    if not pdf_files_to_process:
        logger.error("No valid PDF files found to process.")
        sys.exit(1)
        
    logger.info(f"Discovered {len(pdf_files_to_process)} target PDF(s) to process.")
    for pdf_file in pdf_files_to_process:
        process_single_pdf(pdf_file, config)

    logger.info("All pipeline operations concluded successfully.")

if __name__ == "__main__":
    main()
