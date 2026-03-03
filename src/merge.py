import sys
import argparse
import os
import glob
import subprocess
from pathlib import Path
from src.logger import logger
from src.config import load_config

def merge_audio(directory: str, output_format: str = "mp3"):
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return

    pattern = os.path.join(directory, f"chunk_*.{output_format}")
    files = sorted(glob.glob(pattern))

    if not files:
        logger.error(f"No files found in {directory}")
        return

    logger.info(f"Merging {len(files)} files in {directory} using ffmpeg")

    book_name = os.path.basename(os.path.normpath(directory))
    parent_dir = os.path.dirname(os.path.normpath(directory))
    output_filename = os.path.join(parent_dir, f"{book_name}_full.{output_format}")

    list_path = os.path.join(directory, "concat_list.txt")
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for filepath in files:
                safe_path = os.path.abspath(filepath).replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        logger.info(f"Exporting to {output_filename}")
        
        command = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", list_path, "-c", "copy", output_filename
        ]
        
        import signal
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        def handle_sigterm(signum, frame):
            logger.warning("Received termination sequence. Killing ffmpeg subprocess.")
            process.terminate()
            process.wait()
            sys.exit(1)
            
        signal.signal(signal.SIGTERM, handle_sigterm)
        signal.signal(signal.SIGINT, handle_sigterm)
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"ffmpeg merge failed: {stderr}")
            return
            
        logger.info("Export complete.")
    except Exception as e:
        logger.error(f"Merge error: {e}")
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
def main():
    try:
        config = load_config("config.yaml")
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
        logger.error("No target PDFs found in config to derive merge paths.")
        sys.exit(1)

    merged_count = 0
    for pdf_file in pdf_files:
        book_audio_dir = config.out_audio_dir / pdf_file.stem
        if book_audio_dir.exists() and book_audio_dir.is_dir():
            merge_audio(str(book_audio_dir), config.audio_format)
            merged_count += 1
            
    if merged_count == 0:
        logger.warning("No generated output directories found to merge.")

if __name__ == "__main__":
    main()
