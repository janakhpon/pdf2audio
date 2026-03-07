import sys
import argparse
import os
import glob
import re
import subprocess
from pathlib import Path
from src.logger import logger
from src.config import load_config

def merge_audio(directory: str, output_format: str = "mp3", valid_files: list[str] = None):
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return

    if valid_files is not None:
        files = valid_files # Already strict and chronologically ordered by the DB
    else:
        # Fallback to legacy globbing if strict list not provided (e.g standalone module run)
        pattern = os.path.join(directory, f"chunk_*.wav")
        def natural_sort_key(s):
            match = re.search(r'\d+', os.path.basename(s))
            return int(match.group()) if match else 0
        files = sorted(glob.glob(pattern), key=natural_sort_key)

    if not files:
        logger.error(f"No files safely validated for merge in {directory}")
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
            "ffmpeg", "-y", "-nostdin", "-f", "concat", "-safe", "0", 
            "-i", list_path
        ]
        
        if output_format.lower() == "mp3":
            command.extend(["-c:a", "libmp3lame", "-q:a", "2"])
        elif output_format.lower() == "m4a":
            command.extend(["-c:a", "aac", "-b:a", "128k"])
            
        command.append(output_filename)
        
        # Safe subprocess execution with timeout and child handling
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            timeout=600,  # Failsafe abort after 10 minutes to prevent process hangs
            check=False # Check manually below
        )
        
        if result.returncode != 0:
            logger.error(f"ffmpeg merge failed: {result.stderr}")
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
        logger.error("No target document files found in config to derive merge paths.")
        sys.exit(1)

    merged_count = 0
    for doc_file in doc_files:
        book_audio_dir = config.out_audio_dir / doc_file.stem
        if book_audio_dir.exists() and book_audio_dir.is_dir():
            merge_audio(str(book_audio_dir), config.audio_format)
            merged_count += 1
            
    if merged_count == 0:
        logger.warning("No generated output directories found to merge.")

if __name__ == "__main__":
    main()
