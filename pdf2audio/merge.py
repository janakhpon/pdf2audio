from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pdf2audio import documents
from pdf2audio.config import Config
from pdf2audio.logger import logger


def merge_audio(
    directory: str, output_format: str = "mp3", valid_files: list[str] | None = None
) -> None:
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return

    if valid_files is not None:
        files = valid_files  # Already strict and chronologically ordered by the DB
    else:
        # Fallback to globbing when a strict DB-ordered list isn't provided
        # (e.g. the standalone `merge` command).
        files = [
            str(p)
            for p in sorted(Path(directory).glob("chunk_*.wav"), key=documents.natural_sort_key)
        ]

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
            "ffmpeg",
            "-y",
            "-nostdin",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
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
            check=False,  # Check manually below
        )

        if result.returncode != 0:
            logger.error(f"ffmpeg merge failed: {result.stderr}")
            return

        logger.info("Export complete.")
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error(f"Merge error: {exc}")
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)


def merge_all(config: Config) -> int:
    """Merge each discovered document's chunk directory into one file. Returns the count merged.

    Used by the standalone ``merge`` command to re-assemble output after an interrupted run,
    reusing the same document discovery as the main pipeline.
    """
    merged = 0
    for doc in documents.discover_documents(config.source_path):
        book_audio_dir = config.out_audio_dir / doc.stem
        if book_audio_dir.is_dir():
            merge_audio(str(book_audio_dir), config.audio_format)
            merged += 1
    if merged == 0:
        logger.warning("No generated output directories found to merge.")
    return merged
