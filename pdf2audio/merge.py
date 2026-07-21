"""Assemble the per-chunk audio into a single file with ffmpeg's concat demuxer."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

from pdf2audio import documents
from pdf2audio.config import Config
from pdf2audio.errors import MergeError
from pdf2audio.logger import logger

# Normalize the finished audiobook to a consistent loudness so volume does not drift between
# chunks/chapters. Targets audiobook norms (~ -19 LUFS integrated, -1.5 dBTP peak; ACX allows
# -18 to -23 LUFS). Single-pass loudnorm is approximate but plenty for spoken-word narration.
_LOUDNORM_FILTER = "loudnorm=I=-19:TP=-1.5:LRA=11"

# The merge re-encodes the whole book, so its runtime scales with total audio length. A fixed
# ceiling killed long audiobooks mid-merge, so derive the subprocess timeout from the input size.
# Measured on a typical machine: loudnorm processes input wav at ~4 MB/s, a plain re-encode ~30 MB/s
# (loudnorm is ~8x slower). We use conservative (slower) rates plus a floor for small books.
_MERGE_TIMEOUT_FLOOR = 600
_LOUDNORM_BYTES_PER_SEC = 1_500_000
_ENCODE_BYTES_PER_SEC = 6_000_000


def _timeout_for(total_bytes: int, bytes_per_sec: int) -> int:
    """Subprocess timeout (seconds) for re-encoding ``total_bytes`` of input at ``bytes_per_sec``,
    never below the floor."""
    return max(_MERGE_TIMEOUT_FLOOR, math.ceil(total_bytes / bytes_per_sec))


def _codec_args(output_format: str) -> list[str]:
    fmt = output_format.lower()
    if fmt == "mp3":
        return ["-c:a", "libmp3lame", "-q:a", "2"]
    if fmt == "m4a":
        return ["-c:a", "aac", "-b:a", "128k"]
    return []


def merge_audio(
    directory: str, output_format: str = "mp3", valid_files: list[str] | None = None
) -> None:
    """Concatenate a document's chunk wavs into one file. Raises MergeError on failure.

    Failing loudly matters: a swallowed merge error would let the run exit 0 with a green log
    but no ``*_full.<fmt>`` file (se-brain cli-design: non-zero on failure)."""
    if not os.path.isdir(directory):
        raise MergeError(f"Directory not found: {directory}")

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
        raise MergeError(f"No files safely validated for merge in {directory}")

    logger.info(f"Merging {len(files)} files in {directory} using ffmpeg")

    book_name = os.path.basename(os.path.normpath(directory))
    parent_dir = os.path.dirname(os.path.normpath(directory))
    output_filename = os.path.join(parent_dir, f"{book_name}_full.{output_format}")

    total_bytes = sum(os.path.getsize(f) for f in files if os.path.exists(f))
    concat_input = ["ffmpeg", "-y", "-nostdin", "-f", "concat", "-safe", "0", "-i"]
    codec = _codec_args(output_format)

    list_path = os.path.join(directory, "concat_list.txt")
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for filepath in files:
                safe_path = os.path.abspath(filepath).replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        logger.info(f"Exporting to {output_filename}")

        # Preferred: loudness-normalized (loudnorm forces a re-encode). Its timeout scales with the
        # book length instead of a fixed ceiling, so long audiobooks are not killed mid-merge.
        loud_cmd = [*concat_input, list_path, "-af", _LOUDNORM_FILTER, *codec, output_filename]
        try:
            result = subprocess.run(
                loud_cmd,
                capture_output=True,
                text=True,
                timeout=_timeout_for(total_bytes, _LOUDNORM_BYTES_PER_SEC),
                check=False,
            )
        except subprocess.TimeoutExpired:
            # Pathologically large book: loudnorm is ~8x slower than a plain re-encode. Rather than
            # fail and leave no audiobook, merge without normalization (fast) and warn the user.
            logger.warning(
                f"Loudness normalization timed out on a large book ({total_bytes // 1024**2} MB); "
                "merging without normalization. Re-run `merge` for a loudness-normalized file."
            )
            plain_cmd = [*concat_input, list_path, *codec, output_filename]
            result = subprocess.run(
                plain_cmd,
                capture_output=True,
                text=True,
                timeout=_timeout_for(total_bytes, _ENCODE_BYTES_PER_SEC),
                check=False,
            )

        if result.returncode != 0:
            raise MergeError(f"ffmpeg merge failed (exit {result.returncode}): {result.stderr}")

        logger.info("Export complete.")
    except (OSError, subprocess.SubprocessError) as exc:
        raise MergeError(f"Merge error: {exc}") from exc
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)


def merge_all(config: Config) -> None:
    """Merge each discovered document's chunk directory into one file.

    Used by the standalone ``merge`` command to re-assemble output after an interrupted run,
    reusing the same document discovery as the main pipeline. A failed merge for one document is
    logged and counted, and the command raises MergeError if any document failed so the CLI exits
    non-zero rather than falsely reporting success.
    """
    merged = 0
    failed = 0
    for doc in documents.discover_documents(config.source_path):
        book_audio_dir = config.out_audio_dir / doc.stem
        if book_audio_dir.is_dir():
            try:
                merge_audio(str(book_audio_dir), config.audio_format)
                merged += 1
            except MergeError as exc:
                failed += 1
                logger.error(f"Merge failed for {doc.stem}: {exc}")
    if merged == 0 and failed == 0:
        logger.warning("No generated output directories found to merge.")
    elif merged:
        logger.info(f"Merged {merged} document(s).")
    if failed:
        raise MergeError(f"{failed} document(s) failed to merge.")
