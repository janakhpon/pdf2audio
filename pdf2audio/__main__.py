import shutil
import sys

from pdf2audio import documents, pipeline
from pdf2audio.config import load_config
from pdf2audio.errors import PDF2AudioError
from pdf2audio.logger import logger

_LOW_DISK_WARN_GB = 5.0


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

    doc_files = documents.discover_documents(config.source_path)
    if not doc_files:
        logger.error("No valid PDF, EPUB, or HTML directory found.")
        sys.exit(1)

    logger.info(f"Discovered {len(doc_files)} document(s).")
    try:
        for doc_file in doc_files:
            pipeline.process_document(doc_file, config)
    except PDF2AudioError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted. Progress is saved; re-run to resume where it stopped.")
        sys.exit(130)


if __name__ == "__main__":
    main()
