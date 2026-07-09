"""Command-line interface — the only transport adapter over the core library.

Parses argv into a ``Config`` + intent and calls into the core (``pipeline``, ``merge``,
``preview``); it holds no domain logic. Follows the CLI contract: results go to stdout,
logs/diagnostics to stderr, and typed errors map to exit codes (0 ok, 1 failure, 130 on
Ctrl-C). Installed as the ``pdf2audio`` console command.
"""

from __future__ import annotations

import argparse
import shutil
import sys

from pdf2audio import __version__, documents, logger, merge, pipeline, preview
from pdf2audio.config import Config, load_config
from pdf2audio.errors import PDF2AudioError

_LOW_DISK_WARN_GB = 5.0
_log = logger.logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2audio",
        description="Convert PDFs, EPUBs, and HTML books into audiobooks, fully offline.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Shared flags, inherited by every subcommand via `parents=`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="path to the YAML config file (default: config.yaml)",
    )
    common.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log verbosity on stderr (default: INFO)",
    )

    sub = parser.add_subparsers(dest="command", metavar="{run,preview,merge}")
    run = sub.add_parser("run", parents=[common], help="run the full pipeline over the source(s)")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="list the documents that would be processed, then exit without processing",
    )
    sub.add_parser("preview", parents=[common], help="synthesize a short sample of the voice")
    sub.add_parser("merge", parents=[common], help="re-merge chunk audio from a prior run")
    return parser


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise PDF2AudioError(
            "FFmpeg is required but not installed. Run `brew install ffmpeg` or see ffmpeg.org."
        )


def _load(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    _log.info("Configuration loaded.")
    return config


def _warn_low_disk(config: Config) -> None:
    free_gb = shutil.disk_usage(config.out_audio_dir).free / (1024**3)
    if free_gb < _LOW_DISK_WARN_GB:
        _log.warning(f"Low disk space: {free_gb:.2f} GB free. Extraction/merge may fail.")
    else:
        _log.info(f"Disk check passed: {free_gb:.2f} GB available.")


def _cmd_run(args: argparse.Namespace) -> None:
    config = _load(args)
    doc_files = documents.discover_documents(config.source_path)
    if not doc_files:
        raise PDF2AudioError("No valid PDF, EPUB, or HTML directory found.")

    if args.dry_run:
        # A dry run has no side effects and no prerequisites (no ffmpeg, no output dir).
        _log.info(f"Dry run: {len(doc_files)} document(s) would be processed.")
        for doc in doc_files:
            print(doc)  # the result goes to stdout
        return

    _require_ffmpeg()  # needed for the final merge step
    config.out_audio_dir.mkdir(parents=True, exist_ok=True)
    _warn_low_disk(config)
    _log.info(f"Discovered {len(doc_files)} document(s).")
    for doc in doc_files:
        pipeline.process_document(doc, config)


def _cmd_preview(args: argparse.Namespace) -> None:
    config = _load(args)
    output_path = preview.preview_voice(config)
    print(output_path)  # the result goes to stdout


def _cmd_merge(args: argparse.Namespace) -> None:
    _require_ffmpeg()
    config = _load(args)
    merge.merge_all(config)


_COMMANDS = {"run": _cmd_run, "preview": _cmd_preview, "merge": _cmd_merge}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(2)

    logger.set_level(args.log_level)

    try:
        _COMMANDS[args.command](args)
    except PDF2AudioError as exc:
        _log.error(str(exc))
        sys.exit(1)
    except OSError as exc:
        # e.g. output dir not writable, disk error — report cleanly, don't dump a traceback.
        _log.error(f"I/O error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        _log.warning("Interrupted. Progress is saved; re-run to resume where it stopped.")
        sys.exit(130)


if __name__ == "__main__":
    main()
