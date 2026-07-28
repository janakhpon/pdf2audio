import logging
import sys


def setup_logger(name: str = "pdf2audio") -> logging.Logger:
    """Configure and return the shared logger: timestamped, level-prefixed, on stderr."""
    logger = logging.getLogger(name)

    # Check this logger's OWN handlers, not hasHandlers() (which also sees ancestors): the CLI
    # must always attach its stderr handler, even if something else configured the root logger.
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Timestamp + level + message: enough to trace a batch run without a JSON pipeline
        # (observability: structured-enough for an offline tool; full JSON is out of scope).
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Logs go to stderr so stdout stays free for program output / piping
        # (se-brain cli-design: stdout = result, stderr = diagnostics).
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()


def set_level(level: str) -> None:
    """Set the log level from a name (DEBUG/INFO/WARNING/ERROR); used by the CLI --log-level."""
    logger.setLevel(level.upper())
