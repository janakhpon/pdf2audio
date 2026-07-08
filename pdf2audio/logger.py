import logging
import sys


def setup_logger(name: str = "pdf2audio") -> logging.Logger:
    """
    Configures and returns a professional, standardized logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
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
