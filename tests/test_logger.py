"""logger — level control and the stderr stream contract (cli-design)."""

from __future__ import annotations

import logging

from pdf2audio import logger as logger_mod

# The stdout-vs-stderr contract (results on stdout, logs on stderr) is verified end-to-end
# in test_cli.py; asserting stream identity here is brittle under pytest's stream capture.


def test_uses_a_stream_handler():
    assert any(isinstance(h, logging.StreamHandler) for h in logger_mod.logger.handlers)


def test_formatter_is_structured_with_timestamp_level_message():
    handler = next(h for h in logger_mod.logger.handlers if isinstance(h, logging.StreamHandler))
    fmt = handler.formatter._fmt  # type: ignore[union-attr]
    assert "%(asctime)s" in fmt
    assert "%(levelname)" in fmt
    assert "%(message)s" in fmt


def test_set_level_accepts_names():
    original = logger_mod.logger.level
    try:
        logger_mod.set_level("DEBUG")
        assert logger_mod.logger.level == logging.DEBUG
        logger_mod.set_level("warning")  # case-insensitive
        assert logger_mod.logger.level == logging.WARNING
    finally:
        logger_mod.logger.setLevel(original)


def test_message_is_emitted_with_level(caplog):
    with caplog.at_level(logging.INFO, logger="pdf2audio"):
        logger_mod.logger.info("hello from the pipeline")
    assert "hello from the pipeline" in caplog.text
    assert any(r.levelno == logging.INFO for r in caplog.records)
