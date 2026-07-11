"""Synthesize a short sample of the configured voice so it can be auditioned before a full run."""

from __future__ import annotations

from pathlib import Path

from pdf2audio.audio import AudioEngine
from pdf2audio.config import Config
from pdf2audio.logger import logger

_SAMPLE_TEXT = "This is a sample of my voice. I will be your narrator."


def preview_voice(config: Config) -> Path:
    """Synthesize a short sample with the configured voice and return the output path."""
    logger.info(f"Previewing voice: {config.audio_voice}")
    engine = AudioEngine(config)
    config.out_audio_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.out_audio_dir / f"_preview_{config.audio_voice}"
    engine.generate(_SAMPLE_TEXT, output_path=output_path)
    return output_path.with_suffix(".wav")
