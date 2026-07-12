"""Load and validate config.yaml into a typed Config, failing fast on bad values."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml

from pdf2audio.errors import ConfigError


@dataclass
class Config:
    source_path: Path | list[Path]
    chunk_size: int
    editor_enabled: bool
    editor_model: str
    editor_mode: str
    editor_preserve_context: bool
    editor_url: str
    editor_timeout: int
    audio_model_path: str
    audio_voices_path: str
    audio_voice: str
    audio_speed: float
    audio_format: str
    out_audio_dir: Path
    out_transcripts_dir: Path
    save_transcripts: bool
    # Ollama context window (tokens), held constant for the whole run. Optional; defaults below.
    editor_num_ctx: int = 8192


_PACKAGE_ROOT = Path(__file__).parent.parent

SUPPORTED_AUDIO_FORMATS = {"mp3", "m4a", "wav"}
VALID_EDITOR_MODES = {"short", "medium", "full"}
MAX_CHUNK_SIZE = 10_000  # guard against an accidental huge value causing OOM at extraction
MIN_NUM_CTX = 512  # below this the model can't hold a chunk + its rewrite


def load_config(config_path: str | Path | None = None) -> Config:
    """Load the YAML config and return a validated Config, or raise ConfigError.

    Validation is fail-fast on values only (format, bounds, mode, URL). Model-file
    existence is checked by the audio engine, since `merge`/`preview` also load config.
    """
    if config_path is None:
        config_path = _PACKAGE_ROOT / "config.yaml"
    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping (key: value)")

    raw_source = data.get("source", {}).get("path", "books/")
    parsed_source = (
        [Path(p) for p in raw_source] if isinstance(raw_source, list) else Path(raw_source)
    )

    audio_format = str(data.get("audio", {}).get("format", "mp3")).lower()
    if audio_format not in SUPPORTED_AUDIO_FORMATS:
        raise ConfigError(
            f"unsupported audio format '{audio_format}'; choose from "
            f"{sorted(SUPPORTED_AUDIO_FORMATS)}"
        )

    try:
        chunk_size = int(data.get("source", {}).get("chunk_size", 5))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"chunk_size must be an integer: {exc}") from exc
    if not 1 <= chunk_size <= MAX_CHUNK_SIZE:
        raise ConfigError(f"chunk_size must be between 1 and {MAX_CHUNK_SIZE}, got {chunk_size}")

    editor_mode = str(data.get("editor", {}).get("mode", "full")).lower()
    if editor_mode not in VALID_EDITOR_MODES:
        raise ConfigError(
            f"editor.mode '{editor_mode}' invalid; choose from {sorted(VALID_EDITOR_MODES)}"
        )

    try:
        editor_timeout = int(data.get("editor", {}).get("timeout", 600))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"editor.timeout must be an integer (seconds): {exc}") from exc
    if editor_timeout <= 0:
        raise ConfigError(f"editor.timeout must be > 0, got {editor_timeout}")

    try:
        editor_num_ctx = int(data.get("editor", {}).get("num_ctx", 8192))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"editor.num_ctx must be an integer: {exc}") from exc
    if editor_num_ctx < MIN_NUM_CTX:
        raise ConfigError(f"editor.num_ctx must be >= {MIN_NUM_CTX}, got {editor_num_ctx}")

    try:
        audio_speed = float(data.get("audio", {}).get("speed", 1.0))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"audio.speed must be a number: {exc}") from exc
    if not 0.5 <= audio_speed <= 2.0:
        # kokoro-onnx asserts 0.5 <= speed <= 2.0; a higher value would fail every chunk at
        # synthesis time, so reject it here with a clear message instead.
        raise ConfigError(f"audio.speed must be between 0.5 and 2.0, got {audio_speed}")

    editor_url = os.getenv(
        "OLLAMA_URL", data.get("editor", {}).get("url", "http://localhost:11434")
    )
    editor_enabled = bool(data.get("editor", {}).get("enabled", False))
    if editor_enabled:
        parsed_url = urlparse(editor_url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise ConfigError(f"editor.url must be an http(s) URL, got '{editor_url}'")

    return Config(
        source_path=parsed_source,
        chunk_size=chunk_size,
        editor_enabled=editor_enabled,
        editor_model=data.get("editor", {}).get("model", "qwen2.5:14b"),
        editor_mode=editor_mode,
        editor_preserve_context=data.get("editor", {}).get("preserve_context", True),
        editor_url=editor_url,
        editor_timeout=editor_timeout,
        editor_num_ctx=editor_num_ctx,
        audio_model_path=data.get("audio", {}).get("model_path", "assets/models/kokoro-v1.0.onnx"),
        audio_voices_path=data.get("audio", {}).get("voices_path", "assets/models/voices-v1.0.bin"),
        audio_voice=data.get("audio", {}).get("voice", "af_heart"),
        audio_speed=audio_speed,
        audio_format=audio_format,
        out_audio_dir=Path(data.get("output", {}).get("audio_dir", "output/audio")),
        out_transcripts_dir=Path(
            data.get("output", {}).get("transcripts_dir", "output/transcripts")
        ),
        save_transcripts=data.get("output", {}).get("save_transcripts", True),
    )
