"""load_config: validation, bounds, defaults, and the OLLAMA_URL override."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pdf2audio.config import Config, load_config
from pdf2audio.errors import ConfigError


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(dedent(body), encoding="utf-8")
    return path


_VALID_BODY = """\
    source:
      path: books/
      chunk_size: 5
    editor:
      enabled: false
      model: gemma3:27b
      mode: full
      preserve_context: true
      url: http://localhost:11434
      timeout: 600
    audio:
      voice: af_heart
      speed: 1.0
      format: mp3
    output:
      audio_dir: output/audio
      transcripts_dir: output/transcripts
      save_transcripts: true
"""


def test_load_valid_config(tmp_path):
    cfg = load_config(_write_config(tmp_path, _VALID_BODY))
    assert isinstance(cfg, Config)
    assert cfg.chunk_size == 5
    assert cfg.editor_enabled is False
    assert cfg.editor_mode == "full"
    assert cfg.audio_format == "mp3"
    assert cfg.audio_speed == 1.0
    assert cfg.editor_url == "http://localhost:11434"
    assert isinstance(cfg.source_path, Path)


def test_editor_num_ctx_default(tmp_path):
    assert load_config(_write_config(tmp_path, _VALID_BODY)).editor_num_ctx == 8192


def test_editor_num_ctx_override(tmp_path):
    body = _VALID_BODY.replace("timeout: 600", "timeout: 600\n      num_ctx: 16384")
    assert load_config(_write_config(tmp_path, body)).editor_num_ctx == 16384


def test_editor_num_ctx_too_small_raises(tmp_path):
    body = _VALID_BODY.replace("timeout: 600", "timeout: 600\n      num_ctx: 100")
    with pytest.raises(ConfigError, match="num_ctx must be >="):
        load_config(_write_config(tmp_path, body))


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_invalid_yaml_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("key: [unterminated\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_non_mapping_root_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(path)


def test_empty_file_uses_defaults(tmp_path):
    # `yaml.safe_load` returns None on an empty file; load_config falls back to {} + defaults.
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.chunk_size == 5
    assert cfg.audio_format == "mp3"
    assert cfg.editor_mode == "full"


def test_unsupported_audio_format_raises(tmp_path):
    body = _VALID_BODY.replace("format: mp3", "format: flac")
    with pytest.raises(ConfigError, match="unsupported audio format"):
        load_config(_write_config(tmp_path, body))


@pytest.mark.parametrize("fmt", ["mp3", "m4a", "wav"])
def test_all_supported_audio_formats(tmp_path, fmt):
    body = _VALID_BODY.replace("format: mp3", f"format: {fmt}")
    assert load_config(_write_config(tmp_path, body)).audio_format == fmt


def test_audio_format_is_lowercased(tmp_path):
    body = _VALID_BODY.replace("format: mp3", "format: MP3")
    assert load_config(_write_config(tmp_path, body)).audio_format == "mp3"


@pytest.mark.parametrize("bad", [0, -1, 10001, 99999])
def test_chunk_size_out_of_range_raises(tmp_path, bad):
    body = _VALID_BODY.replace("chunk_size: 5", f"chunk_size: {bad}")
    with pytest.raises(ConfigError, match="chunk_size must be between"):
        load_config(_write_config(tmp_path, body))


@pytest.mark.parametrize("ok", [1, 5, 10000])
def test_chunk_size_within_range(tmp_path, ok):
    body = _VALID_BODY.replace("chunk_size: 5", f"chunk_size: {ok}")
    assert load_config(_write_config(tmp_path, body)).chunk_size == ok


def test_chunk_size_non_integer_raises(tmp_path):
    body = _VALID_BODY.replace("chunk_size: 5", "chunk_size: not-a-number")
    with pytest.raises(ConfigError, match="chunk_size must be an integer"):
        load_config(_write_config(tmp_path, body))


def test_invalid_editor_mode_raises(tmp_path):
    body = _VALID_BODY.replace("mode: full", "mode: gigantic")
    with pytest.raises(ConfigError, match="editor.mode"):
        load_config(_write_config(tmp_path, body))


@pytest.mark.parametrize("mode", ["short", "medium", "full"])
def test_valid_editor_modes(tmp_path, mode):
    body = _VALID_BODY.replace("mode: full", f"mode: {mode}")
    assert load_config(_write_config(tmp_path, body)).editor_mode == mode


@pytest.mark.parametrize("bad", [0, -5])
def test_editor_timeout_non_positive_raises(tmp_path, bad):
    body = _VALID_BODY.replace("timeout: 600", f"timeout: {bad}")
    with pytest.raises(ConfigError, match="editor.timeout must be > 0"):
        load_config(_write_config(tmp_path, body))


@pytest.mark.parametrize("bad", [0.4, 0.49, 2.01, 3.0, 5.0])
def test_audio_speed_out_of_range_raises(tmp_path, bad):
    body = _VALID_BODY.replace("speed: 1.0", f"speed: {bad}")
    with pytest.raises(ConfigError, match="audio.speed must be between"):
        load_config(_write_config(tmp_path, body))


@pytest.mark.parametrize("ok", [0.5, 1.0, 1.5, 2.0])
def test_audio_speed_within_range(tmp_path, ok):
    body = _VALID_BODY.replace("speed: 1.0", f"speed: {ok}")
    assert load_config(_write_config(tmp_path, body)).audio_speed == ok


def test_editor_disabled_skips_url_validation(tmp_path):
    # When the editor is off, a bogus URL must NOT raise (validation is gated on enabled).
    body = _VALID_BODY.replace("url: http://localhost:11434", "url: not-a-url")
    cfg = load_config(_write_config(tmp_path, body))
    assert cfg.editor_enabled is False
    assert cfg.editor_url == "not-a-url"


def test_editor_enabled_invalid_url_raises(tmp_path):
    body = _VALID_BODY.replace("enabled: false", "enabled: true").replace(
        "url: http://localhost:11434", "url: ftp://nope"
    )
    with pytest.raises(ConfigError, match="editor.url must be an http"):
        load_config(_write_config(tmp_path, body))


def test_editor_enabled_missing_netloc_raises(tmp_path):
    body = _VALID_BODY.replace("enabled: false", "enabled: true").replace(
        "url: http://localhost:11434", "url: http://"
    )
    with pytest.raises(ConfigError, match="editor.url must be an http"):
        load_config(_write_config(tmp_path, body))


def test_editor_enabled_valid_url_ok(tmp_path):
    body = _VALID_BODY.replace("enabled: false", "enabled: true")
    cfg = load_config(_write_config(tmp_path, body))
    assert cfg.editor_enabled is True
    assert cfg.editor_url == "http://localhost:11434"


def test_ollama_url_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://remote-host:9999")
    cfg = load_config(_write_config(tmp_path, _VALID_BODY))
    assert cfg.editor_url == "http://remote-host:9999"


def test_ollama_url_env_override_wins_over_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://env-wins:1234")
    body = _VALID_BODY.replace("url: http://localhost:11434", "url: http://config-loses:11434")
    cfg = load_config(_write_config(tmp_path, body))
    assert cfg.editor_url == "http://env-wins:1234"


def test_source_path_list(tmp_path):
    body = """\
        source:
          path:
            - books/a.pdf
            - books/b.epub
    """
    cfg = load_config(_write_config(tmp_path, body))
    assert isinstance(cfg.source_path, list)
    assert all(isinstance(p, Path) for p in cfg.source_path)
    assert len(cfg.source_path) == 2
