"""Opt-in end-to-end smoke test using the REAL kokoro model and ffmpeg.

Unlike the rest of the suite (which mocks the heavy dependencies), this drives the actual
pipeline and asserts real audio comes out. It auto-skips when the model files or ffmpeg are
absent, so normal/CI runs are unaffected — run it locally after touching audio/pipeline code:

    uv run pytest -m e2e          # only this
    uv run pytest -m "not e2e"    # everything else
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import soundfile as sf
from pdf2audio import pipeline
from pdf2audio.audio import AudioEngine
from pdf2audio.config import Config
from pdf2audio.preview import preview_voice

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL = _REPO_ROOT / "assets/models/kokoro-v1.0.onnx"
_VOICES = _REPO_ROOT / "assets/models/voices-v1.0.bin"

_have_models = _MODEL.is_file() and _VOICES.is_file()
_have_ffmpeg = shutil.which("ffmpeg") is not None

needs_models = pytest.mark.skipif(not _have_models, reason="kokoro model files not present")
needs_ffmpeg = pytest.mark.skipif(not _have_ffmpeg, reason="ffmpeg not installed")

pytestmark = pytest.mark.e2e


def _config(tmp_path: Path, source: Path, **overrides) -> Config:
    base = dict(
        source_path=source,
        chunk_size=1,
        editor_enabled=False,  # no Ollama dependency in the smoke test
        editor_model="gemma3:27b",
        editor_mode="full",
        editor_preserve_context=True,
        editor_url="http://localhost:11434",
        editor_timeout=600,
        audio_model_path=str(_MODEL),
        audio_voices_path=str(_VOICES),
        audio_voice="af_heart",
        audio_speed=1.0,
        audio_format="mp3",
        out_audio_dir=tmp_path / "audio",
        out_transcripts_dir=tmp_path / "transcripts",
        save_transcripts=True,
    )
    base.update(overrides)
    return Config(**base)


@needs_models
def test_preview_produces_real_audio(tmp_path):
    """AudioEngine loads the real model and preview writes a non-empty wav."""
    config = _config(tmp_path, tmp_path)
    wav = preview_voice(config)
    assert wav.exists()
    data, sr = sf.read(str(wav))
    assert sr == 24000
    assert len(data) > 0  # actual samples, not an empty file


@needs_models
@needs_ffmpeg
def test_full_pipeline_produces_and_resumes(tmp_path):
    """Extract → synthesize → ffmpeg-merge yields a playable MP3, and a re-run resumes."""
    book = tmp_path / "book"
    book.mkdir()
    (book / "chapter_1.html").write_text("<h1>One</h1><p>The river was calm at dawn.</p>")
    (book / "chapter_2.html").write_text("<h1>Two</h1><p>By noon the sun was high.</p>")
    config = _config(tmp_path, book)

    pipeline.process_document(book, config)

    merged = config.out_audio_dir / "book_full.mp3"
    assert merged.is_file() and merged.stat().st_size > 0
    chunks = sorted((config.out_audio_dir / "book").glob("chunk_*.wav"))
    assert len(chunks) == 2  # one per HTML file (chunk_size=1)

    # A second run must resume: the chunk wavs are already present and stay put.
    mtimes = {p: p.stat().st_mtime_ns for p in chunks}
    pipeline.process_document(book, config)
    assert all(p.stat().st_mtime_ns == mtimes[p] for p in chunks), "resume re-synthesized audio"


@needs_models
def test_audio_engine_streams_multi_segment_text(tmp_path):
    """A multi-line block is chunked and streamed into one wav by the real engine."""
    engine = AudioEngine(_config(tmp_path, tmp_path))
    out = tmp_path / "audio" / "seg"
    engine.generate("First line of narration.\nSecond line of narration.", out)
    wav = out.with_suffix(".wav")
    assert wav.exists()
    data, _ = sf.read(str(wav))
    assert len(data) > 0
