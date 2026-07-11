"""AudioEngine — text chunking and the streaming, atomic wav writer (kokoro is faked)."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from pdf2audio import audio
from pdf2audio.config import Config
from pdf2audio.errors import AudioError


def _make_config(tmp_path, **overrides) -> Config:
    model = tmp_path / "model.onnx"
    voices = tmp_path / "voices.bin"
    model.write_bytes(b"fake")
    voices.write_bytes(b"fake")
    base = dict(
        source_path=tmp_path,
        chunk_size=5,
        editor_enabled=False,
        editor_model="gemma3:27b",
        editor_mode="full",
        editor_preserve_context=True,
        editor_url="http://localhost:11434",
        editor_timeout=600,
        audio_model_path=str(model),
        audio_voices_path=str(voices),
        audio_voice="af_heart",
        audio_speed=1.0,
        audio_format="mp3",
        out_audio_dir=tmp_path / "audio",
        out_transcripts_dir=tmp_path / "transcripts",
        save_transcripts=True,
    )
    base.update(overrides)
    return Config(**base)


class _FakeKokoro:
    """Returns one second of silence per call; raises the empty-phoneme error for '.'."""

    def __init__(self, frames: int = 100) -> None:
        self.frames = frames

    def create(self, text, voice, speed, lang):
        if text.strip() in {".", ",", "-"}:
            raise ValueError("need at least one array to concatenate")
        return np.zeros(self.frames, dtype=np.float32), 24000


@pytest.fixture
def engine(tmp_path, monkeypatch):
    # Skip the NLTK download path; short inputs never hit the tokenizer anyway.
    monkeypatch.setattr(audio.nltk.data, "find", lambda _p: "ok")
    eng = audio.AudioEngine(_make_config(tmp_path))
    eng._kokoro = _FakeKokoro()  # bypass the lazy real-model load
    return eng


def test_missing_model_raises_audio_error(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.nltk.data, "find", lambda _p: "ok")
    cfg = _make_config(tmp_path, audio_model_path=str(tmp_path / "nope.onnx"))
    with pytest.raises(AudioError, match="model_path"):
        audio.AudioEngine(cfg)


@pytest.mark.parametrize(
    "voice,expected",
    [
        ("af_heart", "en-us"),
        ("bf_emma", "en-gb"),
        ("ef_dora", "es"),
        ("ff_siwis", "fr-fr"),
        ("jf_alpha", "ja"),
        ("zf_xiaobei", "cmn"),
        ("", "en-us"),  # empty -> default
        ("xx_unknown", "en-us"),  # unknown prefix -> default
    ],
)
def test_espeak_lang_for_voice(voice, expected):
    assert audio._espeak_lang_for_voice(voice) == expected


def test_synthesize_passes_language_derived_from_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.nltk.data, "find", lambda _p: "ok")
    eng = audio.AudioEngine(_make_config(tmp_path, audio_voice="jf_alpha"))
    captured = {}

    class _CapKokoro:
        def create(self, text, voice, speed, lang):
            captured["lang"] = lang
            return np.zeros(100, dtype=np.float32), 24000

    eng._kokoro = _CapKokoro()
    eng.generate("hello", tmp_path / "audio" / "c1")
    assert captured["lang"] == "ja"  # Japanese voice -> ja phonemization, not en-us


def test_chunk_text_keeps_short_paragraphs_whole(engine):
    assert engine._chunk_text("hello world", max_chars=200) == ["hello world"]


def test_chunk_text_skips_blank_lines(engine):
    assert engine._chunk_text("a\n\n  \nb", max_chars=200) == ["a", "b"]


def test_chunk_text_splits_long_paragraph_within_limit(engine, monkeypatch):
    monkeypatch.setattr(
        audio.nltk.tokenize,
        "sent_tokenize",
        lambda p: [s.strip() + "." for s in p.split(".") if s.strip()],
    )
    para = " ".join(f"sentence number {i} is right here." for i in range(60))
    chunks = engine._chunk_text(para, max_chars=80)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)  # every chunk respects the ceiling


def test_chunk_text_hard_splits_an_overlong_word(engine, monkeypatch):
    monkeypatch.setattr(audio.nltk.tokenize, "sent_tokenize", lambda p: [p])
    word = "x" * 500
    chunks = engine._chunk_text(word, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == word  # no characters lost


def test_chunk_text_handles_unicode_paragraph(engine, monkeypatch):
    monkeypatch.setattr(audio.nltk.tokenize, "sent_tokenize", lambda p: [p])
    para = "これはテスト文です" * 30  # CJK, no ASCII spaces, longer than max_chars
    chunks = engine._chunk_text(para, max_chars=50)
    assert chunks
    assert all(len(c) <= 50 for c in chunks)


def test_generate_streams_all_segments_to_one_wav(engine, tmp_path):
    out = tmp_path / "audio" / "chunk_0001"
    engine.generate("first line\nsecond line\nthird line", out)

    wav = out.with_suffix(".wav")
    assert wav.exists()
    assert not out.with_suffix(".wav.tmp").exists()  # temp cleaned up

    data, sr = sf.read(str(wav))
    assert sr == 24000
    assert len(data) == 3 * 100  # three segments x 100 frames, written sequentially


def test_generate_skips_unpronounceable_but_keeps_the_rest(engine, tmp_path):
    out = tmp_path / "audio" / "chunk_0002"
    engine.generate("real text\n.\nmore text", out)  # the "." segment is skipped

    data, _ = sf.read(str(out.with_suffix(".wav")))
    assert len(data) == 2 * 100  # only the two pronounceable segments


def test_generate_with_no_pronounceable_content_writes_empty_wav(engine, tmp_path):
    out = tmp_path / "audio" / "chunk_0003"
    engine.generate(".", out)

    wav = out.with_suffix(".wav")
    assert wav.exists()
    data, _ = sf.read(str(wav))
    assert len(data) == 0


def test_generate_empty_text_is_a_noop(engine, tmp_path):
    out = tmp_path / "audio" / "chunk_0004"
    engine.generate("   ", out)
    assert not out.with_suffix(".wav").exists()


def test_synthesis_error_raises_audio_error_and_cleans_temp(engine, tmp_path):
    def _boom(text, voice, speed, lang):
        raise ValueError("model exploded")

    engine._kokoro.create = _boom  # type: ignore[method-assign]
    out = tmp_path / "audio" / "chunk_0005"
    with pytest.raises(AudioError, match="synthesis failed"):
        engine.generate("some text", out)
    assert not out.with_suffix(".wav").exists()
    assert not out.with_suffix(".wav.tmp").exists()  # partial temp removed
