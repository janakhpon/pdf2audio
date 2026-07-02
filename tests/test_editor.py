"""SmartEditor: passthrough when off, context slicing, prompt building, graceful degradation,
and the happy-path Ollama call (with the network fully mocked)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from src.config import Config
from src.editor import SmartEditor
from src.errors import EditorError


def make_config(**overrides) -> Config:
    """Build a Config dataclass directly; override only the fields a test cares about."""
    defaults = dict(
        source_path=Path("books/"),
        chunk_size=5,
        editor_enabled=True,
        editor_model="llama3.2",
        editor_mode="full",
        editor_preserve_context=True,
        editor_url="http://localhost:11434",
        editor_timeout=600,
        audio_model_path="assets/models/kokoro-v1.0.onnx",
        audio_voices_path="assets/models/voices-v1.0.bin",
        audio_voice="af_heart",
        audio_speed=1.0,
        audio_format="mp3",
        out_audio_dir=Path("output/audio"),
        out_transcripts_dir=Path("output/transcripts"),
        save_transcripts=True,
    )
    defaults.update(overrides)
    return Config(**defaults)


class _FakeResponse:
    """Minimal stand-in for the object returned by urllib.request.urlopen (context manager)."""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc) -> None:
        return None


# --------------------------------------------------------------------------- passthrough


def test_disabled_editor_returns_text_unchanged():
    editor = SmartEditor(make_config(editor_enabled=False))
    text = "The quick brown fox."
    assert editor.process_transcript(text) is text


def test_blank_text_returned_unchanged():
    editor = SmartEditor(make_config(editor_enabled=True))
    assert editor.process_transcript("   ") == "   "


# --------------------------------------------------------------------------- _safe_slice_context


def test_safe_slice_passthrough_when_short():
    editor = SmartEditor(make_config())
    text = "short enough"
    assert editor._safe_slice_context(text, max_chars=2000) == text


def test_safe_slice_trims_to_word_boundary():
    editor = SmartEditor(make_config())
    # 50 chars, keep last 20; the raw slice starts mid-word and must be trimmed forward
    # to the first whitespace boundary.
    text = "alpha beta gamma delta epsilon zeta eta theta iota"
    result = editor._safe_slice_context(text, max_chars=20)
    assert len(result) <= 20
    # The trimmed result should begin at a clean word (no leading partial token / whitespace).
    assert not result[0].isspace()
    assert result in text
    # It starts strictly after where the raw slice began (boundary trim happened).
    assert result != text[-20:]


def test_safe_slice_no_boundary_returns_stripped():
    editor = SmartEditor(make_config())
    text = "x" * 100  # no whitespace/period/newline at all
    result = editor._safe_slice_context(text, max_chars=10)
    assert result == "x" * 10


# --------------------------------------------------------------------------- load_saved_context


def test_load_saved_context_noop_when_preserve_disabled():
    editor = SmartEditor(make_config(editor_preserve_context=False))
    editor.load_saved_context("some earlier text")
    assert editor._previous_context is None


@pytest.mark.parametrize("mode", ["short", "medium"])
def test_load_saved_context_stores_whole_string_short_medium(mode):
    editor = SmartEditor(make_config(editor_mode=mode))
    saved = "a" * 5000  # longer than the 2000-char full-mode slice window
    editor.load_saved_context(saved)
    assert editor._previous_context == saved


def test_load_saved_context_slices_in_full_mode():
    editor = SmartEditor(make_config(editor_mode="full"))
    saved = "word " * 1000  # ~5000 chars, exceeds the 2000 slice window
    editor.load_saved_context(saved)
    assert editor._previous_context is not None
    assert len(editor._previous_context) <= 2000
    assert len(editor._previous_context) < len(saved)


# --------------------------------------------------------------------------- _build_prompt


@pytest.mark.parametrize("mode", ["short", "medium", "full"])
def test_build_prompt_has_language_and_no_markdown_constraints(mode):
    editor = SmartEditor(make_config(editor_mode=mode))
    prompt = editor._build_prompt()
    assert "same language" in prompt  # language constraint
    assert "NEVER use asterisks" in prompt  # no-markdown / formatting constraint
    assert "markdown" in prompt.lower()


def test_build_prompt_differs_per_mode():
    prompts = {
        mode: SmartEditor(make_config(editor_mode=mode))._build_prompt()
        for mode in ("short", "medium", "full")
    }
    assert prompts["short"] != prompts["medium"]
    assert prompts["medium"] != prompts["full"]
    assert prompts["short"] != prompts["full"]
    assert "summary" in prompts["short"].lower()
    assert "complete audiobook chapter transcript" in prompts["full"]


# --------------------------------------------------------------------------- graceful degradation


def test_process_transcript_degrades_when_validate_fails(monkeypatch):
    editor = SmartEditor(make_config(editor_enabled=True))

    def _boom() -> None:
        raise EditorError("Ollama not reachable")

    monkeypatch.setattr(editor, "validate_environment", _boom)

    original = "leave me exactly as I am"
    result = editor.process_transcript(original)
    assert result == original
    assert editor.enabled is False


# --------------------------------------------------------------------------- happy path


def test_process_transcript_happy_path(monkeypatch):
    editor = SmartEditor(make_config(editor_enabled=True))
    editor._validated = True  # skip the reachability check; we only exercise /api/chat

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"message": {"content": "polished"}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = editor.process_transcript("raw document text")
    assert result == "polished"

    # Posted to the chat endpoint.
    assert captured["url"].endswith("/api/chat")

    messages = captured["payload"]["messages"]
    roles = {m["role"]: m["content"] for m in messages}
    # Instructions belong in the system role; the document goes in the user role.
    assert "system" in roles and "user" in roles
    assert "NEVER use asterisks" in roles["system"]
    assert "raw document text" in roles["user"]
    assert "TEXT TO PROCESS:" in roles["user"]
    # The document text must NOT leak into the system instructions.
    assert "raw document text" not in roles["system"]
    assert captured["payload"]["stream"] is False


def test_process_transcript_includes_previous_context_in_user_message(monkeypatch):
    editor = SmartEditor(make_config(editor_enabled=True, editor_mode="full"))
    editor._validated = True
    editor._previous_context = "PRIOR NARRATIVE"

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"message": {"content": "polished"}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    editor.process_transcript("current chunk")

    user_msg = next(m["content"] for m in captured["payload"]["messages"] if m["role"] == "user")
    assert "PREVIOUS_CONTEXT" in user_msg
    assert "PRIOR NARRATIVE" in user_msg


def test_process_transcript_updates_context_after_success(monkeypatch):
    editor = SmartEditor(make_config(editor_enabled=True, editor_mode="short"))
    editor._validated = True

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse({"message": {"content": "the polished output"}}),
    )
    editor.process_transcript("input")
    # short mode stores the whole polished string as the next context.
    assert editor._previous_context == "the polished output"


def test_validate_environment_noop_when_disabled(monkeypatch):
    editor = SmartEditor(make_config(editor_enabled=False))

    def _should_not_be_called(*a, **k):  # pragma: no cover - guard
        raise AssertionError("network must not be touched when editor is disabled")

    monkeypatch.setattr(urllib.request, "urlopen", _should_not_be_called)
    editor.validate_environment()  # must return without raising / calling the network
