"""SmartEditor: passthrough when off, context slicing, prompt building, graceful degradation,
and the happy-path Ollama call (with the network fully mocked)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest
from pdf2audio.config import Config
from pdf2audio.editor import SmartEditor
from pdf2audio.errors import EditorError


def make_config(**overrides) -> Config:
    """Build a Config dataclass directly; override only the fields a test cares about."""
    defaults = dict(
        source_path=Path("books/"),
        chunk_size=5,
        editor_enabled=True,
        editor_model="gemma3:27b",
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


@pytest.mark.parametrize("mode", ["short", "medium", "full"])
def test_build_prompt_forbids_fabrication(mode):
    prompt = SmartEditor(make_config(editor_mode=mode))._build_prompt()
    assert "FIDELITY" in prompt
    # the old "ground every concept with real-world examples" invent-content directive is gone
    assert "Ground every concept" not in prompt


def test_build_prompt_explains_previous_context_only_when_enabled():
    on = SmartEditor(make_config(editor_preserve_context=True))._build_prompt()
    off = SmartEditor(make_config(editor_preserve_context=False))._build_prompt()
    assert "PREVIOUS_CONTEXT" in on
    assert "PREVIOUS_CONTEXT" not in off


def test_build_prompt_full_mode_handles_structured_input():
    prompt = SmartEditor(make_config(editor_mode="full"))._build_prompt()
    assert "STRUCTURED INPUT" in prompt  # TOC/code/list narration rule
    # summary modes do not carry the full-mode structured-input rule
    assert "STRUCTURED INPUT" not in SmartEditor(make_config(editor_mode="short"))._build_prompt()


def test_build_prompt_full_mode_has_spoken_listen_rules():
    prompt = SmartEditor(make_config(editor_mode="full"))._build_prompt()
    assert "SPOKEN, NOT VISUAL" in prompt
    # names the visual-only things to naturalize/omit rather than read aloud
    for cue in ("page numbers", "Figure", "table", "citation"):
        assert cue in prompt
    # a concrete before->after example (few-shot) and the audiobook conventions
    assert "should be narrated simply" in prompt
    assert "AUDIOBOOK CONVENTIONS" in prompt
    for cue in ("Chapter 3", "copyright", "ISBN"):
        assert cue in prompt


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


@pytest.mark.parametrize("mode", ["short", "medium", "full"])
def test_build_prompt_forbids_preamble_and_meta(mode):
    prompt = SmartEditor(make_config(editor_mode=mode))._build_prompt()
    assert "NO PREAMBLE" in prompt
    assert "NO META-COMMENTARY" in prompt
    assert "Okay" in prompt  # names the specific filler openers to avoid
    assert "and nothing else" in prompt


# --------------------------------------------------------------------------- _strip_artifacts


@pytest.mark.parametrize(
    "raw",
    [
        "Okay, here's a breakdown of the provided text. Hash tables store key-value pairs.",
        "Sure! Here is the rewritten transcript. Hash tables store key-value pairs.",
        "Here's a breakdown of the chapter: Hash tables store key-value pairs.",
        "Here begins the chapter transcript. Hash tables store key-value pairs.",
        "Summary of Key Concepts from the Text: Hash tables store key-value pairs.",
        "Okay. Here is a breakdown of the text. Hash tables store key-value pairs.",
    ],
)
def test_strip_artifacts_removes_leading_preamble(raw):
    editor = SmartEditor(make_config())
    out = editor._strip_artifacts(raw)
    assert out.startswith("Hash tables store key-value pairs.")
    assert "breakdown" not in out.lower()
    assert "here begins" not in out.lower()


def test_strip_artifacts_removes_period_terminated_meta_opener():
    editor = SmartEditor(make_config())
    out = editor._strip_artifacts(
        "Analysis of the provided code and its relation to performance. Hash tables are fast."
    )
    assert out.startswith("Hash tables are fast.")


def test_strip_artifacts_keeps_legitimate_analysis_sentence():
    # "Analysis of variance..." has no meta cue (the text/the provided/key concepts) — keep it.
    editor = SmartEditor(make_config())
    clean = "Analysis of variance is a statistical method used to compare group means."
    assert editor._strip_artifacts(clean) == clean


@pytest.mark.parametrize(
    "narration",
    [
        "Right triangles form the basis of Euclidean geometry.",
        "Certainly the most important development was the transistor.",
        "Absolutely essential to this argument is the notion of scarcity.",
        "Right now we turn to recursion.",
        "Overview of the market shows three trends: growth, saturation, decline.",
    ],
)
def test_strip_artifacts_preserves_legit_narration_openers(narration):
    # Discourse-marker words and "Overview of ..." can legitimately open real narration; the
    # stripper must only fire on interjections (marker + punctuation) and cue-confirmed headings.
    editor = SmartEditor(make_config())
    assert editor._strip_artifacts(narration) == narration


def test_strip_artifacts_removes_doubled_signoff():
    editor = SmartEditor(make_config())
    out = editor._strip_artifacts(
        "Hash tables store pairs. I hope this helps! Let me know if you need anything else."
    )
    assert out == "Hash tables store pairs."


def test_strip_artifacts_removes_trailing_signoff():
    editor = SmartEditor(make_config())
    out = editor._strip_artifacts("Hash tables store key-value pairs. I hope this helps!")
    assert out == "Hash tables store key-value pairs."


def test_strip_artifacts_leaves_clean_narration_untouched():
    editor = SmartEditor(make_config())
    clean = "Hash tables store key-value pairs, offering constant-time lookups on average."
    assert editor._strip_artifacts(clean) == clean


def test_strip_artifacts_never_returns_empty():
    editor = SmartEditor(make_config())
    # A response that is nothing but preamble must not collapse to an empty string.
    assert editor._strip_artifacts("Okay, sure.") != ""


def test_process_transcript_strips_preamble_from_model_output(monkeypatch):
    editor = SmartEditor(make_config(editor_enabled=True))
    editor._validated = True
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(
            {"message": {"content": "Okay, here's the transcript. The real narration."}}
        ),
    )
    assert editor.process_transcript("raw") == "The real narration."


# --------------------------------------------------------------------------- num_ctx (per run)


def test_num_ctx_comes_from_config():
    assert SmartEditor(make_config()).num_ctx == 8192  # default
    assert SmartEditor(make_config(editor_num_ctx=16384)).num_ctx == 16384  # override


def test_num_ctx_is_constant_across_chunks(monkeypatch):
    # num_ctx must not change between calls, or Ollama reloads the model each time.
    editor = SmartEditor(make_config(editor_enabled=True))
    editor._validated = True
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(json.loads(req.data.decode("utf-8"))["options"]["num_ctx"])
        return _FakeResponse({"message": {"content": "ok."}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    editor.process_transcript("a short chunk")
    editor.process_transcript("a much, much longer chunk " * 200)
    assert seen[0] == seen[1] == editor.num_ctx  # same window regardless of chunk size


def test_payload_sets_keep_alive_and_context_window(monkeypatch):
    editor = SmartEditor(make_config(editor_enabled=True))
    editor._validated = True
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"message": {"content": "polished"}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    editor.process_transcript("some document text to polish")

    payload = captured["payload"]
    assert payload["keep_alive"]  # model stays resident between chunks
    assert payload["options"]["num_ctx"] == editor.num_ctx
    assert 0.0 <= payload["options"]["temperature"] <= 1.0


# --------------------------------------------------------------------------- completeness guard


def test_full_mode_falls_back_to_raw_when_polish_collapses(monkeypatch):
    editor = SmartEditor(make_config(editor_mode="full"))
    editor._validated = True
    raw = "word " * 100  # 100 words of source
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse({"message": {"content": "a tiny summary."}}),
    )
    # 3 words of 100 (3%) is a collapse (< the collapse floor) -> use complete raw text
    assert editor.process_transcript(raw) == raw
    assert editor.last_degraded is True


def test_full_mode_falls_back_to_raw_on_output_truncation(monkeypatch):
    editor = SmartEditor(make_config(editor_mode="full"))
    editor._validated = True
    raw = "word " * 50
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(
            {"message": {"content": "word " * 50}, "done_reason": "length"}
        ),
    )
    assert editor.process_transcript(raw) == raw  # truncated output rejected in favor of raw
    assert editor.last_degraded is True


def test_full_mode_falls_back_to_raw_when_prompt_overflows(monkeypatch):
    editor = SmartEditor(make_config(editor_mode="full"))
    editor._validated = True
    editor.num_ctx = 8  # tiny window so any real prompt overflows
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return _FakeResponse({"message": {"content": "x"}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    raw = "some source text that will not fit in the window"
    assert editor.process_transcript(raw) == raw
    assert editor.last_degraded is True
    assert calls["n"] == 0  # never sent a request that would silently truncate the source


def test_full_mode_keeps_a_faithful_full_length_polish(monkeypatch):
    editor = SmartEditor(make_config(editor_mode="full"))
    editor._validated = True
    raw = "word " * 100
    polished = "polished word " * 90  # comparable length -> faithful, kept
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse({"message": {"content": polished}}),
    )
    assert editor.process_transcript(raw) == polished.strip()
    assert editor.last_degraded is False


def test_full_mode_keeps_a_cleaned_shorter_polish(monkeypatch):
    # Normal cleanup makes the narration shorter than the raw (dropped page-noise); that is kept
    # and used, not discarded — only a collapse below the floor falls back.
    editor = SmartEditor(make_config(editor_mode="full"))
    editor._validated = True
    raw = "word " * 100
    polished = "clean narration word " * 40  # ~50% of the source words -> kept
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse({"message": {"content": polished}}),
    )
    assert editor.process_transcript(raw) == polished.strip()
    assert editor.last_degraded is False


@pytest.mark.parametrize("mode", ["short", "medium"])
def test_summary_modes_are_not_subject_to_the_ratio_guard(mode, monkeypatch):
    # A short/medium summary is meant to be much shorter than the source; it must be kept.
    editor = SmartEditor(make_config(editor_mode=mode))
    editor._validated = True
    raw = "word " * 100
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse({"message": {"content": "a short summary."}}),
    )
    assert editor.process_transcript(raw) == "a short summary."
    assert editor.last_degraded is False


def test_strip_artifacts_keeps_content_after_a_filler_opener():
    editor = SmartEditor(make_config())
    # the interjection is removed, the real sentence it introduces is kept
    assert (
        editor._strip_artifacts("Right, the hypotenuse is a side.") == "the hypotenuse is a side."
    )
    assert editor._strip_artifacts("Sure, the derivation follows.") == "the derivation follows."


# --------------------------------------------------------------------------- graceful degradation


def test_last_degraded_false_on_success(monkeypatch):
    editor = SmartEditor(make_config(editor_enabled=True))
    editor._validated = True
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse({"message": {"content": "clean narration."}}),
    )
    assert editor.process_transcript("x") == "clean narration."
    assert editor.last_degraded is False


def test_ensure_ready_degrades_and_flags_subsequent_chunks(monkeypatch):
    editor = SmartEditor(make_config(editor_enabled=True))

    def _boom() -> None:
        raise EditorError("Ollama down")

    monkeypatch.setattr(editor, "validate_environment", _boom)
    assert editor.ensure_ready() is False
    assert editor.enabled is False
    # a chunk that arrives after the editor disabled itself still counts as degraded
    assert editor.process_transcript("some text") == "some text"
    assert editor.last_degraded is True


def test_timeout_retried_at_most_once(monkeypatch):
    editor = SmartEditor(make_config(editor_enabled=True))
    editor._validated = True
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise TimeoutError("too slow")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("pdf2audio.editor.time.sleep", lambda *_: None)

    result = editor.process_transcript("text to polish")
    assert result == "text to polish"  # degrades to unpolished
    assert editor.last_degraded is True
    assert calls["n"] == 2  # initial attempt + exactly one retry (not _MAX_RETRIES)


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
        # comparable length to the input so the completeness guard keeps it (this test is about
        # the request/response plumbing, not completeness)
        return _FakeResponse({"message": {"content": "the polished document text"}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = editor.process_transcript("raw document text")
    assert result == "the polished document text"

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
