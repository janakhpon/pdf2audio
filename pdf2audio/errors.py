"""Typed exception hierarchy for pdf2audio.

A single base (`PDF2AudioError`) so callers can catch the family, with specific
subclasses per failure domain (config, extraction, editing/LLM, audio, merge, database).
Prefer raising these over bare `Exception`/`RuntimeError` so failures are distinguishable
and carry actionable messages (se-brain: engineering-standards error taxonomy)."""

from __future__ import annotations


class PDF2AudioError(Exception):
    """Base class for all pdf2audio errors."""


class ConfigError(PDF2AudioError):
    """Invalid or unusable configuration (bad values, missing required paths)."""


class ExtractionError(PDF2AudioError):
    """A document could not be read/extracted (unsupported, corrupt, empty)."""


class EditorError(PDF2AudioError):
    """The LLM polish step failed (Ollama unreachable, timeout, empty output)."""


class AudioError(PDF2AudioError):
    """TTS synthesis failed."""


class MergeError(PDF2AudioError):
    """Merging the audio chunks failed."""


class DatabaseError(PDF2AudioError):
    """The resumable-state database could not be read/written."""
