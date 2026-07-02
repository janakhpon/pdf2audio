import json
import re
import time
import urllib.error
import urllib.request

from src.config import Config
from src.errors import EditorError
from src.logger import logger

_MAX_RETRIES = 3
_BASE_DELAY = 2.0
_VALIDATE_TIMEOUT = 5  # seconds — quick reachability check before the (slow) polish calls


class SmartEditor:
    """Optional Ollama-backed text polish. Degrades to the unpolished text on any failure
    (unreachable, timeout, empty output) so a chunk is never lost — see docs/adr and AUDIT."""

    def __init__(self, config: Config) -> None:
        self.model = config.editor_model
        self.mode = config.editor_mode
        self.url_chat = f"{config.editor_url}/api/chat"
        self.url_tags = f"{config.editor_url}/api/tags"
        self.enabled = config.editor_enabled
        self.preserve_context = config.editor_preserve_context
        self.timeout = config.editor_timeout
        self._previous_context: str | None = None
        self._validated = False

    def _safe_slice_context(self, text: str, max_chars: int = 2000) -> str:
        """Return at most the last `max_chars`, trimmed to start at a word/sentence boundary."""
        if len(text) <= max_chars:
            return text
        sliced = text[-max_chars:]
        match = re.search(r"[\s.\n]", sliced)
        if match:
            return sliced[match.end() :].strip()
        return sliced.strip()

    def load_saved_context(self, saved_text: str) -> None:
        """Restore narrative context from a previously-saved transcript (used on resume)."""
        if not self.preserve_context:
            return
        self._previous_context = (
            saved_text if self.mode in ("short", "medium") else self._safe_slice_context(saved_text)
        )

    def validate_environment(self) -> None:
        """Check Ollama is reachable and the model is pulled. Raises EditorError otherwise."""
        if not self.enabled:
            return
        try:
            req = urllib.request.Request(self.url_tags)
            with urllib.request.urlopen(req, timeout=_VALIDATE_TIMEOUT) as response:
                tags_data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EditorError(
                f"Ollama not reachable at {self.url_tags} ({exc}). Is `ollama serve` running?"
            ) from exc
        except json.JSONDecodeError as exc:
            raise EditorError(f"Ollama returned invalid JSON from {self.url_tags}: {exc}") from exc

        models = [m.get("name") for m in tags_data.get("models", [])]
        if self.model not in models and f"{self.model}:latest" not in models:
            raise EditorError(
                f"Ollama model '{self.model}' not pulled. Run `ollama pull {self.model}`."
            )
        self._validated = True
        logger.info(f"Ollama environment validated (model: {self.model})")

    def process_transcript(self, text: str) -> str:
        """Return polished text, or the original text unchanged if polishing is off/unavailable."""
        if not self.enabled or not text.strip():
            return text

        if not self._validated:
            try:
                self.validate_environment()
            except EditorError as exc:
                # Degrade for the rest of the run rather than failing every chunk.
                logger.warning(f"{exc} Continuing with unpolished text.")
                self.enabled = False
                return text

        logger.info(f"Polishing transcript (model: {self.model}, mode: {self.mode})")
        context_block = ""
        if self.preserve_context and self._previous_context:
            context_block = f"<PREVIOUS_CONTEXT>\n{self._previous_context}\n</PREVIOUS_CONTEXT>\n\n"

        # Instructions stay in the system role; the document text goes in the user role so it
        # cannot override the instructions (se-brain ai-orchestration: input boundaries).
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_prompt()},
                {"role": "user", "content": f"{context_block}TEXT TO PROCESS:\n{text}"},
            ],
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")

        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    self.url_chat,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                polished = str(result.get("message", {}).get("content", "")).strip()
                if polished:
                    if self.preserve_context:
                        self._previous_context = (
                            polished
                            if self.mode in ("short", "medium")
                            else self._safe_slice_context(polished)
                        )
                    return polished
                logger.warning(f"Ollama returned empty output (attempt {attempt + 1}).")
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                if attempt == _MAX_RETRIES - 1:
                    break
                delay = _BASE_DELAY * (2**attempt)
                logger.warning(
                    f"Ollama error (attempt {attempt + 1}/{_MAX_RETRIES}): {exc}. "
                    f"Retrying in {delay}s."
                )
                time.sleep(delay)

        logger.error("Ollama polishing failed; using the unpolished text for this chunk.")
        return text

    def _build_prompt(self) -> str:
        lang_constraint = (
            "CRITICAL: Do NOT translate the text. "
            "You MUST respond in the exact same language as the original text."
        )
        purpose = (
            "PURPOSE: You are preparing a chapter of an audiobook. "
            "The output will be narrated by a text-to-speech voice and listened to, "
            "never read visually. "
            "Every word you write will be spoken aloud to the listener."
        )
        formatting_constraint = (
            "CRITICAL FORMATTING RULES: "
            "Because this is an audiobook, NEVER use asterisks, hashes, bullet points, "
            "dashes, numbered lists, markdown, or any other visual formatting symbols. "
            "Write only in full, flowing prose sentences and natural paragraphs. "
            "Use plain punctuation only: commas, periods, colons, semicolons, and question marks."
        )
        voice_constraint = (
            "VOICE AND TONE: "
            "Write as a distinguished professor and leading expert in the field, "
            "speaking directly to their students in a live lecture. "
            "Your tone should be warm, authoritative, and intellectually engaging. "
            "Ground every concept with both theoretical foundations and practical, "
            "real-world examples. "
            "Use clear transitions between ideas, as a skilled speaker would, "
            "so the listener can follow along effortlessly."
        )

        if self.mode == "short":
            return (
                f"{purpose} "
                f"Condense the following text into a brief spoken summary of the core idea, "
                f"suitable for an audiobook chapter introduction. "
                f"{voice_constraint} {lang_constraint} {formatting_constraint} "
                f"Return ONLY the spoken summary."
            )
        elif self.mode == "medium":
            return (
                f"{purpose} "
                f"Summarize the following text into a medium-length spoken explanation "
                f"for an audiobook, covering all key points and their significance. "
                f"{voice_constraint} {lang_constraint} {formatting_constraint} "
                f"Return ONLY the spoken summary."
            )
        else:  # full
            return (
                f"{purpose} "
                f"Rewrite the following text as a complete audiobook chapter transcript. "
                f"Preserve every concept, argument, and detail from the source. "
                f"Fix awkward phrasing, broken sentences, and any formatting artifacts "
                f"from PDF or HTML extraction. "
                f"Do NOT summarize, skip, or omit any content. "
                f"{voice_constraint} {lang_constraint} {formatting_constraint} "
                f"Return ONLY the complete audiobook transcript."
            )
