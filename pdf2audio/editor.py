import json
import re
import time
import urllib.error
import urllib.request

from pdf2audio.config import Config
from pdf2audio.errors import EditorError
from pdf2audio.logger import logger

_MAX_RETRIES = 3
_BASE_DELAY = 2.0
_VALIDATE_TIMEOUT = 5  # seconds — quick reachability check before the (slow) polish calls

# Deterministic backstop for LLM preamble/meta leakage. Even with an explicit "no preamble"
# instruction, chat models intermittently open a chunk with a conversational filler ("Okay,
# here's a breakdown…"), a meta heading ("Summary of the Text:"), or a transcript announcement.
# Narration must never contain these, so we strip a matching *leading* segment (and trailing
# sign-offs). Patterns are deliberately conservative — they only fire at the very start and only
# on unambiguous meta phrasing, so real narration is left untouched.
_PREAMBLE_RE = re.compile(
    r"""^\s*(?:
        # pure conversational filler opener, up to the first sentence break
        (?:okay|ok|sure|alright|all\s+right|certainly|of\s+course|got\s+it|understood
           |no\s+problem|absolutely|right)\b[^.!?\n]*[.!?\n]
      | # "here's / below is the <breakdown|summary|transcript|version> …" (ends at .!?: or newline)
        (?:here(?:['’]s|\s+is)|below\s+is|the\s+following\s+is)\b[^.!?:\n]*
        \b(?:breakdown|summary|rewrite|rewritten|version|transcript|text|analysis
             |explanation|takeaways?)\b[^.!?:\n]*[.!?:\n]
      | # "here begins the (chapter) transcript."
        here\s+begins\s+the\s+(?:chapter\s+)?transcript\b[^.!?\n]*[.!?\n]
      | # a meta heading such as "Summary of …:" / "Analysis of …:" (must end in colon/newline)
        (?:summary|analysis|overview|breakdown)\s+(?:of|and)\b[^:\n]{0,120}[:\n]
    )\s*""",
    re.IGNORECASE | re.VERBOSE,
)
_SIGNOFF_RE = re.compile(
    r"\s*(?:i\s+hope\s+this\s+helps|let\s+me\s+know\s+if[^.!?\n]*|feel\s+free\s+to[^.!?\n]*"
    r"|hope\s+(?:you|this)[^.!?\n]*)[.!?]?\s*$",
    re.IGNORECASE,
)


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

    def _strip_artifacts(self, text: str) -> str:
        """Remove leaked conversational preamble/meta framing from the model's output.

        A safety net over the prompt's "no preamble" instruction: chat models still open a
        chunk with "Okay, here's a breakdown…", a "Summary of…:" heading, or a transcript
        announcement often enough that narration needs a deterministic guard. Strips up to two
        leading meta segments (handles a doubled "Okay, here's X. Summary of Y:" opener) plus a
        trailing sign-off. Falls back to the original text if stripping would empty it."""
        out = text.strip()
        for _ in range(2):
            stripped = _PREAMBLE_RE.sub("", out, count=1).strip()
            if stripped == out:
                break
            out = stripped
        out = _SIGNOFF_RE.sub("", out).strip()
        return out or text

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
                polished = self._strip_artifacts(
                    str(result.get("message", {}).get("content", "")).strip()
                )
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
        no_preamble = (
            "CRITICAL - NO PREAMBLE: Begin your response immediately with the actual spoken "
            "content. Your very first words must be the narration itself. Do NOT open with any "
            "acknowledgement, preamble, or meta-comment. Never begin with words or phrases such "
            "as 'Okay', 'Sure', 'Alright', 'Certainly', 'Of course', 'Here is', \"Here's\", "
            "'Here is a breakdown', 'Below is', 'Here begins the transcript', or with a heading "
            "such as 'Summary of...', 'Analysis of...', 'Overview of...', or 'Key Takeaways'."
        )
        no_meta = (
            "CRITICAL - NO META-COMMENTARY: Never mention 'the text', 'the provided text', 'the "
            "passage', 'the source', 'the document', 'the author', or the fact that you are "
            "rewriting, summarizing, or processing anything. Do NOT describe what the material "
            "does (never write things like 'This text explains...' or 'This section covers...'); "
            "instead, narrate the material directly in the lecturer's own voice, as if the ideas "
            "are your own. Do NOT add a title or heading, and do NOT end with a closing remark "
            "such as 'I hope this helps' or 'Let me know if you need anything else'."
        )
        rules = f"{no_preamble} {no_meta} {voice_constraint} {lang_constraint} {formatting_constraint}"

        if self.mode == "short":
            return (
                f"{purpose} "
                f"Condense the following text into a brief spoken summary of the core idea, "
                f"suitable for an audiobook chapter introduction. "
                f"{rules} "
                f"Return ONLY the spoken summary, and nothing else."
            )
        elif self.mode == "medium":
            return (
                f"{purpose} "
                f"Summarize the following text into a medium-length spoken explanation "
                f"for an audiobook, covering all key points and their significance. "
                f"{rules} "
                f"Return ONLY the spoken summary, and nothing else."
            )
        else:  # full
            return (
                f"{purpose} "
                f"Rewrite the following text as a complete audiobook chapter transcript. "
                f"Preserve every concept, argument, and detail from the source. "
                f"Fix awkward phrasing, broken sentences, and any formatting artifacts "
                f"from PDF or HTML extraction. "
                f"Do NOT summarize, skip, or omit any content. "
                f"{rules} "
                f"Return ONLY the complete audiobook transcript, and nothing else."
            )
