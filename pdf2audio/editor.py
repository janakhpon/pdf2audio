import json
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from pdf2audio.config import Config
from pdf2audio.errors import EditorError
from pdf2audio.logger import logger

_MAX_RETRIES = 3
_BASE_DELAY = 2.0
_VALIDATE_TIMEOUT = 5  # seconds — quick reachability check before the (slow) polish calls

_KEEP_ALIVE = "30m"  # keep the model resident across chunks (Ollama default unloads after 5m)
_TEMPERATURE = 0.4  # faithful, on-task rewrite over creative embellishment
_MIN_NUM_CTX = 4096
_MAX_NUM_CTX = 32768  # bound the KV cache so a huge chunk can't blow up memory


def _num_ctx_for_chunk_size(chunk_size: int) -> int:
    """Pick one context window for the whole run, sized to hold a chunk plus its rewrite.

    Ollama's default num_ctx (~2-4k tokens) would silently truncate our multi-KB chunk prompts
    and drop source content. But num_ctx must stay CONSTANT across calls: Ollama reloads the
    model (~seconds) whenever it changes, which would fight keep_alive and discard the cached
    system-prompt prefix. So size it once from chunk_size rather than per-chunk.

    Budget generously — roughly 900 tokens per chunk_size unit (the chunk plus its full-mode
    rewrite) plus ~2000 fixed for the system prompt and context slice — then round up to a power
    of two for headroom and clamp. At the default chunk_size=6 this lands on 8192, the window the
    smoke tests ran cleanly on; larger chunk sizes scale up.
    """
    est_tokens = chunk_size * 900 + 2000
    ctx = 1 << max(1, est_tokens - 1).bit_length()  # next power of two >= est_tokens
    return min(max(ctx, _MIN_NUM_CTX), _MAX_NUM_CTX)


# Deterministic backstop for LLM preamble/meta leakage. Even with an explicit "no preamble"
# instruction, chat models intermittently open a chunk with a conversational filler ("Okay,
# here's a breakdown…"), a meta heading ("Summary of the Text:"), or a transcript announcement.
# Narration must never contain these, so we strip a matching *leading* segment (and trailing
# sign-offs). Patterns are deliberately conservative — they only fire at the very start and only
# on unambiguous meta phrasing, so real narration is left untouched.
_PREAMBLE_RE = re.compile(
    r"""^\s*(?:
        # conversational filler used as an interjection: the discourse marker must be followed
        # immediately by clause punctuation, then the rest of that sentence is consumed. The
        # punctuation gate is what keeps ordinary narration that merely starts with one of these
        # words intact ("Right triangles form…", "Certainly the most important…", "Sure footing…").
        (?:okay|ok|sure|alright|all\s+right|certainly|of\s+course|got\s+it|understood
           |no\s+problem|absolutely|right)\b\s*(?:[,:][^.!?\n]*[.!?\n]|[.!?])
      | # "here's / below is the <breakdown|summary|transcript|version> …" (ends at .!?: or newline)
        (?:here(?:['’]s|\s+is)|below\s+is|the\s+following\s+is)\b[^.!?:\n]*
        \b(?:breakdown|summary|rewrite|rewritten|version|transcript|text|analysis
             |explanation|takeaways?)\b[^.!?:\n]*[.!?:\n]
      | # "here begins the (chapter) transcript."
        here\s+begins\s+the\s+(?:chapter\s+)?transcript\b[^.!?\n]*[.!?\n]
      | # a meta heading such as "Summary of the Text:" / "Analysis of the provided code …." —
        # gated on a meta cue that confirms it refers to the source, so a real sentence like
        # "Overview of the market shows three trends: …" is NOT treated as a heading.
        (?:summary|analysis|overview|breakdown)\s+(?:of|and)\b[^.:\n]{0,200}?
        \b(?:the\s+(?:provided|following|above)
             |the\s+(?:text|passage|document|code|chapter|section)
             |key\s+(?:concepts|takeaways|points))\b[^.:\n]*[.:\n]
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
        # One context window for the whole run — kept constant so Ollama never reloads the model
        # between chunks (a changing num_ctx forces a ~seconds reload; see _num_ctx_for_chunk_size).
        self.num_ctx = _num_ctx_for_chunk_size(config.chunk_size)
        self._previous_context: str | None = None
        self._validated = False
        # Degradation tracking: last_degraded reflects whether the most recent chunk was left
        # unpolished due to an editor failure (not a config-off editor); _degraded_run latches
        # once the editor disables itself mid-run so subsequent short-circuited chunks still count.
        self.last_degraded = False
        self._degraded_run = False

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
        for _ in range(2):  # a doubled sign-off ("I hope this helps! Let me know…") needs 2 passes
            stripped = _SIGNOFF_RE.sub("", out).strip()
            if stripped == out:
                break
            out = stripped
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
        # Privacy: the whole book text is POSTed to this host. If it is not loopback, the "text
        # stays on your machine" promise no longer holds — warn loudly (plaintext http, too).
        host = (urlparse(self.url_chat).hostname or "").lower()
        if host not in ("localhost", "127.0.0.1", "::1", ""):
            logger.warning(
                f"Editor URL points at a remote host ({host}); the full text of the book will be "
                f"sent there for polishing. Only the local editor keeps text on your machine."
            )
        self._validated = True
        logger.info(f"Ollama environment validated (model: {self.model})")

    def ensure_ready(self) -> bool:
        """Validate the editor now and return whether polishing is available.

        On failure, disable the editor and latch the run as degraded (so later chunks are
        counted as unpolished) rather than raising — the run continues with unpolished text.
        Call this eagerly before the slow extraction step so a misconfigured Ollama gives fast
        feedback instead of surfacing only after a multi-minute PDF conversion."""
        if not self.enabled:
            return False
        if self._validated:
            return True
        try:
            self.validate_environment()
            return True
        except EditorError as exc:
            logger.warning(f"{exc} Continuing with unpolished text.")
            self.enabled = False
            self._degraded_run = True
            return False

    def process_transcript(self, text: str) -> str:
        """Return polished text, or the original text unchanged if polishing is off/unavailable."""
        if not text.strip():
            self.last_degraded = False
            return text
        if not self.enabled:
            # enabled is False either because the user turned the editor off (not degraded) or
            # because it disabled itself after a failure earlier in the run (_degraded_run).
            self.last_degraded = self._degraded_run
            return text

        if not self._validated and not self.ensure_ready():
            self.last_degraded = True
            return text

        logger.info(f"Polishing transcript (model: {self.model}, mode: {self.mode})")
        context_block = ""
        if self.preserve_context and self._previous_context:
            context_block = f"<PREVIOUS_CONTEXT>\n{self._previous_context}\n</PREVIOUS_CONTEXT>\n\n"

        # Instructions stay in the system role; the document text goes in the user role so it
        # cannot override the instructions (se-brain ai-orchestration: input boundaries).
        system_content = self._build_prompt()
        user_content = f"{context_block}TEXT TO PROCESS:\n{text}"
        # num_ctx is fixed for the run. If a chunk's prompt alone overflows it, Ollama truncates
        # the prompt (silent source loss), so warn; output-side truncation is caught separately
        # via done_reason below.
        prompt_tokens = (len(system_content) + len(user_content)) // 4
        if prompt_tokens > self.num_ctx:
            logger.warning(
                f"Chunk prompt (~{prompt_tokens} tokens) exceeds the context window "
                f"({self.num_ctx}); its tail will be truncated. Lower chunk_size in config."
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            # Keep the model resident between chunks so it is not unloaded (default 5m) and
            # reloaded (~20s) whenever the TTS queue backs up the main thread.
            "keep_alive": _KEEP_ALIVE,
            "options": {
                "num_ctx": self.num_ctx,
                # Low temperature keeps the rewrite faithful and on-task rather than embellishing.
                "temperature": _TEMPERATURE,
            },
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
                if result.get("done_reason") == "length":
                    # The generation hit the context ceiling — the chunk's tail was cut off.
                    logger.warning(
                        "Ollama stopped at the context limit (done_reason=length); this chunk's "
                        "narration may be truncated. Reduce chunk_size in config."
                    )
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
                    self.last_degraded = False
                    return polished
                logger.warning(f"Ollama returned empty output (attempt {attempt + 1}).")
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                # A timeout usually means the prompt is genuinely too slow for this model, so an
                # identical retry will likely time out again — retry a timeout at most once, but
                # keep the full backoff budget for transient connection errors.
                timed_out = isinstance(exc, TimeoutError)
                if attempt == _MAX_RETRIES - 1 or (timed_out and attempt >= 1):
                    break
                delay = _BASE_DELAY * (2**attempt)
                logger.warning(
                    f"Ollama error (attempt {attempt + 1}/{_MAX_RETRIES}): {exc}. "
                    f"Retrying in {delay}s."
                )
                time.sleep(delay)

        logger.error("Ollama polishing failed; using the unpolished text for this chunk.")
        self.last_degraded = True
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
            "Use clear transitions between ideas, as a skilled speaker would, "
            "so the listener can follow along effortlessly."
        )
        fidelity_constraint = (
            "CRITICAL - FIDELITY: Narrate only the ideas, facts, and examples that are present in "
            "the source text. Do NOT invent examples, analogies, statistics, opinions, or claims "
            "that are not in the source. Your job is to voice the source faithfully, not to "
            "augment it."
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
        context_note = ""
        if self.preserve_context:
            context_note = (
                "CONTINUITY: Text inside <PREVIOUS_CONTEXT> tags is the end of the previous "
                "section, given ONLY so your narration flows on smoothly. Do NOT re-narrate, "
                "summarize, quote, or repeat it in your output. Narrate ONLY the text under "
                "'TEXT TO PROCESS:'. "
            )
        rules = (
            f"{no_preamble} {no_meta} {fidelity_constraint} {context_note}"
            f"{voice_constraint} {lang_constraint} {formatting_constraint}"
        )

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
            structured_input = (
                "HANDLING STRUCTURED INPUT: When the material is a table of contents, an index, a "
                "bare list, or a block of source code, do NOT recite it item by item or reproduce "
                "it verbatim, and do NOT turn it into an outline or a numbered 'breakdown'. "
                "Instead narrate what it contains in flowing spoken sentences: for a contents "
                "list, describe what the chapter or section will cover; for code, explain in "
                "prose what the code does and why, rather than reading the symbols aloud."
            )
            return (
                f"{purpose} "
                f"Rewrite the following text as a complete audiobook chapter transcript. "
                f"Preserve every concept, argument, and detail from the source. "
                f"Fix awkward phrasing, broken sentences, and any formatting artifacts "
                f"from PDF or HTML extraction. "
                f"Do NOT summarize, skip, or omit any content. "
                f"{structured_input} "
                f"{rules} "
                f"Return the COMPLETE transcript covering every point in the source, in order, "
                f"without summarizing, shortening, or omitting anything — and nothing else."
            )
