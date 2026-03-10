import json
import urllib.request
import urllib.error
import re
import time
from typing import Optional
from src.config import Config
from src.logger import logger

class SmartEditor:
    def __init__(self, config: Config):
        self.model = config.editor_model
        self.mode = config.editor_mode
        self.url_generate = f"{config.editor_url}/api/generate"
        self.url_tags = f"{config.editor_url}/api/tags"
        self.enabled = config.editor_enabled
        self.preserve_context = config.editor_preserve_context
        self.timeout = config.editor_timeout
        self._previous_context: Optional[str] = None
        self._validated = False

    def _safe_slice_context(self, text: str, max_chars: int = 2000) -> str:
        if len(text) <= max_chars:
            return text
        sliced = text[-max_chars:]
        match = re.search(r'[\s.\n]', sliced)
        if match:
            return sliced[match.end():].strip()
        return sliced.strip()
        
    def validate_environment(self):
        if not self.enabled:
            return
            
        try:
            req = urllib.request.Request(self.url_tags)
            with urllib.request.urlopen(req, timeout=5) as response:
                tags_data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name") for m in tags_data.get("models", [])]
                
                # Check if exact model name or model with :latest exists
                if self.model not in models and f"{self.model}:latest" not in models:
                    raise RuntimeError(f"Ollama model '{self.model}' not pulled. Run `ollama pull {self.model}`.")
                    
            self._validated = True
            logger.info(f"Ollama environment validated (Model: {self.model})")
        except urllib.error.URLError:
            raise RuntimeError(f"Ollama connection refused at {self.url_tags}. Is Ollama running?")

    def process_transcript(self, text: str) -> str:
        if not self.enabled or not text.strip():
            return text
            
        if not self._validated:
            self.validate_environment()
            
        logger.info(f"Polishing transcript (Model: {self.model}, Mode: {self.mode})")
        system_prompt = self._build_prompt()
        
        context_block = ""
        if self.preserve_context and self._previous_context:
            context_block = (
                f"\n\n<PREVIOUS_CONTEXT>\n"
                f"{self._previous_context}\n"
                f"</PREVIOUS_CONTEXT>\n"
            )

        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}{context_block}\n\nTEXT TO PROCESS:\n{text}",
            "stream": False
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url_generate, data=data, headers={"Content-Type": "application/json"})

        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    polished = result.get("response", "").strip()
                    
                    if polished:
                        if self.preserve_context:
                            self._previous_context = polished if self.mode in ("short", "medium") else self._safe_slice_context(polished)
                        return polished
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Ollama polishing failed after {max_retries} attempts.")
                    break
                    
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Ollama error (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
        
        raise RuntimeError("Critical LLM Failure: Unable to process transcript chunk via Ollama.")

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
                f"{voice_constraint} "
                f"{lang_constraint} "
                f"{formatting_constraint} "
                f"Return ONLY the spoken summary."
            )
        elif self.mode == "medium":
            return (
                f"{purpose} "
                f"Summarize the following text into a medium-length spoken explanation "
                f"for an audiobook, covering all key points and their significance. "
                f"{voice_constraint} "
                f"{lang_constraint} "
                f"{formatting_constraint} "
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
                f"{voice_constraint} "
                f"{lang_constraint} "
                f"{formatting_constraint} "
                f"Return ONLY the complete audiobook transcript."
            )
