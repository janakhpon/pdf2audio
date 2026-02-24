import json
import urllib.request
from typing import Optional
from src.config import Config
from src.logger import logger

class SmartEditor:
    """Seamlessly integrates with a local Ollama instance for professional transcript generation."""

    def __init__(self, config: Config):
        self.model = config.editor_model
        self.mode = config.editor_mode
        self.url = f"{config.editor_url}/api/generate"
        self.enabled = config.editor_enabled
        self.preserve_context = config.editor_preserve_context
        self._previous_context: Optional[str] = None

    def process_transcript(self, text: str) -> str:
        """
        Sends the chunk to local Ollama for narrative enhancement.
        If preserve_context is enabled, injects the preceding narrative dynamically.
        """
        if not self.enabled or not text.strip():
            return text
            
        logger.info(f"Polishing transcript via Ollama (Model: {self.model}, Mode: {self.mode})")
            
        system_prompt = self._build_prompt()
        
        # Inject the sliding window if we have established one
        context_block = ""
        if self.preserve_context and self._previous_context:
            context_block = (
                f"\n\n<PREVIOUS_CONTEXT>\n"
                f"For your awareness, here is what happened in the immediately preceding section. "
                f"Use this to maintain narrative coherence and resolve any mid-sentence breaks:\n"
                f"{self._previous_context}\n"
                f"</PREVIOUS_CONTEXT>\n"
            )

        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}{context_block}\n\nTEXT TO PROCESS:\n{text}",
            "stream": False
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})

        # Implement robust timeout and retry logic for high reliability
        max_retries = 2
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    polished = result.get("response", "").strip()
                    
                    # Update the state sliding window for the next extraction chunk
                    if polished and self.preserve_context:
                        if self.mode in ("short", "medium"):
                            self._previous_context = polished
                        else:
                            self._previous_context = polished[-800:] if len(polished) > 800 else polished
                            
                    return polished if polished else text
            except Exception as e:
                logger.warning(f"Ollama Polishing Error (Attempt {attempt + 1}/{max_retries}): {e}")
        
        logger.error("Ollama polishing failed after retries. Falling back to the raw generated text.")
        return text

    def _build_prompt(self) -> str:
        if self.mode == "short":
            return (
                "You are an expert audiobook editor. Summarize the following raw text "
                "into a highly concise, punchy version. Write it so that it flows beautifully "
                "when read aloud by a narrator. Return ONLY the polished short summary."
            )
        elif self.mode == "medium":
            return (
                "You are an expert audiobook editor. Summarize the following raw text "
                "into a medium-length version, capturing the main points and key details. "
                "Write it so that it flows beautifully when read aloud by a narrator. "
                "Return ONLY the polished medium summary."
            )
        else:
            return (
                "You are an expert audiobook editor. Polish the following raw text "
                "into a proper, clean transcript. Fix awkward line breaks, spellings, expand acronyms, "
                "remove meaningless artifacts, and ensure natural punctuation. Do NOT summarize or "
                "remove important narrative details. Return ONLY the polished text."
            )
