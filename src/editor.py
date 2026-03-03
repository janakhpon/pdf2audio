import json
import urllib.request
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
        self._previous_context: Optional[str] = None
        self._validated = False
        
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

        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    polished = result.get("response", "").strip()
                    
                    if polished:
                        if self.preserve_context:
                            self._previous_context = polished if self.mode in ("short", "medium") else polished[-800:]
                        return polished
            except Exception as e:
                logger.warning(f"Ollama error (Attempt {attempt + 1}/2): {e}")
        
        logger.error("Ollama polishing failed.")
        raise RuntimeError("Critical LLM Failure: Unable to process transcript chunk via Ollama.")

    def _build_prompt(self) -> str:
        if self.mode == "short":
            return "Summarize the text into a truly concise, punchy version. Return ONLY the polished short summary."
        elif self.mode == "medium":
            return "Summarize the text into a medium-length version, capturing main points. Return ONLY the polished medium summary."
        else:
            return (
                "Polish the following text into a clean transcript. Fix awkward line breaks and grammar. "
                "CRITICAL: Do NOT summarize, cut, or skip ANY sentences. Output the ENTIRE text word for word. "
                "Return ONLY the polished text."
            )
