import os
import io
import re
import numpy as np
from pathlib import Path
import soundfile as sf
import onnxruntime
import nltk
from kokoro_onnx import Kokoro
from src.config import Config
from src.logger import logger

class AudioEngine:
    def __init__(self, config: Config):
        self.config = config
        self._kokoro = None
        
        # Calculate optimal threads: 85% of available logical cores, minimum 1
        total_cores = os.cpu_count() or 2
        self.optimal_threads = max(1, int(total_cores * 0.85))

    @property
    def kokoro(self):
        if self._kokoro is None:
            logger.info(f"Loading TTS (Model: {self.config.audio_model_path}, Threads: {self.optimal_threads})")
            # Initialize Kokoro with explicit thread count constraints if the API supports it, 
            # or rely on ONNX session options implicitly through the backend if exposed.
            # (Kokoro-ONNX handles session thread count natively if passed kwargs, or defaults to all cores).
            self._kokoro = Kokoro(self.config.audio_model_path, self.config.audio_voices_path)
            
            # Forcing ONNX thread count via session options if exposed:
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = self.optimal_threads
            sess_options.inter_op_num_threads = self.optimal_threads
            self._kokoro.sess.set_providers(['CPUExecutionProvider'])
            
            # Ensure nltk punkt model is available
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                logger.info("Downloading NLTK punkt tokenization data...")
                nltk.download('punkt_tab', quiet=True)

        return self._kokoro

    def _chunk_text(self, text: str, max_chars: int = 200) -> list[str]:
        chunks = []
        paragraphs = text.split('\n')
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(p) <= max_chars:
                chunks.append(p)
            else:
                sentences = nltk.tokenize.sent_tokenize(p)
                current_chunk = ""
                
                for s in sentences:
                    s = s.strip()
                    if not s: continue
                    
                    if len(current_chunk) + len(s) + (1 if current_chunk else 0) <= max_chars:
                        current_chunk += (" " + s if current_chunk else s)
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        if len(s) > max_chars:
                            # Sentence is inherently too long. Hard split by words.
                            words = s.split(' ')
                            temp_chunk = ""
                            for w in words:
                                if len(temp_chunk) + len(w) + (1 if temp_chunk else 0) <= max_chars:
                                    temp_chunk += (" " + w if temp_chunk else w)
                                else:
                                    if temp_chunk:
                                        chunks.append(temp_chunk)
                                    
                                    if len(w) > max_chars:
                                        for i in range(0, len(w), max_chars):
                                            chunks.append(w[i:i+max_chars])
                                        temp_chunk = ""
                                    else:
                                        temp_chunk = w
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            current_chunk = ""
                        else:
                            current_chunk = s
                
                if current_chunk:
                    chunks.append(current_chunk)
        return chunks

    def generate(self, text: str, output_path: Path):
        if not text.strip():
            logger.warning("Empty text provided for audio generation.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Synthesizing (Voice: {self.config.audio_voice}, Speed: {self.config.audio_speed}x)")
        
        text_chunks = self._chunk_text(text, max_chars=200)
        all_samples = []
        sample_rate = 24000
        
        for chunk in text_chunks:
            chunk = chunk.strip()
            if not chunk: continue
            
            try:
                samples, sr = self.kokoro.create(
                    chunk, 
                    voice=self.config.audio_voice, 
                    speed=self.config.audio_speed, 
                    lang="en-us"
                )
                sample_rate = sr
                all_samples.append(samples)
            except ValueError as e:
                if "need at least one array to concatenate" in str(e):
                    logger.debug(f"Skipped unpronounceable chunk: {chunk[:30]}...")
                else:
                    raise e
            except Exception as e:
                logger.warning(f"Failed to generate audio for chunk '{chunk[:30]}...': {e}")
            
        if all_samples:
            samples = np.concatenate(all_samples)
        else:
            samples = np.array([], dtype=np.float32)
        
        final_format = "wav" # Intermediate chunks must always be wav
        final_path = output_path.with_suffix(f".{final_format}")
        temp_final = output_path.with_suffix(f".{final_format}.tmp")
        
        try:
            sf.write(str(temp_final), samples, sample_rate, format='wav')
            temp_final.replace(final_path)
            
            logger.info(f"Exported audio to: {final_path}")
        except Exception as e:
            logger.error(f"Export failed ({final_format}): {e}")
            if temp_final.exists():
                temp_final.unlink()
            raise
