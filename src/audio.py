from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
from kokoro_onnx import Kokoro
from src.config import Config
from src.logger import logger

class AudioEngine:
    """Handles generating highly realistic TTS audio using Kokoro-ONNX and pydub conversion."""

    def __init__(self, config: Config):
        self.config = config
        self._kokoro = None  # Lazy loading: instantiated only when needed

    @property
    def kokoro(self):
        if self._kokoro is None:
            logger.info(f"Lazy-loading Kokoro TTS Engine (Model: {self.config.audio_model_path})")
            self._kokoro = Kokoro(self.config.audio_model_path, self.config.audio_voices_path)
        return self._kokoro

    def generate(self, text: str, output_path: Path):
        """
        Synthesizes text and exports to the configured audio format seamlessly.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Kokoro always generates raw WAV samples locally
        temp_wav_path = output_path.with_suffix(".tmp.wav")
        logger.info(f"Synthesizing speech (Voice: {self.config.audio_voice}, Speed: {self.config.audio_speed}x)")
        
        samples, sample_rate = self.kokoro.create(
            text, 
            voice=self.config.audio_voice, 
            speed=self.config.audio_speed, 
            lang="en-us"
        )
        
        sf.write(str(temp_wav_path), samples, sample_rate)
        
        # Convert format if necessary utilizing pydub
        final_format = self.config.audio_format
        final_path = output_path.with_suffix(f".{final_format}")
        
        try:
            audio_segment = AudioSegment.from_wav(str(temp_wav_path))
            audio_segment.export(str(final_path), format=final_format)
            logger.info(f"Exported audio chunk to: {final_path}")
        except Exception as e:
            logger.error(f"Failed to export {final_format}: {e}. Retaining raw wav file.")
            temp_wav_path.rename(output_path.with_suffix(".wav"))
            return
        finally:
            if temp_wav_path.exists():
                temp_wav_path.unlink()
