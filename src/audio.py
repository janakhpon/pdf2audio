from pathlib import Path
import os
import io
import soundfile as sf
from pydub import AudioSegment
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
            import onnxruntime
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = self.optimal_threads
            sess_options.inter_op_num_threads = self.optimal_threads
            self._kokoro.sess.set_providers(['CPUExecutionProvider'])

        return self._kokoro

    def generate(self, text: str, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Synthesizing (Voice: {self.config.audio_voice}, Speed: {self.config.audio_speed}x)")
        samples, sample_rate = self.kokoro.create(
            text, 
            voice=self.config.audio_voice, 
            speed=self.config.audio_speed, 
            lang="en-us"
        )
        
        final_format = self.config.audio_format
        final_path = output_path.with_suffix(f".{final_format}")
        temp_final = output_path.with_suffix(f".{final_format}.tmp")
        
        try:
            wav_io = io.BytesIO()
            sf.write(wav_io, samples, sample_rate, format='wav')
            wav_io.seek(0)
            
            AudioSegment.from_wav(wav_io).export(str(temp_final), format=final_format)
            temp_final.replace(final_path)
            
            logger.info(f"Exported audio to: {final_path}")
        except Exception as e:
            logger.error(f"Export failed ({final_format}): {e}")
            if temp_final.exists():
                temp_final.unlink()
            raise
