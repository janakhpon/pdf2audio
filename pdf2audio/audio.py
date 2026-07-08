from __future__ import annotations

import os
from pathlib import Path

import nltk
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

from pdf2audio.config import Config
from pdf2audio.errors import AudioError
from pdf2audio.logger import logger


class AudioEngine:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._kokoro: Kokoro | None = None

        # Fail fast with an actionable message if the TTS model files are missing, rather than
        # deep inside kokoro-onnx on first use (se-brain: fail-fast, actionable errors).
        for label, path in (
            ("model_path", config.audio_model_path),
            ("voices_path", config.audio_voices_path),
        ):
            if not Path(path).is_file():
                raise AudioError(
                    f"TTS {label} not found: {path}. "
                    "Download the kokoro-onnx model/voices into assets/models/."
                )

        # Optimal threads: 85% of available logical cores, minimum 1.
        total_cores = os.cpu_count() or 2
        self.optimal_threads = max(1, int(total_cores * 0.85))

        # Ensure the NLTK punkt model is available up front.
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenization data...")
            nltk.download("punkt_tab", quiet=True)

    @property
    def kokoro(self) -> Kokoro:
        if self._kokoro is None:
            logger.info(
                f"Loading TTS (Model: {self.config.audio_model_path}, "
                f"Threads: {self.optimal_threads})"
            )
            # kokoro-onnx handles ONNX session thread count natively; it does not expose
            # SessionOptions at construction time.
            self._kokoro = Kokoro(self.config.audio_model_path, self.config.audio_voices_path)
            # Kokoro-ONNX does not expose SessionOptions at construction time.
            # Provider pinned to CPU to avoid unexpected GPU fallback.
            self._kokoro.sess.set_providers(["CPUExecutionProvider"])

        return self._kokoro

    def _chunk_text(self, text: str, max_chars: int = 200) -> list[str]:
        chunks: list[str] = []
        paragraphs = text.split("\n")
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
                    if not s:
                        continue

                    if len(current_chunk) + len(s) + (1 if current_chunk else 0) <= max_chars:
                        current_chunk += " " + s if current_chunk else s
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)

                        if len(s) > max_chars:
                            # Sentence is inherently too long. Hard split by words.
                            words = s.split(" ")
                            temp_chunk = ""
                            for w in words:
                                if len(temp_chunk) + len(w) + (1 if temp_chunk else 0) <= max_chars:
                                    temp_chunk += " " + w if temp_chunk else w
                                else:
                                    if temp_chunk:
                                        chunks.append(temp_chunk)

                                    if len(w) > max_chars:
                                        for i in range(0, len(w), max_chars):
                                            chunks.append(w[i : i + max_chars])
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

    def generate(self, text: str, output_path: Path) -> None:
        if not text.strip():
            logger.warning("Empty text provided for audio generation.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        text_chunks = self._chunk_text(text, max_chars=200)
        all_samples: list[np.ndarray] = []
        sample_rate = 24000

        for chunk in text_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            try:
                samples, sr = self.kokoro.create(
                    chunk,
                    voice=self.config.audio_voice,
                    speed=self.config.audio_speed,
                    lang="en-us",
                )
                sample_rate = sr
                all_samples.append(samples)
            except ValueError as exc:
                # kokoro raises ValueError when a chunk yields no phonemes (e.g. only
                # punctuation/symbols). That is expected and skippable; anything else is real.
                if "need at least one array to concatenate" in str(exc):
                    logger.debug(f"Skipped unpronounceable chunk: {chunk[:30]}...")
                else:
                    raise AudioError(
                        f"TTS synthesis failed for chunk '{chunk[:30]}...': {exc}"
                    ) from exc

        samples = np.concatenate(all_samples) if all_samples else np.array([], dtype=np.float32)

        final_path = output_path.with_suffix(".wav")  # intermediate chunks are always wav
        temp_final = output_path.with_suffix(".wav.tmp")

        try:
            sf.write(str(temp_final), samples, sample_rate, format="wav")
            temp_final.replace(final_path)  # atomic publish so partial writes never look done
            logger.info(f"Exported audio to: {final_path}")
        except OSError as exc:
            if temp_final.exists():
                temp_final.unlink()
            raise AudioError(f"Failed to write audio {final_path}: {exc}") from exc
