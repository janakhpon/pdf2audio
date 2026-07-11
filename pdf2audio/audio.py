from __future__ import annotations

from pathlib import Path

import nltk
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

from pdf2audio.config import Config
from pdf2audio.errors import AudioError
from pdf2audio.logger import logger

_DEFAULT_SAMPLE_RATE = 24000  # kokoro's output rate; used only for the empty-output fallback

# kokoro encodes the language in the first letter of the voice name; map it to the espeak
# language code that kokoro-onnx feeds its g2p phonemizer. Passing the wrong lang phonemizes
# e.g. a Japanese voice with English rules, producing garbled pronunciation.
_VOICE_LANG = {
    "a": "en-us",  # American English
    "b": "en-gb",  # British English
    "e": "es",  # Spanish
    "f": "fr-fr",  # French
    "h": "hi",  # Hindi
    "i": "it",  # Italian
    "j": "ja",  # Japanese
    "p": "pt-br",  # Brazilian Portuguese
    "z": "cmn",  # Mandarin Chinese
}


def _espeak_lang_for_voice(voice: str) -> str:
    """Return the espeak language code for a kokoro voice name (default American English)."""
    return _VOICE_LANG.get(voice[:1].lower(), "en-us") if voice else "en-us"


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

        # Ensure the NLTK punkt model is available up front.
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenization data...")
            nltk.download("punkt_tab", quiet=True)

    @property
    def kokoro(self) -> Kokoro:
        if self._kokoro is None:
            logger.info(f"Loading TTS (Model: {self.config.audio_model_path})")
            # kokoro-onnx does not expose ONNX SessionOptions at construction; it uses the
            # runtime's default thread count.
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

    def _synthesize(self, chunk: str) -> tuple[np.ndarray, int] | None:
        """Synthesize one text chunk, or None if it has no pronounceable content."""
        try:
            samples, sr = self.kokoro.create(
                chunk,
                voice=self.config.audio_voice,
                speed=self.config.audio_speed,
                lang=_espeak_lang_for_voice(self.config.audio_voice),
            )
            return samples, int(sr)
        except ValueError as exc:
            # kokoro raises ValueError when a chunk yields no phonemes (e.g. only
            # punctuation/symbols). That is expected and skippable; anything else is real.
            if "need at least one array to concatenate" in str(exc):
                logger.debug(f"Skipped unpronounceable chunk: {chunk[:30]}...")
                return None
            raise AudioError(f"TTS synthesis failed for chunk '{chunk[:30]}...': {exc}") from exc

    def generate(self, text: str, output_path: Path) -> None:
        """Synthesize `text` to a single wav, streaming segments to disk (bounded memory).

        Segments are written incrementally into a temp file, then atomically renamed, so a
        partial or crashed write never leaves a file that looks complete.
        """
        if not text.strip():
            logger.warning("Empty text provided for audio generation.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_path = output_path.with_suffix(".wav")  # intermediate chunks are always wav
        temp_final = output_path.with_suffix(".wav.tmp")

        writer: sf.SoundFile | None = None
        published = False
        try:
            for chunk in self._chunk_text(text, max_chars=200):
                chunk = chunk.strip()
                if not chunk:
                    continue
                result = self._synthesize(chunk)
                if result is None:
                    continue
                samples, sr = result
                if writer is None:
                    writer = sf.SoundFile(
                        str(temp_final), mode="w", samplerate=sr, channels=1, format="WAV"
                    )
                writer.write(samples)

            if writer is None:
                # No segment produced audio. This is expected for punctuation-only text, but for
                # a chunk with real words it means silent output — surface it rather than hide it.
                if any(c.isalnum() for c in text):
                    logger.warning(
                        f"No audio produced for a chunk with text content "
                        f"('{text.strip()[:40]}...'); writing silent audio. Check the voice's "
                        f"language matches the text."
                    )
                # Emit a valid empty wav so downstream state stays consistent.
                writer = sf.SoundFile(
                    str(temp_final),
                    mode="w",
                    samplerate=_DEFAULT_SAMPLE_RATE,
                    channels=1,
                    format="WAV",
                )

            writer.close()
            writer = None
            temp_final.replace(final_path)  # atomic publish
            published = True
            logger.info(f"Exported audio to: {final_path}")
        except (OSError, sf.LibsndfileError) as exc:
            # LibsndfileError subclasses RuntimeError, not OSError — catch it too so a write
            # failure surfaces as AudioError (which callers handle) rather than a raw error.
            raise AudioError(f"Failed to write audio {final_path}: {exc}") from exc
        finally:
            if writer is not None:
                # Best-effort close during cleanup; never let a close error mask the real
                # exception or skip the temp-file unlink below.
                try:
                    writer.close()
                except Exception as close_exc:
                    logger.debug(f"Ignoring error closing audio writer: {close_exc}")
            if not published:
                temp_final.unlink(missing_ok=True)
