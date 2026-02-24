# Architecture

The `pdf2audio` codebase uses a modular, configuration-driven architecture designed for reliability and offline execution.

## Core Flow

1. **Configuration (`src/config.py`)**:
   - All runtime settings are managed in `config.yaml`.

2. **Extraction (`src/extractor.py`)**:
   - Reads PDFs and yields text in chunks to maintain low memory usage.
   - Cleans basic text formatting (e.g., page numbers, newlines).

3. **Smart Editor (`src/editor.py`)**:
   - Routes text chunks to a local Ollama instance for formatting or summarization.
   - Maintains narrative context between chunks when `preserve_context` is enabled.

4. **Speech Synthesis (`src/audio.py`)**:
   - Uses `kokoro-onnx` to generate realistic speech from the processed text.
   - Exports the final audio using `pydub`.
   - Models are loaded only when needed to optimize startup time.

## Audio Merger (`src/merge.py`)

- Reads all generated audio chunks (`output/audio/X/chunk_*.mp3/wav`).
- Utilizes `pydub` to securely append them into a seamless `{book_name}_full.mp3` file.

## Logging

Standardized logging is handled by `src/logger.py` to ensure clean console output.
