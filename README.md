# PDF to AudioBook

An offline tool that converts PDFs into realistic audiobooks.
Powered by Kokoro-ONNX for lightweight text-to-speech, with optional Ollama integration for transcript editing.

## Features

- **Massive PDF Support**: Processes large PDFs efficiently without memory issues. Memory usage stays constant regardless of file size.
- **Smart Editor**: Uses local LLMs (via Ollama) to clean up raw PDF text, removing awkward formatting before generating audio.

## Prerequisites

- Python 3.11+
- `uv` package manager

## Installation

```bash
git clone git@github.com:janakhpon/pdf2audio.git
cd pdf2audio
uv sync
```

### Download Audio Models

The text-to-speech engine requires the Kokoro models (~80MB).

```bash
mkdir -p assets/models
curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o assets/models/kokoro-v1.0.onnx
curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o assets/models/voices-v1.0.bin
```

### Optional: Ollama Setup

To use the Smart Editor for text cleanup, install [Ollama](https://ollama.com).

Recommended models:

- `llama3.2` (Default) - Fast and accurate for formatting text.
- `qwen2.5` - Highly capable for strict formatting instructions.
- `gemma2` / `mistral` - Great for summarizing text.

To download a model, run:

```bash
ollama run llama3.2
```

## Usage

### Preview Voices

Before generating an entire audiobook, you can test how a voice sounds.

1. Check the [Voices](docs/voices.md) list.
2. Run the preview script with your chosen voice:

```bash
uv run python -m src.preview af_bella
```

This generates a short audio sample in `output/audio/_preview_af_bella.mp3`.

### Full Processing

All settings are managed in `config.yaml`.

1. Choose your PDF file or directory in `config.yaml`.
2. Select your voice and language.
3. Run the application:

```bash
uv run python -m src
```

Output audio and transcripts are saved to the `output/` directory.

### Merge Audio Chunks

Because books are split into chunks for processing, you can merge all generated chunks into a single audiobook file:

```bash
uv run python -m src.merge output/audio/BOOK_NAME
```

Example:

```bash
uv run python -m src.merge output/audio/gold-rush-adventures
```

This merges all the chunks and saves a single `{BOOK_NAME}_full.mp3` file.

## Documentation

- **[Architecture](docs/architecture.md)**: System design and code structure.
- **[Voices](docs/voices.md)**: List of supported languages and voices.
