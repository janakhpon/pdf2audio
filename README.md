# PDF2Audio

An offline tool designed to convert large PDF files into realistic audiobooks with stable memory footprint. It utilizes Kokoro-ONNX for text-to-speech synthesis and offers optional Ollama integration for transcript processing.

## Architecture

- **Extractor**: Reads PDFs efficiently (`pypdf`), ensuring O(1) memory usage by yielding chunks.
- **Editor**: Cleans layout artifacts, optionally interfacing with a local `ollama` instance for narrative enhancement.
- **AudioEngine**: Deploys `kokoro-onnx` for realistic voice synthesis, outputting locally.
- **Merge**: Utility to concatenate chunks into a single distribution file executing `ffmpeg concat`.

## Setup

Requires Python 3.11+ and the `uv` package manager.

1. **Clone & Sync**

```bash
git clone git@github.com:janakhpon/pdf2audio.git
cd pdf2audio
uv sync
```

2. **Acquire Audio Models**

```bash
mkdir -p assets/models
curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o assets/models/kokoro-v1.0.onnx
curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o assets/models/voices-v1.0.bin
```

3. **Install Output Dependencies**
   Ensure `ffmpeg` is available on your system path for `pydub` format conversions.

#### Optional: Environment Configuration (LLM Editor)

If transcript enhancement is desired, verify a local `ollama` service is active on `http://localhost:11434`.

```bash
ollama run llama3.2
```

## Run Instructions

Configuration is strictly managed via `config.yaml`. Define input source (`source.path`) and target audio directory (`output.audio_dir`).

**Preview Synthesis Voice:**

```bash
uv run python -m src.preview
```

**Execute Processing Pipeline:**

```bash
uv run python -m src
```

**Merge Final Output:**

```bash
uv run python -m src.merge
```

## Development Workflow

- Modify configuration via `config.yaml`.
- All TTS processing is deterministic and output formats are dictated exclusively by the configuration schema.
- System operates statelessly; processed chunks are cached in `output/` and skipped securely during subsequent executions.

## Deployment Overview

Designed strictly for local daemon execution or containerized workloads on hardware with adequate CPU/RAM limits for TTS loading logic. Due to dependencies on ONNX binaries and local generation, isolated deployment topologies are recommended.
