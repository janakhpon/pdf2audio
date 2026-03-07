# pdf2audio

Convert digital documents into structured audiobooks locally and deterministically.

## Supported Formats

- **PDF**: Uses AI layout analysis (`docling`) to structurally understand multi-column text, headers, and tables instead of blindly scraping characters.
- **EPUB**: Natively parses chapters sequentially.
- **HTML Directories**: Sorts and extracts content from a folder of HTML files, intelligently decomposing navigation and style tags to isolate reading material.

## Architecture Highlights

- **100% Offline**: Local TTS generation utilizing `kokoro-onnx`.
- **Fault-Tolerant State**: SQLite tracks chunk extraction progress natively. Safe to kill, configure, and resume operations at any time without data loss.
- **Constant Memory**: Pipeline utilizes true Python generators to process infinite-scale books with near-zero idle RAM overhead.
- **LLM Polisher (Optional)**: Can automatically offload chunks to a local Ollama model to refine NLP syntax or patch raw OCR artifacts _before_ TTS generation.

## Quick Start Guide

### Prerequisites

- Python 3.11+
- `uv` package manager
- `ffmpeg` installed on the host OS

### Setup

1. Clone the repository and install dependencies:

```bash
git clone git@github.com:janakhpon/pdf2audio.git
cd pdf2audio
uv sync
```

2. Acquire the necessary ONNX voice models:

```bash
mkdir -p assets/models
curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -o assets/models/kokoro-v1.0.onnx
curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -o assets/models/voices-v1.0.bin
```

## Usage

Execution boundaries are strictly managed via `config.yaml`.

1. Modify `config.yaml` to set your `source.path` (accepts distinct files, lists of files, or entire directories).
2. Adjust `audio` schemas to map target voices or playback speed.
3. (Optional) Toggle the `editor` block to utilize LLM NLP polishing via a local Ollama instance (ensure `ollama run [model]` is active).

### Running the Pipeline

Preview the TTS engine's configured voice profile:

```bash
uv run python -m src.preview
```

Execute the primary extraction and synthesis daemon:

```bash
uv run python -m src
```

## Operational Notes

- **Disk Validation**: The system actively monitors disk capacity mid-generation. Ensure the host drive maintains at least 500MB of free space or the daemon will gracefully halt extraction to protect the OS.
- **Concurrency**: Text generation (LLM block) and Audio generation (ONNX Engine) are decoupled async workers. CPU loads will scale dynamically to match threaded targets.
- **Safe-Merge**: Synthesized audio chunks are concatenated automatically at the end of the pipeline. If a run crashes natively, execute `uv run python -m src.merge` anytime to compile all currently successful `.wav` chunks deterministically.
