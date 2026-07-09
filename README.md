# pdf2audio

Convert PDFs, EPUBs, and HTML books into audiobooks — offline, on your own hardware.

## Supported Formats

| Format             | How it works                                                                                                     |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **PDF**            | Uses `docling` for layout-aware extraction — handles multi-column text, headers, and tables correctly            |
| **EPUB**           | Parses chapters sequentially via `ebooklib`                                                                      |
| **HTML directory** | Sorts files naturally (0, 1, 2… or named chapters), strips nav/script/style tags, extracts clean reading content |

## How It Works

```
Document → Extract text → LLM polish (optional) → TTS synthesis → Merge to MP3
```

- **Local** — no cloud APIs; text and audio stay on your machine (see the first-run note below)
- **Resumable** — SQLite tracks every chunk; kill it anytime and re-run to continue
- **Streaming** — extraction yields chunk-by-chunk, so EPUB/HTML books are processed incrementally rather than all at once. Note: PDF extraction loads the whole document into memory (a `docling` limitation), so peak memory scales with the size of a single PDF
- **Pipelined** — extraction and LLM polishing run on the main thread while TTS synthesis runs on a separate worker (bounded queue), so audio generation overlaps with reading the next chunk

> **First-run / offline note:** the first run downloads the NLTK `punkt_tab` sentence tokenizer (~a few MB, one time). After that the pipeline runs fully offline. The TTS models (below) and any Ollama model must also be downloaded ahead of time.

## Quick Start

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager
- `ffmpeg` (`brew install ffmpeg`)
- [Ollama](https://ollama.ai) (optional, for transcript polishing)

### Install

```bash
git clone git@github.com:janakhpon/pdf2audio.git
cd pdf2audio
uv sync
```

### Download TTS models

```bash
mkdir -p assets/models
curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
     -o assets/models/kokoro-v1.0.onnx
curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin \
     -o assets/models/voices-v1.0.bin
```

## Usage

All settings are in `config.yaml`:

1. Set `source.path` to your file or folder
2. Choose a voice and speed under `audio`
3. (Optional) Enable the `editor` block with an Ollama model for LLM polishing — makes a significant difference on PDFs with raw OCR artifacts

```bash
# Preview the configured TTS voice
uv run pdf2audio preview

# Run the full pipeline
uv run pdf2audio run

# See what would be processed, without running
uv run pdf2audio run --dry-run

# Manually merge chunks (if a previous run was interrupted)
uv run pdf2audio merge
```

Global flags: `--config PATH` (default `config.yaml`) and `--log-level {DEBUG,INFO,WARNING,ERROR}`.
`pdf2audio --help` lists everything. (`python -m pdf2audio` also works.)

## Recommended Models for Transcript Polishing

For the best audiobook quality, enable the `editor` block and use one of these models:

| Model          | RAM needed | Best for                                                       |
| -------------- | ---------- | -------------------------------------------------------------- |
| `gemma3:27b`   | ~18 GB     | **Best overall** — excellent prose flow, rarely leaks markdown |
| `qwen2.5:72b`  | ~45 GB     | Best raw quality if you have the RAM                           |
| `llama3.3:70b` | ~45 GB     | Strong instruction-following, natural lecture tone             |
| `qwen2.5:14b`  | ~10 GB     | Best mid-range — good quality on 16 GB machines                |
| `phi4:14b`     | ~10 GB     | Best for constrained hardware, punches above its weight        |

Pull a model and set it in `config.yaml`:

```bash
ollama pull gemma3:27b
```

```yaml
editor:
  enabled: true
  model: "gemma3:27b"
  mode: "full" # "full" preserves all content, "medium"/"short" summarize
```

## Operational Notes

- **Disk space** — the pipeline monitors free space mid-run and halts cleanly if it drops below 500 MB
- **Chunk size** — `source.chunk_size` controls how many files/blocks are grouped into one audio chunk. Set to `1` for one audio file per chapter
- **Output** — audio chunks are automatically merged into a single MP3/M4A/WAV at the end of each run

## Development

Install the dev extras and run the quality gate (lint, format, type-check, tests):

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy pdf2audio
uv run pytest
```

The test suite is fully offline (all heavy dependencies are mocked). CI runs the same
gate on every push and pull request.

## Documentation

- [Architecture](docs/architecture.md) — pipeline design, concurrency model, and module breakdown
- [Audit](docs/audit.md) — correctness/safety audit, and [ADR 0001](docs/adr/0001-audit-hardening.md) for those decisions
- [Staff audit](docs/staff-audit.md) — the craftsmanship review, and [ADR 0002](docs/adr/0002-staff-refactor.md) for the refactor decisions
- [Voices & Languages](docs/voices.md) — all supported voices, languages, and speed tuning
