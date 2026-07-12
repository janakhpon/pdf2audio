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

The optional Smart Editor rewrites each chunk into clean spoken narration — dropping page numbers,
figure/table references, and layout artifacts that only make sense on a page — while preserving the
meaning. If it is unavailable or would drop content, the pipeline narrates the complete extracted
text instead, so nothing meaningful is lost.

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

Everything is driven by `config.yaml`. A minimal config:

```yaml
source:
  path: "books/my-book.pdf" # a single file, a folder of PDFs/EPUBs, or a folder of HTML
audio:
  voice: "af_heart" # see docs/voices.md for the full list
  format: "mp3" # mp3 | m4a | wav
editor:
  enabled: false # true = polish the text with a local Ollama model first (optional)
```

The editor is optional — with it off, the extracted text is narrated directly. The bundled
`config.yaml` enables it with `qwen2.5:14b`; see the model guidance below.

Then, from the project directory:

```bash
uv run pdf2audio preview       # hear the configured voice — a ~3s sample
uv run pdf2audio run --dry-run # list the documents that would be processed; makes no changes
uv run pdf2audio run           # the full pipeline: extract → (polish) → synthesize → merge
```

The finished audiobook is written to **`output/audio/<name>_full.<format>`**, with the
per-chapter audio chunks and transcripts kept alongside it under `output/`.

**If a run is interrupted**, just run it again — every chunk's state is tracked in SQLite, so it
resumes where it stopped and never re-synthesizes finished audio. If a run was cut off after the
audio was made but before the final file was assembled, `uv run pdf2audio merge` stitches the
chunks together.

Global flags: `--config PATH` (default `config.yaml`) and `--log-level {DEBUG,INFO,WARNING,ERROR}`.
Run `pdf2audio --help`, or `pdf2audio <command> --help`, for the full surface. (`python -m pdf2audio`
works too.)

## Recommended Models for Transcript Polishing

For the best audiobook quality, enable the `editor` block and use one of these models:

| Model          | RAM needed | Best for                                                       |
| -------------- | ---------- | -------------------------------------------------------------- |
| `qwen2.5:14b`  | ~9 GB      | **Recommended default** — good quality, fits 16-24 GB machines |
| `phi4:14b`     | ~9 GB      | Constrained hardware, punches above its weight                 |
| `gemma3:27b`   | ~17 GB     | Excellent prose, but only if it fits (needs ~32 GB, see below) |
| `qwen2.5:72b`  | ~45 GB     | Best raw quality if you have the RAM                           |
| `llama3.3:70b` | ~45 GB     | Strong instruction-following, natural lecture tone             |

**The model must fit in RAM.** The Smart Editor makes one LLM call per chunk (hundreds per book),
so if the model does not fit alongside the OS and the TTS engine, Ollama pages it to disk and each
call runs roughly 10x slower — a book can take a full day instead of a few hours. On a 24 GB
machine, `gemma3:27b` (~17 GB) swaps once the OS + Python + Kokoro are loaded; `qwen2.5:14b` (~9 GB)
leaves headroom. Pick the largest model that comfortably fits, not the largest you can load.

Pull a model and set it in `config.yaml`:

```bash
ollama pull qwen2.5:14b
```

```yaml
editor:
  enabled: true
  model: "qwen2.5:14b"
  mode: "full" # "full" = faithful full narration (not a summary); "medium"/"short" summarize
```

**On dense technical books**, the editor guards against dropped meaning: if the polish comes back
much shorter than the source (a sign the model summarized rather than narrated), that chunk is read
from the complete extracted text instead — you'll see a `using the complete raw text` log line.
Nothing is lost; those chunks just sound rougher (notation and references read aloud). A stronger
model or a smaller `chunk_size` increases how much gets fully polished.

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

`pytest` runs the offline suite — the heavy dependencies (kokoro, docling, Ollama, ffmpeg) are
mocked, so it needs no models or network and CI runs the same gate on every push and pull request.

There is also an opt-in **end-to-end smoke test** that drives the *real* model and ffmpeg:

```bash
uv run pytest -m e2e   # needs assets/models/ + ffmpeg; auto-skips if absent
```

Run it locally after changing anything in `audio.py`, `pipeline.py`, or `merge.py` — it's the
check that proves a real audiobook still comes out end to end.

## Documentation

- [Architecture](docs/architecture.md) — pipeline design, concurrency model, and module breakdown
- [Voices & languages](docs/voices.md) — all supported voices, languages, and speed tuning
- Decision records — [ADR 0001](docs/adr/0001-audit-hardening.md) correctness/safety (see [audit](docs/audit.md)) · [ADR 0002](docs/adr/0002-staff-refactor.md) craftsmanship refactor (see [staff audit](docs/staff-audit.md)) · [ADR 0003](docs/adr/0003-perf-and-narration-quality.md) performance + narration quality
