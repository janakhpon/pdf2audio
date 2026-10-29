# pdf2audio

Convert PDF, EPUB and HTML books into audiobooks on your own machine, with no cloud APIs. It
extracts the text, optionally rewrites it for listening with a local LLM, synthesizes speech with
Kokoro and merges the result into one audio file.

```
Document → Extract text → LLM polish (optional) → TTS synthesis → Merge to MP3/M4A/WAV
```

- **Inputs:** a PDF (layout-aware extraction with `docling`: multi-column text, headers, tables),
  an EPUB (chapters in order via `ebooklib`), or a folder of HTML files (natural sort order;
  nav, script and style tags stripped).
- **Local:** text and audio stay on your machine.
- **Resumable:** SQLite tracks every chunk. Stop it at any time and re-run to continue.
- **Pipelined:** extraction and LLM polish run on the main thread while TTS runs on a worker
  behind a bounded queue, so synthesis overlaps with reading the next chunk.
- **Optional Smart Editor:** rewrites each chunk into spoken narration, dropping page numbers and
  layout artifacts. If it is unavailable or would drop content, the full extracted text is
  narrated instead. See [docs/smart-editor.md](docs/smart-editor.md).

## Setup

Needs Python 3.11+, [`uv`](https://github.com/astral-sh/uv), `ffmpeg` (`brew install ffmpeg`) and,
only for the Smart Editor, [Ollama](https://ollama.ai).

```bash
git clone git@github.com:janakhpon/pdf2audio.git
cd pdf2audio
uv sync

mkdir -p assets/models
curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
     -o assets/models/kokoro-v1.0.onnx
curl -sL https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin \
     -o assets/models/voices-v1.0.bin
```

The first run also downloads the NLTK `punkt_tab` sentence tokenizer (a few MB, once). After that
the pipeline runs fully offline, provided the TTS models and any Ollama model are already present.

## Usage

Everything is driven by `config.yaml`. A minimal config:

```yaml
source:
  path: "books/my-book.pdf" # a single file, a folder of PDFs/EPUBs, or a folder of HTML
audio:
  voice: "af_heart" # see docs/voices.md for the full list
  format: "mp3" # mp3 | m4a | wav
editor:
  enabled: false # true = polish the text with a local Ollama model first
```

The bundled `config.yaml` enables the editor with `qwen2.5:14b`. Set `enabled: false` if you do
not run Ollama, or pick a model that fits your RAM: [docs/smart-editor.md](docs/smart-editor.md).

```bash
uv run pdf2audio preview       # hear the configured voice in a short sample
uv run pdf2audio run --dry-run # list the documents that would be processed; changes nothing
uv run pdf2audio run           # extract → (polish) → synthesize → merge
uv run pdf2audio merge         # re-merge chunk audio from a prior run
```

The audiobook is written to `output/audio/<name>_full.<format>`, with per-chapter audio chunks and
transcripts kept under `output/`.

**If a run is interrupted, run it again.** It resumes where it stopped and never re-synthesizes
finished audio. If it stopped after the audio was made but before the final file was assembled,
`pdf2audio merge` stitches the chunks together.

Global flags: `--config PATH` (default `config.yaml`) and `--log-level {DEBUG,INFO,WARNING,ERROR}`.
`pdf2audio <command> --help` shows the rest; `python -m pdf2audio` also works.

## Limits

- **Memory:** EPUB and HTML are processed chunk by chunk. PDF extraction loads the whole document
  into memory (a `docling` limitation), so peak memory scales with the size of a single PDF.
- **Disk:** the pipeline checks free space during a run and halts cleanly below 500 MB.
- **Chunk size:** `source.chunk_size` sets how many files or blocks form one audio chunk; `1` gives
  one audio file per chapter. Changing it re-chunks the book, so the run starts fresh.
- **Loudness:** the merged file is normalized to about -19 LUFS, an audiobook-standard level. The
  merge re-encodes the whole book, so it takes longer on long books. If a very long book would
  exceed the merge timeout, it merges without normalization and logs a warning; re-run
  `pdf2audio merge` for a normalized file.
- **Language:** tuned for English and Latin-script books (default voice `af_heart`). Other
  languages synthesize, but chunk sizing assumes space-delimited words, so scripts without spaces
  (Japanese, Chinese) are best-effort.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy pdf2audio
uv run pytest
```

`pytest` runs offline: kokoro, docling, Ollama and ffmpeg are mocked, so it needs no models or
network. CI runs the same gate on every push to `main` and every pull request.

An opt-in end-to-end test drives the real model and ffmpeg. Run it after changing `audio.py`,
`pipeline.py` or `merge.py`; it proves a real audiobook still comes out end to end.

```bash
uv run pytest -m e2e   # needs assets/models/ and ffmpeg; skips itself if they are absent
```

## Documentation

- [Architecture](docs/architecture.md): pipeline design, concurrency model, modules
- [Smart Editor](docs/smart-editor.md): model choice by RAM, what the editor keeps and drops
- [Voices and languages](docs/voices.md): every voice, language and speed setting
- Decision records: [ADR 0001](docs/adr/0001-audit-hardening.md) correctness and safety (see
  [audit](docs/audit.md)) · [ADR 0002](docs/adr/0002-staff-refactor.md) craftsmanship refactor (see
  [staff audit](docs/staff-audit.md)) · [ADR 0003](docs/adr/0003-perf-and-narration-quality.md)
  performance and narration quality · [ADR 0004](docs/adr/0004-audit-hardening-and-loudness.md)
  resumability, extraction limits and loudness

## License

MIT. See [LICENSE](LICENSE).
