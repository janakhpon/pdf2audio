# Architecture

`pdf2audio` is a modular, configuration-driven pipeline built for reliability and fully
offline execution. It follows a one-core-library + thin-adapter shape: the domain logic
lives in importable modules that take a `Config` and raise typed errors, and the `cli`
module is the only part that touches argv, stdout/stderr, and exit codes.

## Pipeline

```
pdf2audio {run,preview,merge}   ← CLI adapter (cli.py): argv → Config → core calls
    │
    ▼
config.yaml ──► Config (config.py, validated on load)
    │
    ▼
DocumentExtractor  ──────────────────────────────────────────────┐
(extractor.py)                                                    │
  • PDF   → docling (layout-aware markdown extraction)           │
  • EPUB  → ebooklib (chapter-by-chapter)                        │
  • HTML  → BeautifulSoup (natural-sorted, nav/script stripped)  │
    │                                                             │
    ▼ raw text chunks (generator; PDF loads whole doc first)     │
SmartEditor                                                       │
(editor.py)                                                       │
  • Optional: sends chunks to local Ollama via /api/chat         │
  • Instructions in the system role, document text in the user   │
  • Degrades to unpolished text on any failure (never drops)     │
  • Preserves narrative context across chunks                     │
    │                                                             │
    ▼ polished text → sanitize_for_tts() (pipeline.py)           │
    ├─── saved to output/transcripts/ (if save_transcripts)      │
    │                                                             │
    ▼ job_queue (maxsize=3, decouples LLM latency from TTS)      │
AudioEngine                                                       │
(audio.py)                                                        │
  • kokoro-onnx: ONNX-based local TTS (CPU-pinned)               │
  • NLTK sentence tokenizer for chunk splitting                   │
  • streams each segment into a soundfile writer (bounded mem)   │
  • atomic temp→rename publish; outputs .wav chunks              │
    │                                                             │
    ▼                                                             │
merge_audio()  ◄─────────────────────────────────────────────────┘
(merge.py)
  • ffmpeg concat demuxer — state-ordered, not glob-sorted
  • Exports final MP3 / M4A / WAV
  • Raises MergeError (run exits 1) on ffmpeg failure — never a silent empty run
```

`pipeline.process_document()` is the orchestrator that wires these together for one document.

## State management

`ChunkStateStore` (`state.py`) is a small SQLite (WAL) repository that tracks every chunk as
`PENDING → PROCESSING → DONE | FAILED` (a typed `ChunkStatus`). On restart, `PROCESSING`
chunks are reverted to `PENDING` so interrupted work is never lost. The DB file lives beside
the output audio, keyed by `documents.document_hash()` (source content + relevant config), so
changing either restarts the affected work.

## Concurrency model

`pipeline.process_document` runs extraction and LLM polish on the calling thread; a single
daemon worker thread runs TTS. They communicate via a bounded `queue.Queue(maxsize=3)`, which:

- limits memory by capping how many polished chunks can wait for TTS, and
- lets LLM and TTS overlap in time (pipeline parallelism).

Both threads share one SQLite connection. Because a `sqlite3` connection is not safe for
concurrent use (WAL protects the file, not the Python object), every read and write goes
through the single `threading.Lock` inside `ChunkStateStore`. On shutdown — normal, low-disk,
or `KeyboardInterrupt` — the worker is always drained via a `try/finally`
(`job_queue.put(None); join()`) so in-flight audio finishes and no chunk is left half-written.
A mid-run low-disk halt raises `PDF2AudioError` from the core; the CLI maps it to exit 1.

## Modules

| Module         | Responsibility                                                              |
| -------------- | --------------------------------------------------------------------------- |
| `cli.py`       | The only transport adapter: argparse subcommands, stdout/stderr, exit codes |
| `config.py`    | Loads and validates `config.yaml` into a typed `Config`                     |
| `documents.py` | Document discovery, natural sort, and content+config hashing                |
| `extractor.py` | PDF / EPUB / HTML extraction, generator-based                               |
| `editor.py`    | Optional Ollama polish with graceful degradation                            |
| `audio.py`     | TTS synthesis via kokoro-onnx; streaming, atomic wav writer                 |
| `state.py`     | `ChunkStateStore` — the resumable SQLite chunk-state repository             |
| `pipeline.py`  | `process_document` orchestration, the TTS worker, disk policy               |
| `merge.py`     | ffmpeg-based audio assembly from the state-ordered chunk list               |
| `preview.py`   | `preview_voice` — a quick TTS sample without the full pipeline              |
| `errors.py`    | Typed exception hierarchy (`PDF2AudioError` + domain subclasses)            |
| `logger.py`    | Shared stderr logger with level control                                     |
| `__main__.py`  | Shim so `python -m pdf2audio` calls the CLI                                 |

## Logging

All modules use the shared `logger` from `pdf2audio/logger.py`, which writes timestamped
`level + message` lines to **stderr** (stdout is reserved for CLI results). Verbosity is set
with `--log-level {DEBUG,INFO,WARNING,ERROR}`. Structured JSON logging is intentionally out of
scope for an offline single-process tool (see ADR 0002).

## See also

- [Audit](audit.md) + [ADR 0001](adr/0001-audit-hardening.md) — the correctness/safety pass
- [Staff audit](staff-audit.md) + [ADR 0002](adr/0002-staff-refactor.md) — the craftsmanship
  refactor that shaped this module layout
- [ADR 0003](adr/0003-perf-and-narration-quality.md) — the fit-in-RAM model, faithful full-mode
  narration, and fail-loud merge
- [Voices & languages](voices.md) — voice IDs and the language setting
