# Architecture

`pdf2audio` uses a modular, configuration-driven pipeline designed for reliability and fully offline execution.

## Pipeline

```
config.yaml
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
  • Optional: sends chunks to local Ollama via HTTP              │
  • Rewrites as professor-style audiobook prose                   │
  • Strips markdown symbols before TTS (sanitizer)               │
  • Preserves narrative context across chunks (2000-char window)  │
  • 3-attempt retry with exponential backoff                      │
    │                                                             │
    ▼ polished + sanitized text                                   │
    ├─── saved to output/transcripts/ (if save_transcripts: true)│
    │                                                             │
    ▼ job_queue (maxsize=3, decouples LLM latency from TTS)      │
AudioEngine                                                       │
(audio.py)                                                        │
  • kokoro-onnx: ONNX-based local TTS                            │
  • NLTK sentence tokenizer for optimal chunk splitting           │
  • numpy + soundfile for audio I/O (no pydub dependency)        │
  • 3-attempt retry with exponential backoff                      │
  • Outputs intermediate .wav chunks                              │
    │                                                             │
    ▼                                                             │
merge_audio()  ◄─────────────────────────────────────────────────┘
(merge.py)
  • ffmpeg concat demuxer — DB-ordered, not glob-sorted
  • Exports final MP3 / M4A / WAV
```

## State Management

SQLite (WAL mode) tracks every chunk as `PENDING → PROCESSING → DONE | FAILED`. On any restart, `PROCESSING` chunks are automatically reverted to `PENDING` so work is never lost. The DB file lives alongside the output audio and is keyed by a hash of the source document + config settings.

## Concurrency Model

The main thread runs the LLM polisher synchronously. A single daemon worker thread runs TTS. They communicate via a bounded `queue.Queue(maxsize=3)`, which:

- Limits memory by capping how many polished chunks can pile up waiting for TTS
- Allows LLM and TTS to overlap in time (pipeline parallelism)

Both threads share one SQLite connection. Because a `sqlite3` connection/cursor is not safe for concurrent use (WAL protects the file, not the Python objects), every read and write goes through a single `threading.Lock` (`db_write` / `db_query` helpers in `__main__.py`). On shutdown — normal, low-disk, or `KeyboardInterrupt` — the worker is always drained via a `try/finally` (`job_queue.put(None); join()`) so in-flight audio finishes and no chunk is left half-written.

## Modules

| Module         | Responsibility                                                             |
| -------------- | -------------------------------------------------------------------------- |
| `config.py`    | Loads and validates `config.yaml`, resolves paths relative to package root |
| `extractor.py` | PDF / EPUB / HTML extraction with generator-based streaming                |
| `editor.py`    | Ollama LLM polishing, context management, retry logic                      |
| `audio.py`     | TTS synthesis via kokoro-onnx, text chunking, NLTK tokenization            |
| `merge.py`     | ffmpeg-based audio assembly from DB-validated chunk list                   |
| `logger.py`    | Standardized console logging                                               |
| `preview.py`   | Quick TTS voice preview without running the full pipeline                  |

## Logging

All modules use a shared `logger` from `src/logger.py`. Key checkpoints logged at `INFO` level; chunked file reads at `DEBUG` level (visible with `--log-level DEBUG` if added).
