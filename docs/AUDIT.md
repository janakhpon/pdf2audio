# pdf2audio — Engineering Audit

**Date:** 2026-07-02
**Scope:** Comprehensive correctness/safety review of the `pdf2audio` pipeline
(`src/`), plus tooling, tests, CI, and documentation.
**Method:** Multi-agent read-only recon across the codebase, adjudicated into the
findings below, then fixes applied and verified. Each finding is mapped to the internal
engineering standard it relates to, a severity, and a status.

## Scope & method

Standards applied (internal `se-brain` ruleset):

- **python** — uv + lockfile, ruff + mypy + pytest, src layout, typed errors, extras +
  graceful degradation.
- **cli-design** — stdout = result, stderr = diagnostics; clean exit codes; Ctrl-C handling.
- **ai-orchestration** — user/document text must not sit in the system prompt; bounded
  timeouts/retries; graceful degradation on model failure.
- **concurrency-patterns** — no shared mutable client across threads without serialization;
  clean shutdown/drain; no double-writes.
- **testing** — a real test pyramid; deterministic, offline unit tests for core logic.
- **observability** — no swallowed exceptions; log with context.
- **documentation** — README/architecture claims must match the code.

Severity: **High** = correctness/safety bug that can corrupt output, lose data, or crash
unexpectedly · **Med** = latent risk, missing guardrail, or standards gap · **Low** =
polish/hygiene. Status: **Fixed** · **Deferred** (logged in backlog) · **Won't-fix**
(intentional, with rationale).

## Findings

### Concurrency & state

| # | Finding | File | Standard | Severity | Status |
|---|---------|------|----------|----------|--------|
| C1 | One `sqlite3` connection + cursors shared by the main thread and the TTS worker, with commits from both and **no lock**. WAL protects the file, not the Python objects — a real race. | `__main__.py` | concurrency-patterns | High | **Fixed** — all access routed through `db_write`/`db_query` under a single `threading.Lock`. |
| C2 | Shutdown paths (low-disk halt, exceptions) could exit without draining the TTS worker, orphaning in-flight/queued chunks. | `__main__.py` | concurrency-patterns | High | **Fixed** — extraction wrapped in `try/finally` that always sends the sentinel and `join()`s the worker; low-disk breaks then exits after drain. |
| C3 | `KeyboardInterrupt` was not handled — Ctrl-C produced a traceback and unclear resume state. | `__main__.py` | cli-design | Med | **Fixed** — caught at the doc loop; logs "progress saved, re-run to resume" and exits 130. |
| C4 | Daemon TTS thread flagged by recon as risky. | `__main__.py` | concurrency-patterns | — | **Won't-fix** — daemon is correct here (safe interpreter exit); the real gap was the missing drain (C2), now fixed. |

### Error handling & validation

| # | Finding | File | Standard | Severity | Status |
|---|---------|------|----------|----------|--------|
| E1 | No typed exception hierarchy; modules raised bare `RuntimeError`/`ValueError` and callers caught `except Exception`, swallowing context. | all | python / observability | High | **Fixed** — `src/errors.py` (`PDF2AudioError` + domain subclasses); narrow catches; `logger.exception` / `raise ... from exc`. |
| E2 | `load_config` did no validation — bad `format`, `chunk_size`, `mode`, `timeout`, `speed`, or a malformed Ollama URL failed late and cryptically. | `config.py` | python (fail-fast) | High | **Fixed** — fail-fast validation raising `ConfigError` with actionable messages. |
| E3 | No input validation on documents — missing/empty/huge files failed deep inside `docling`/`ebooklib`. | `extractor.py` | python | Med | **Fixed** — existence + empty + 500 MB size guards raising `ExtractionError`; docling/ebooklib errors wrapped. |
| E4 | TTS model/voices files not checked; a missing model crashed deep in `kokoro-onnx`. | `audio.py` | python (fail-fast) | Med | **Fixed** — existence check in `AudioEngine.__init__` raising `AudioError`. Kept out of `load_config` since `merge`/`preview` also load config. |
| E5 | Audio synthesis branched on the **text of an exception message** (`"need at least one array to concatenate"`) and otherwise caught broad `Exception`, hiding real TTS failures. | `audio.py` | observability | Med | **Fixed** — the known empty-phoneme `ValueError` is still skipped (documented); anything else raises `AudioError` and is retried/marked `FAILED` by the worker rather than silently dropped. |
| E6 | HTML read used `errors="ignore"`, silently discarding undecodable bytes. | `extractor.py` | observability | Low | **Fixed** — changed to `errors="replace"` so corruption is visible. |

### AI / LLM orchestration

| # | Finding | File | Standard | Severity | Status |
|---|---------|------|----------|----------|--------|
| A1 | Document text was interpolated into a **single blended prompt string** (`/api/generate`), mixing instructions and untrusted extracted text. | `editor.py` | ai-orchestration | High | **Fixed** — moved to `/api/chat`: instructions in the `system` role, document text (and prior context) in the `user` role. |
| A2 | On Ollama failure the editor raised, which (given the call site) risked losing the chunk. | `editor.py` | ai-orchestration | Med | **Fixed** — graceful degradation: on unreachable/timeout/exhausted-retries it returns the **unpolished** text and disables further polish attempts for the run. |
| A3 | Per-request Ollama timeout defaulted to 600 s with 3 retries (up to 30 min of hangs) and no traceability/IDs on requests. | `editor.py` / `config.py` | ai-orchestration / observability | Med | **Partially fixed** — timeout is validated and configurable; a fast reachability pre-check (5 s) fails early. Per-request trace IDs / structured latency logging: **Deferred** (backlog). |

### Tooling, tests & CI

| # | Finding | File | Standard | Severity | Status |
|---|---------|------|----------|----------|--------|
| T1 | Legacy `black` + `isort` + `flake8`; no type checking. | `pyproject.toml` | python | Med | **Fixed** — `ruff` (lint + format, line-length 100) + `mypy` (typed-def enforcement, scoped `ignore_missing_imports`). |
| T2 | **Zero tests.** | — | testing | High | **Fixed** — offline `pytest` suite covering config validation, editor sanitize/context/prompt/degradation, extractor sort/HTML/validation, merge concat, and the SQLite state contract. |
| T3 | **No CI.** | — | python / production-readiness | Med | **Fixed** — GitHub Actions runs ruff → ruff-format-check → mypy → pytest via `uv` on push/PR. |
| T4 | Logs went to `stdout`, colliding with any piped program output. | `logger.py` | cli-design | Low | **Fixed** — logger writes to `stderr`. |
| T5 | Missing type hints throughout. | all | python | Low | **Fixed** — hints added; `mypy src` is clean. |

### Documentation

| # | Finding | File | Standard | Severity | Status |
|---|---------|------|----------|----------|--------|
| D1 | README claimed "low memory — generator-based pipeline processes arbitrarily large books"; PDF extraction loads the whole document via `docling`, and audio buffers all samples per chunk. | `README.md` | documentation | Med | **Fixed** — reworded to "streaming (EPUB/HTML)"; PDF whole-document caveat called out. |
| D2 | README implied fully offline, but the first run downloads the NLTK `punkt_tab` tokenizer. | `README.md` | documentation | Low | **Fixed** — first-run/offline note added. |
| D3 | "Parallel — LLM polishing and audio synthesis run as decoupled workers" was imprecise (polish runs on the main thread; only TTS is a worker). | `README.md` / `architecture.md` | documentation | Low | **Fixed** — reworded to "pipelined"; concurrency + DB-lock model documented in `architecture.md`. |

## Deferred backlog

- **Streaming PDF extraction** to cap peak memory (blocked by `docling` loading the full
  document; would need a page-range or streaming API).
- **Stream audio samples to disk** instead of buffering all chunk samples in memory before
  a single `soundfile.write`.
- **LLM traceability**: per-request IDs and structured per-stage latency logging
  (ai-orchestration / observability).
- **Structured (JSON) logging** with a `--log-level` flag.
- **Integration test** of the full `process_single_document` resume path (currently the DB
  state contract is unit-tested in isolation).

## Verification

All gates green at time of writing:

```
uv run ruff check .            # All checks passed
uv run ruff format --check .   # all files formatted
uv run mypy src                # no issues in 9 source files
uv run pytest                  # suite passes offline
```
