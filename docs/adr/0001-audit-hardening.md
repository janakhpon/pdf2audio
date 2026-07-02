# ADR 0001 — Audit hardening: typed errors, concurrency safety, tooling, and tests

- **Status:** Accepted
- **Date:** 2026-07-02
- **Context source:** Comprehensive audit against the internal engineering standards
  (`python`, `cli-design`, `ai-orchestration`, `concurrency-patterns`, `testing`,
  `observability`, `documentation`). Full finding-by-finding mapping is in
  [`docs/AUDIT.md`](../AUDIT.md).

## Context

`pdf2audio` was a working offline pipeline (extract → optional LLM polish → TTS → merge)
with several latent correctness and safety issues typical of a fast prototype: a single
SQLite connection shared across two threads with no lock, broad `except Exception` blocks
that swallowed context, no input/config validation, document text interpolated into a
single LLM prompt string, legacy formatting tooling (`black`/`isort`/`flake8`), and no
tests or CI. None of these fail visibly in the happy path, which is exactly why they
warranted an explicit hardening pass before further feature work.

## Decisions

1. **Typed exception hierarchy (`src/errors.py`).** A single `PDF2AudioError` base with
   domain subclasses (`ConfigError`, `ExtractionError`, `EditorError`, `AudioError`,
   `MergeError`, `DatabaseError`). Modules raise these instead of bare
   `Exception`/`RuntimeError`, and callers catch `PDF2AudioError`/`OSError` narrowly rather
   than `except Exception`. Rationale: distinguishable, actionable failures; no swallowed
   context (every rescue logs with `logger.exception` or re-raises `from exc`).

2. **Serialize SQLite access with one lock.** The resume-state connection is shared by the
   main (extract/edit) thread and the TTS worker. WAL protects the *file*, but a Python
   `sqlite3` connection/cursor is not safe for concurrent use. All access now goes through
   `db_write`/`db_query` helpers guarded by a `threading.Lock`. Rejected alternative:
   one connection per thread — more moving parts than needed for a single-writer,
   single-reader workload.

3. **Keep the TTS worker as a single daemon thread, but always drain it.** The daemon flag
   is correct (safe interpreter exit), but shutdown paths previously could drop in-flight
   work. Extraction now runs inside `try/finally` that always sends the sentinel and joins
   the worker; low-disk halts and `KeyboardInterrupt` (exit 130) go through the same drain.

4. **Fail-fast config + input validation.** `load_config` validates format, bounds, mode,
   timeout, speed, and (when the editor is enabled) the Ollama URL, raising `ConfigError`
   with an actionable message. TTS model/voices file existence is checked in `AudioEngine`
   init (not in `load_config`, since `merge`/`preview` also load config). Extraction rejects
   missing/empty/oversized inputs up front.

5. **LLM input boundary (`editor.py`).** Switched from `/api/generate` with one blended
   prompt string to `/api/chat` with the instructions in the `system` role and the document
   text in the `user` role, so extracted text cannot override the instructions. On an
   unreachable/timed-out Ollama or exhausted retries, the editor **degrades to the
   unpolished text** rather than raising — a polish failure must never lose a chunk.

6. **Tooling: `ruff` + `mypy` + `pytest` via `uv`.** Replaced `black`/`isort`/`flake8` with
   `ruff` (lint + format, `line-length = 100`) and added `mypy` (not full `--strict`:
   `disallow_untyped_defs` + related, with `ignore_missing_imports` scoped to the untyped
   third-party libs). Added a `pytest` suite and a GitHub Actions workflow running
   ruff → ruff-format-check → mypy → pytest.

7. **`stderr` for logs.** The shared logger writes to `stderr` so `stdout` stays free for
   program output/piping (cli-design: stdout = result, stderr = diagnostics).

## Consequences

- Failures are now typed, logged with context, and actionable; the happy path is unchanged.
- Concurrent DB access is race-free; shutdown never orphans a chunk.
- `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest`
  is the single quality gate, enforced in CI.
- Not addressed here (logged as backlog in `docs/AUDIT.md`): streaming PDF extraction to cap
  peak memory (a `docling` limitation), streaming audio samples to disk instead of buffering
  per chunk, and structured/JSON logging with per-stage timings.
