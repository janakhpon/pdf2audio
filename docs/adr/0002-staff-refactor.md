# ADR 0002 — Staff-level craftsmanship refactor: package, CLI, and module structure

- **Status:** Accepted
- **Date:** 2026-07-02
- **Follows:** [ADR 0001](0001-audit-hardening.md) (correctness/safety hardening)
- **Scored review:** [`docs/staff-audit.md`](../staff-audit.md)

## Context

ADR 0001 fixed the correctness and safety layer (typed errors, one-lock SQLite, always-drain
worker, LLM input boundary, fail-fast validation, tests, CI). What remained was structural: the
code imported itself as `src.*` (not a real package), a 297-line `__main__.py` mixed seven
concerns, three entrypoints each re-loaded config and re-discovered documents, discovery and
natural-sort were duplicated three ways, audio buffered all samples in memory, and there was no
CLI. None of this fails at runtime — it is maintainability and craftsmanship debt. This pass
raises the codebase to staff-level per the internal `staff-engineer` role and the
`delivery-surfaces`, `python`, `cli-design`, and `simplicity-first` standards, executed
incrementally (one item per commit, tests green throughout — no big-bang).

## Decisions

1. **Real package (`src/` → `pdf2audio/`).** Add `__init__.py`, rewrite the 23 `src.*` imports,
   and delete the workarounds the non-package layout forced (the `conftest.py` `sys.path` hack
   and the `mypy_path`/`explicit_package_bases`/`namespace_packages` config). This is not
   cosmetic: it aligns with the standard src-layout and enables a `[project.scripts]` console
   entry. Reversible.

2. **One core library + a thin CLI adapter (delivery-surfaces).** All domain logic
   (`extractor`, `editor`, `audio`, `merge`, `state`, `pipeline`, `documents`) takes a `Config`
   and returns values / raises typed errors. `cli.py` is the only module that touches argv,
   stdout/stderr, and exit codes. The three former entrypoints (`python -m src`, `.merge`,
   `.preview`) collapse into `pdf2audio {run,preview,merge}` with `--config`, `--log-level`, and
   `run --dry-run`; `merge.main`/`preview.main` become `merge_all`/`preview_voice`; `__main__` is
   a shim so `python -m pdf2audio` still works.

3. **Decompose the god module.** Extract `ChunkStateStore` (`state.py`), `process_document` +
   the TTS worker + disk policy (`pipeline.py`), and discovery/sort/hash (`documents.py`) out of
   `__main__.py` (297 → ~5 lines). This also let `test_state.py` and `test_pipeline.py` exercise
   the *real* code instead of the schema replica the previous test was forced to use.

4. **Core raises, adapter exits.** The library no longer calls `sys.exit`. A mid-run low-disk
   halt raises `PDF2AudioError`; the CLI maps typed errors to exit codes (0/1/130). Transport
   concerns belong to the adapter, not the core.

5. **Stream audio to disk.** `audio.generate` writes each synthesized segment straight into a
   `soundfile.SoundFile` instead of concatenating all samples in memory first. Peak memory is
   now bounded by one segment, and the atomic temp→rename publish is preserved.

6. **Dependency + logging hygiene.** Drop the unused `pydub`; declare the direct `numpy`. Add a
   timestamp to the stderr log format and a `--log-level` flag. Fix `setup_logger`'s
   `hasHandlers()` guard (which inspected ancestors) to check the logger's own handlers, so the
   CLI always owns its stderr output.

## Consequences

- Installable as `pdf2audio`; `uv run pdf2audio --help` documents the surface.
- The core is unit-testable without standing up a transport; the suite grew from 92 to 138 tests
  and now covers the real state store, pipeline resume/failure paths, CLI, logging, and audio.
- Behavior is preserved end-to-end; the same quality gate
  (`ruff check` · `ruff format --check` · `mypy pdf2audio` · `pytest`) stays green.

## Deliberately not done (simplicity-first / down-scoped)

- **Full JSON/structured logging, metrics, tracing, health endpoints** — over-engineering for an
  offline single-process tool; timestamped stderr logging is sufficient.
- **mypy `--strict`** — the current config already enforces typed definitions; a full `--strict`
  flip adds noise around the untyped third-party libs (kokoro/docling/soundfile/nltk/bs4) for
  little gain. Kept as ADR 0001 chose.
- **Streaming PDF extraction** — still bounded by `docling` loading the whole document; noted in
  `docs/AUDIT.md` backlog.
