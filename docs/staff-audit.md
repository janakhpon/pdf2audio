# Staff Engineering Audit — pdf2audio

**Date:** 2026-07-02
**Reviewer lens:** Staff Engineer (se-brain `roles/staff-engineer`, `reviews/staff-engineering-audit`)
**Standards applied:** `delivery-surfaces`, `python`, `cli-design`, `simplicity-first`,
`observability` (down-scoped for an offline tool), `checklists/code-hygiene`.
**Scope:** the `pdf2audio` offline pipeline after the ADR 0001 correctness/safety hardening.
This audit covers *craftsmanship* — structure, organization, efficiency, cleanliness — and the
refactor applied in response (ADR 0002).

## Method

Read-only recon mapped the codebase and the exact yardstick, then changes were applied
incrementally (one concern per commit, the `ruff → ruff format → mypy → pytest` gate green
throughout). "Before" = the state at the end of ADR 0001; "After" = current.

## Scores (1–10)

| Dimension | Before | After | Notes |
|-----------|:------:|:-----:|-------|
| Reliability | 8 | 9 | Was already strong (resumable state, always-drain worker, graceful LLM degradation). Streaming audio removes an OOM path; core raises instead of `sys.exit`. |
| Correctness | 8 | 9 | Typed errors + validation from ADR 0001; this pass added real-code tests for the state store and the pipeline resume/failure paths, and fixed a logger handler-guard bug. |
| Simplicity | 5 | 9 | A 297-line god module and three duplicated entrypoints → cohesive modules + one CLI. No new abstractions beyond what earned their place; no new runtime deps (argparse is stdlib). |
| Scalability | 6 | 7 | Appropriate for the domain (one machine, batch). Bounded queue + streaming audio; PDF extraction still loads the whole document (docling limit, documented). |
| Performance | 6 | 8 | Audio no longer buffers a whole chunk's samples in memory before writing; lazy heavy-model loads retained. |
| Security | 8 | 8 | Offline, no network egress except local Ollama; LLM input boundary from ADR 0001; no secrets. Unchanged. |
| Observability | 5 | 7 | Timestamped stderr logging + `--log-level`. Full JSON/metrics intentionally out of scope (simplicity-first). |
| Maintainability & DX | 4 | 9 | Real package, `pdf2audio` console script, feature-named modules, no duplicated discovery/sort, typed `ChunkStatus`, docstrings where non-obvious. A new engineer can follow it. |
| Cost | 8 | 8 | Zero infra; local models. Unchanged. |
| Long-term sustainability | 5 | 9 | Testable core (92 → 138 tests), CI gate, two ADRs, accurate docs, tracked backlog. Debt is visible and small. |

## Current engineering level

**Before:** Production-Ready — correct and safe, but organized like a script (non-package
layout, god module, duplicated entrypoints) that a new maintainer would struggle to extend.

**After:** **Staff/Principal-Engineer Caliber for its scope.** The scope is deliberately small
(a single-machine offline CLI), and within it the codebase now reads as carefully crafted:
a clean core/adapter split, cohesive feature modules, tests that exercise real behavior, honest
docs, and complexity kept proportional to the problem. It is not "enterprise-grade" and should
not be — that would violate simplicity-first for a tool this size.

## Highest-ROI improvements (this pass)

| # | Improvement | Why it mattered | Status |
|---|-------------|-----------------|--------|
| 1 | Decompose `__main__` into `state`/`pipeline`/`documents` | Untestable god module → the orchestration and state machine are now exercised by real tests, not a replica | Done |
| 2 | Real `pdf2audio` package + `[project.scripts]` | Removed the `src.*` non-package layout and its mypy/hatch/conftest workarounds; shippable console command | Done |
| 3 | One CLI adapter over the core (delivery-surfaces) | Three duplicated entrypoints → `pdf2audio {run,preview,merge}` with a proper flag/stream/exit-code contract | Done |
| 4 | Stream audio to disk | Removed the whole-chunk in-memory sample buffer (the main memory risk after docling) | Done |
| 5 | Dedup discovery / natural-sort / hashing; dep + logging hygiene | Three drifted copies unified; dead `pydub` dropped, `numpy` declared; `--log-level`; logger guard bug fixed | Done |

## Verdict

The system is genuinely well-engineered for its purpose, not merely sophisticated-looking. The
refactor removed accidental complexity (a god module, duplicated logic, a fake package) rather
than adding cleverness, and every change is behavior-preserving and covered by tests. Remaining
items are intentionally deferred as over-engineering for an offline tool (JSON logging, metrics,
`--strict` mypy) or blocked by a dependency (streaming PDF extraction) — all recorded in ADR
0002 and `docs/audit.md`.
