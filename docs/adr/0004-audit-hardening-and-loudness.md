# ADR 0004 — Audit hardening: resumability key, safe sanitizer, extraction limits, loudness

- **Status:** Accepted
- **Date:** 2026-07-21
- **Follows:** [ADR 0003](0003-perf-and-narration-quality.md) (perf + narration quality)
- **Context source:** an se-brain role-driven review (staff-engineer, principal-engineer,
  ai-system-review-framework, production-audit, security-engineer lenses) run as a four-reviewer
  adversarial panel. Every lead was verified against the code before acting; refuted leads are
  recorded below so they are not re-raised.

## Context

After the loudness-normalization and merge-timeout work landed on top of ADR 0003, a final
staff/principal audit went looking for correctness, robustness, and fidelity gaps in the whole
pipeline. It confirmed nine issues (and refuted more than it confirmed). The load-bearing ones were
a resumability-key gap that could silently reuse stale audio, and a sanitizer rule that corrupted
math notation.

## Decisions

1. **`chunk_size` keys the resumable state.** `document_hash` folded in voice/speed/model/mode but
   not `chunk_size`, which moves the chunk boundaries. Changing it and re-running silently reused the
   old chunking's audio (wrong text-to-audio mapping, or a stale merge). It is now in the hash, so a
   changed `chunk_size` starts fresh. (`num_ctx`/`preserve_context` are deliberately excluded — they
   change polish quality but not the text-to-index mapping, and including them would nuke progress on
   a tuning tweak.)

2. **Pipe→comma is scoped to real table rows.** `sanitize_for_tts` rewrote every `|` to `", "`
   globally, corrupting inline math (`P(θ|x)` → `P(θ, x)` inverts a conditional into a joint;
   set-builder and absolute value likewise). It now only de-pipes lines with two or more dividers; a
   lone inline `|` is left alone.

3. **TOC skip is density-gated, not count-gated.** `is_structural_noise` fired on ≥3 dot-leader runs
   anywhere, so a chunk that merely straddled the tail of a TOC and real prose was dropped whole
   (with a misleading "TOC page" log). It now requires the dot-leaders to be dense relative to the
   word count, so a mostly-prose chunk is narrated (its few leaders are stripped downstream).

4. **The TTS worker cannot die before draining.** The FAILED-mark and `task_done()` ran outside any
   guard; a throwing `store.mark` (e.g. a locked SQLite) would kill the worker and hang the main
   thread on the bounded queue. The per-job body is now wrapped so `task_done()` always runs and the
   worker only exits on the sentinel.

5. **Extraction has a memory ceiling, not just a disk one.** The on-disk 500 MB cap bounds only the
   compressed input; EPUB and PDF both blow up after decompression/parsing, before the mid-run disk
   guard runs. Added a decompressed-size cap for EPUB (zip-bomb guard) and a page-count cap for PDF
   (`docling.convert(max_num_pages=...)`).

6. **Smaller hardening.** Null-safe Ollama `message` parse (`{"message": null}` no longer aborts the
   run); explicit `num_predict: -1` so a small server default doesn't truncate every rewrite; upper
   bounds on `editor.num_ctx`/`editor.timeout`; `_chunk_text` default aligned to the real cap.

7. **Loudness + merge timeout (recording the prior two commits).** The final audiobook is
   loudness-normalized to ~-19 LUFS (ACX range) via a single ffmpeg `loudnorm` pass; the merge
   subprocess timeout scales with input size (loudnorm forces a full re-encode); a pathologically
   large book that would still time out falls back to a plain, un-normalized merge with a warning.

## Explicitly not changed

- **Collapse floor stays at 0.10.** Trusting the polish unless it near-totally collapses is a
  deliberate product decision (complete-but-noisier raw is the fallback only when the model clearly
  breaks). Raising it to catch 40-89% condensation was considered and rejected as reversing that
  call; the trade-off is now documented in the README.
- **CJK counting.** The `//4` token estimate and `.split()` word counts assume space-delimited text.
  The tool is English/Latin-focused (default `af_heart`); CJK is documented as best-effort rather
  than given a script-aware tokenizer (out of scope at current use).
- **Refuted leads (verified false, recorded):** shell injection, path traversal via `doc.stem`, SQL
  injection, ReDoS, secrets exposure, non-determinism across resume, per-chunk empty-wav merge
  consistency, and the committed `mode` (it is `full`; `medium` was only a local working override).

## Verification

234 offline tests (+8 for the fixes above) and the 3 real-dependency e2e tests pass; ruff, format,
and mypy clean. The merge fix was verified end to end on a ~3.2 GB / 366-chunk book that previously
timed out: it completes and measures −19.0 LUFS.
