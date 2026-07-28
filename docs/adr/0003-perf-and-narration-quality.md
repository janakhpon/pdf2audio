# ADR 0003 — Performance and narration quality: fit-in-RAM model, faithful full mode, fail-loud merge

- **Status:** Accepted
- **Date:** 2026-07-11
- **Follows:** [ADR 0002](0002-staff-refactor.md) (craftsmanship refactor)
- **Context source:** an se-brain role-driven review (staff-engineer, ml-systems-architect,
  applied-ai-engineer, conversation-designer, ai-ml-engineer, qa-engineer, security-engineer
  lenses) that produced 18 verified findings. Finding-level detail is in the commit history.

## Context

On a 24 GB machine two problems surfaced in real use. A full audiobook took roughly a day, and
`full`-mode narration read as conversational meta-summaries ("Okay, here's a breakdown… 1. …")
rather than the source text. The root causes were independent: the default editor model
(`gemma3:27b`, ~17 GB) did not fit in RAM alongside the OS and the TTS engine, so Ollama paged it
to disk (~10x slower per call, one call per chunk); and the `full`-mode prompt neither forbade
preamble nor handled structured input, so the model summarized instead of narrating. The review
also surfaced correctness gaps around merging, TTS language, and config bounds.

## Decisions

1. **Default to a model that fits in RAM.** `gemma3:27b` → `qwen2.5:14b` (~9 GB). The editor makes
   one LLM call per chunk (hundreds per book); a model that does not fit swaps to disk and runs an
   order of magnitude slower. Pick the largest model that comfortably fits, not the largest that
   loads. The README documents the rule and lists RAM-sized options.

2. **Tune the Ollama request.** Set `keep_alive` so the model stays resident between chunks; set a
   `num_ctx` large enough for a chunk plus its rewrite (Ollama's small default silently truncated
   multi-KB chunks and dropped source content) and hold it **constant for the whole run** —
   Ollama reloads the model (~seconds) whenever `num_ctx` changes, so a per-chunk value would
   fight `keep_alive` and discard the cached prompt prefix; size it once from `chunk_size`. Use a
   low temperature for faithful rewrites; retry a timed-out call at most once (an identical retry
   just times out again); warn on `done_reason == "length"` or a prompt that overflows the window.

3. **Make `full` mode faithful narration.** `full` is a verbatim rewrite, not a summary. The prompt
   now narrates tables of contents, code, and lists as flowing prose instead of numbered
   "breakdowns"; forbids fabrication (a prior "ground every concept with real-world examples"
   instruction that invited invention was removed); forbids conversational preamble and
   meta-commentary; and explains the `<PREVIOUS_CONTEXT>` block so it is not re-narrated. A
   deterministic backstop strips any leaked preamble or sign-off, gated so it never deletes a
   legitimate opening sentence.

4. **Fail loudly on merge errors.** `merge_audio` logged and returned on ffmpeg failure, so a run
   could exit 0 with a success log and no `*_full` file, and the `MergeError` type was never
   raised. It now raises `MergeError` (the CLI maps it to exit 1). The run stays resumable: chunks
   remain `DONE`, so a re-run retries only the merge.

5. **Derive the TTS language from the voice.** `lang` was hardcoded to `en-us`, so the documented
   non-English voices were phonemized with English rules and came out garbled. It is now derived
   from the voice-name prefix, which makes the multilingual workflow in `voices.md` actually work.

6. **Match config bounds to the engine, and drop dead config.** `audio.speed` now rejects values
   above 2.0 (kokoro-onnx asserts `<= 2.0`; the old 3.0 ceiling failed every chunk mid-run).
   Removed `optimal_threads`, which was computed and logged but never applied to the ONNX session.

7. **Surface editor degradation.** Count chunks narrated from unpolished text and report the total
   at the end of a run, and validate Ollama eagerly (before the slow extraction) so a down or
   misconfigured editor gives fast feedback instead of surfacing only after a multi-minute PDF
   conversion.

## Consequences

- A book on a 24 GB machine completes in well under an hour instead of roughly a day; narration
  starts mid-content and stays faithful to the source.
- A merge or ffmpeg failure now fails the run (exit 1) rather than silently producing no audiobook.
- Non-English voices pronounce correctly; the "just set the voice" workflow holds.
- The quality gate (`ruff check` · `ruff format --check` · `mypy pdf2audio` · `pytest`) stays
  green; tests were added for the editor payload/prompt/stripping, audio chunking and language
  mapping, merge failure, and the corrected speed bound.
- Advances two `docs/audit.md` backlog items: LLM traceability (partial — `done_reason` is now
  surfaced) and the timeout-hang risk (a timeout is retried at most once).

## Deliberately not done (deferred, with rationale)

- **Re-polish degraded chunks on resume.** The run now reports how many chunks were unpolished, but
  re-polishing them on a later run needs a new `ChunkStatus` (e.g. `DEGRADED`) and a state-schema
  change. Deferred to a focused change so the schema migration is isolated.
- **Token-bounded chunking.** Chunks are still grouped by block count, not token budget. Sizing by
  tokens would change chunk boundaries and therefore the resume-state hash, so it is deferred; the
  `num_ctx` truncation warning is the interim guard.
- **Interrupt-time TTS drain.** Ctrl-C still lets the buffered TTS jobs (bounded queue) finish
  before exiting. Making the interrupt discard them is a threading change kept out of this pass.
