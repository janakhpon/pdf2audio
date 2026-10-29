# Smart Editor: models and behaviour

The Smart Editor is the optional step that rewrites each chunk into spoken narration with a local
Ollama model before synthesis. It drops page numbers, figure and table references and other layout
artifacts that only make sense on a page, and keeps the meaning. Turn it on with
`editor.enabled: true` in `config.yaml`.

## Pick a model that fits in RAM

The editor makes one LLM call per chunk, which is hundreds per book. If the model does not fit
alongside the OS and the TTS engine, Ollama pages it to disk and each call runs roughly 10x slower:
a book can take a full day instead of a few hours. On a 24 GB machine, `gemma3:27b` (~17 GB) swaps
once the OS, Python and Kokoro are loaded; `qwen2.5:14b` (~9 GB) leaves headroom. Pick the largest
model that comfortably fits, not the largest you can load.

| Model          | RAM needed | Best for                                                       |
| -------------- | ---------- | -------------------------------------------------------------- |
| `qwen2.5:14b`  | ~9 GB      | **Recommended default**: good quality, fits 16-24 GB machines  |
| `phi4:14b`     | ~9 GB      | Constrained hardware; strong for its size                      |
| `gemma3:27b`   | ~17 GB     | Excellent prose, but only if it fits (needs ~32 GB)            |
| `qwen2.5:72b`  | ~45 GB     | Best raw quality if you have the RAM                           |
| `llama3.3:70b` | ~45 GB     | Strong instruction-following, natural lecture tone             |

```bash
ollama pull qwen2.5:14b
```

```yaml
editor:
  enabled: true
  model: "qwen2.5:14b"
  mode: "full" # "full" = faithful full narration (not a summary); "medium"/"short" summarize
```

## What it keeps and what it drops

- **Dense technical books** come out shorter than the source. The editor drops page-only noise
  (figure and page references, hex and ID dumps, verbatim code), and that polished result is
  trusted and used, including legitimately short renderings such as an intro lead-in or a chapter
  summary.
- **The complete raw text is used only when the polish is broken:** empty, errored, truncated, or a
  near-empty fragment. The log then shows a `using the complete raw text` line. If the editor is
  unavailable, the pipeline narrates the extracted text, so nothing meaningful is lost.
- **Worked-example data** such as hex values may still be read aloud, because the model keeps it as
  content. A stronger model or a smaller `source.chunk_size` improves this.
- **Table-of-contents and index pages are skipped.** They are dot-leaders and page numbers with no
  spoken value, so they are detected and not narrated, as in a real audiobook.

See also [ADR 0003](adr/0003-perf-and-narration-quality.md) for why the model must fit in RAM and
why full mode narrates faithfully.
