# Voices and Languages

The `pdf2audio` pipeline uses Kokoro v1.0, an open-weights Text-to-Speech engine that supports realistic voices across multiple languages.

## Configuring Voices

Change the voice at any time in `config.yaml`:

```yaml
audio:
  voice: "af_heart" # see list below
  speed: 1.0 # 1.0 is normal; 1.1–1.2 is good for technical content
```

## Supported Languages and Voices

The voice ID prefix indicates language and gender:

- `a` — American English, `b` — British English
- `e` — Spanish, `f` — French, `h` — Hindi
- `i` — Italian, `j` — Japanese, `z` — Mandarin Chinese
- Second letter: `f` = female, `m` = male

### American English (Recommended for audiobooks)

| Voice                                                                | Character                                                  |
| -------------------------------------------------------------------- | ---------------------------------------------------------- |
| `af_heart`                                                           | (Default) Smooth, narrative — best for long-form listening |
| `af_alloy`                                                           | Crisp and professional                                     |
| `af_bella`                                                           | Warm                                                       |
| `af_nicole`                                                          | Conversational                                             |
| `am_adam`                                                            | Deep and clear                                             |
| `am_onyx`                                                            | Authoritative                                              |
| `am_michael`                                                         | Neutral male                                               |
| `af_aoede`, `af_jessica`, `af_kore`, `af_nova`, `af_river`, `af_sky` | Additional female options                                  |
| `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_puck`              | Additional male options                                    |

### British English

| Voice         | Character      |
| ------------- | -------------- |
| `bf_emma`     | British Female |
| `bf_isabella` | British Female |
| `bf_alice`    | British Female |
| `bf_lily`     | British Female |
| `bm_george`   | British Male   |
| `bm_fable`    | British Male   |
| `bm_lewis`    | British Male   |
| `bm_daniel`   | British Male   |

### Other Languages

The LLM editor is already instructed to preserve the source language — no config change needed. Just set the matching voice:

| Language | Voices                                               |
| -------- | ---------------------------------------------------- |
| Spanish  | `ef_dora`, `em_alex`, `em_santa`                     |
| French   | `ff_siwis`                                           |
| Hindi    | `hf_alpha`, `hf_beta`, `hm_omega`                    |
| Italian  | `if_sara`, `im_nicola`                               |
| Japanese | `jf_alpha`, `jf_gongitsune`, `jm_kumo`               |
| Mandarin | `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi` |

## Performance Tips

- **Speed `1.1–1.2`** — slightly faster narration, good for dense technical books
- **Speed `0.9`** — slower pace, better for complex concepts or language learners
- Voice quality is identical regardless of speed setting
