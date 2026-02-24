# Voices and Languages

The `pdf2audio` pipeline uses Kokoro v1.0, an open-weights Text-to-Speech (TTS) engine that supports realistic voices across multiple languages.

## Configuring Voices

You can change the voice at any time by modifying `audio.voice` in `config.yaml`:

```yaml
audio:
  voice: "af_heart" # Change this to any of the supported voices below
  speed: 1.0 # Set speaking speed (1.0 is default, 1.2 is slightly faster)
```

## Supported Languages and Voices

Kokoro supports multiple languages. The first letter of the voice ID indicates its language mapping:

- `a`: American English
- `b`: British English
- `e`: Spanish
- `f`: French
- `h`: Hindi
- `i`: Italian
- `j`: Japanese
- `z`: Mandarin Chinese

The second letter implies the gender (`f` for female, `m` for male).

### American English (Recommended)

Excellent for standard audiobook generation.

- **`af_heart`**: (Default) American Female, smooth and narrative.
- **`af_alloy`**: American Female, crisp and professional.
- **`af_aoede`**: American Female.
- **`af_bella`**: American Female, warm.
- **`af_jessica`**: American Female.
- **`af_kore`**: American Female.
- **`af_nicole`**: American Female, conversational.
- **`af_nova`**: American Female.
- **`af_river`**: American Female.
- **`af_sky`**: American Female.
- **`am_adam`**: American Male, deep and clear.
- **`am_echo`**: American Male.
- **`am_eric`**: American Male.
- **`am_fenrir`**: American Male.
- **`am_liam`**: American Male.
- **`am_michael`**: American Male.
- **`am_onyx`**: American Male, authoritative.
- **`am_puck`**: American Male.

### British English

- **`bf_emma`**: British Female.
- **`bf_isabella`**: British Female.
- **`bf_alice`**: British Female.
- **`bf_lily`**: British Female.
- **`bm_george`**: British Male.
- **`bm_fable`**: British Male.
- **`bm_lewis`**: British Male.
- **`bm_daniel`**: British Male.

### Other Languages

To narrate a PDF written in a different language, ensure the Ollama Smart Editor logic in `config.yaml` is disabled or modified to support that language, and simply pass the appropriate voice ID:

- **Spanish**: `ef_dora`, `em_alex`, `em_santa`
- **French**: `ff_siwis`
- **Hindi**: `hm_omega`, `hf_alpha`, `hf_beta`
- **Italian**: `if_sara`, `im_nicola`
- **Japanese**: `jf_alpha`, `jf_gongitsune`, `jm_kumo`
- **Mandarin Chinese**: `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`

## Performance Tuning

If narration feels slightly too slow (a common feedback loop for Audiobooks), simply increase `audio.speed` in your `config.yaml` to `1.1` or `1.2`.
