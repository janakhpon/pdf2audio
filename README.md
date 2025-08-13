# PDF to Audio Converter

A simple, open-source tool that converts PDFs into audiobooks for free.

## Prerequisites

- Python 3.11+
- `uv` package manager

### Installing uv

```bash
# Install using curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
git clone git@github.com:janakhpon/pdf2audio.git
cd pdf2audio

# uv venv --python=3.11

# Create virtual environment and install dependencies
uv sync


# Test voice synthesis
uv run python voice_test_coqui.py
uv run python voice_test_google.py

# Convert PDFs to audio
uv run python pdf2audio_google.py books/gold-rush-adventures.pdf  # gtts -> we can not change voice unless we have gtts api key
uv run python pdf2audio_coqui.py books/gold-rush-adventures.pdf
uv run python pdf2audio_llm.py books/gold-rush-adventures.pdf  # content parsing and enhancement with GPT-2 Medium
uv run python pdf2audio_summarizer.py book.pdf # medium summary is default, provide --long or --short
```

## References

- [TTS](https://github.com/coqui-ai/TTS)
- [gTTS](https://pypi.org/project/gTTS/)
- [Transformers](https://huggingface.co/docs/transformers/)
- [GPT-2 model on Hugging Face](https://huggingface.co/openai-community/gpt2-medium/tree/main)
- [Facebook Bart](https://huggingface.co/facebook/bart-large-cnn)
