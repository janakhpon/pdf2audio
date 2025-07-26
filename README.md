# PDF to Audio Converter

A simple, open-source tool that converts PDFs into audiobooks for free.

## Prerequisites

- Python 3.11
- `uv` package manager (recommended) or `pip`

### Installing uv

```bash
# Install using curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
git clone git@github.com:janakhpon/pdf2audio.git
cd pdf2audio

uv venv --python=3.11 && source .venv/bin/activate 
uv pip install -r requirements.txt

python voice_test_coqui.py 
python voice_test_google.py 

python pdf2audio_google.py books/gold-rush-adventures.pdf  # gtts -> we can not change voice unless we have gtts api key
python pdf2audio_coqui.py books/gold-rush-adventures.pdf 
```


## References

- [TTS](https://github.com/coqui-ai/TTS)
- [gTTS](https://pypi.org/project/gTTS/)