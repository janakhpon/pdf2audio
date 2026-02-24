import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    source_path: Path | list[Path]
    chunk_size: int
    editor_enabled: bool
    editor_model: str
    editor_mode: str
    editor_preserve_context: bool
    editor_url: str
    audio_model_path: str
    audio_voices_path: str
    audio_voice: str
    audio_speed: float
    audio_format: str
    out_audio_dir: Path
    out_transcripts_dir: Path
    save_transcripts: bool

def load_config(config_path: str | Path = "config.yaml") -> Config:
    """Loads the YAML configuration file and returns a validated Config dataclass."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_source = data.get("source", {}).get("path", "books/")
    parsed_source = [Path(p) for p in raw_source] if isinstance(raw_source, list) else Path(raw_source)

    return Config(
        source_path=parsed_source,
        chunk_size=data.get("source", {}).get("chunk_size", 5),
        
        editor_enabled=data.get("editor", {}).get("enabled", False),
        editor_model=data.get("editor", {}).get("model", "llama3.2"),
        editor_mode=data.get("editor", {}).get("mode", "full"),
        editor_preserve_context=data.get("editor", {}).get("preserve_context", True),
        editor_url=data.get("editor", {}).get("url", "http://localhost:11434"),
        
        audio_model_path=data.get("audio", {}).get("model_path", "kokoro-v1.0.onnx"),
        audio_voices_path=data.get("audio", {}).get("voices_path", "voices-v1.0.bin"),
        audio_voice=data.get("audio", {}).get("voice", "af_heart"),
        audio_speed=float(data.get("audio", {}).get("speed", 1.0)),
        audio_format=data.get("audio", {}).get("format", "wav").lower(),
        
        out_audio_dir=Path(data.get("output", {}).get("audio_dir", "output/audio")),
        out_transcripts_dir=Path(data.get("output", {}).get("transcripts_dir", "output/transcripts")),
        save_transcripts=data.get("output", {}).get("save_transcripts", True)
    )
