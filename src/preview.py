import sys
from pathlib import Path
from src.config import load_config
from src.logger import logger
from src.audio import AudioEngine

def main():
    try:
        config = load_config("config.yaml")
    except Exception as e:
        logger.error(f"Config error: {e}")
        sys.exit(1)

    logger.info(f"Previewing Voice: {config.audio_voice}")

    audio_engine = AudioEngine(config)
    preview_text = "This is a sample of my voice. I will be your narrator."
    
    config.out_audio_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.out_audio_dir / f"_preview_{config.audio_voice}"
    
    try:
        audio_engine.generate(preview_text, output_path=output_path)
        logger.info(f"Preview generated in {config.out_audio_dir}/")
    except Exception as e:
        logger.error(f"Preview failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
