import sys
from pathlib import Path
from src.config import load_config
from src.logger import logger
from src.audio import AudioEngine

def main():
    try:
        config = load_config("config.yaml")
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {e}")
        sys.exit(1)

    # Allow overriding the voice securely via command line argument
    if len(sys.argv) > 1:
        config.audio_voice = sys.argv[1].strip()

    logger.info(f"--- Starting Voice Preview for '{config.audio_voice}' ---")

    audio_engine = AudioEngine(config)
    
    preview_text = (
        "Hello, this is a sample of my voice. "
        "I will be your narrator for this audiobook. "
        "I hope you enjoy the experience."
    )
    
    # Export securely to the ignored output audio directory
    config.out_audio_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.out_audio_dir / f"_preview_{config.audio_voice}"
    
    try:
        audio_engine.generate(preview_text, output_path=output_path)
        logger.info(f"--- Preview successfully generated in {config.out_audio_dir}/ ---")
    except Exception as e:
        logger.error(f"Failed to generate preview audio: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
