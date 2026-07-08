import sys

from pdf2audio.audio import AudioEngine
from pdf2audio.config import load_config
from pdf2audio.errors import PDF2AudioError
from pdf2audio.logger import logger


def main() -> None:
    try:
        config = load_config("config.yaml")
    except PDF2AudioError as exc:
        logger.error(f"Config error: {exc}")
        sys.exit(1)

    logger.info(f"Previewing Voice: {config.audio_voice}")

    try:
        audio_engine = AudioEngine(config)
    except PDF2AudioError as exc:
        logger.error(f"Audio engine error: {exc}")
        sys.exit(1)

    preview_text = "This is a sample of my voice. I will be your narrator."

    config.out_audio_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.out_audio_dir / f"_preview_{config.audio_voice}"

    try:
        audio_engine.generate(preview_text, output_path=output_path)
        logger.info(f"Preview generated in {config.out_audio_dir}/")
    except (PDF2AudioError, OSError) as exc:
        logger.error(f"Preview failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
