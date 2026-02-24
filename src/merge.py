import argparse
import os
import glob
from pydub import AudioSegment
from src.logger import logger

def merge_audio(directory: str, output_format: str = "mp3"):
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return

    # Find all audio chunks of the specified format
    pattern = os.path.join(directory, f"chunk_*.{output_format}")
    files = sorted(glob.glob(pattern))

    if not files:
        logger.error(f"No chunk files found in {directory}")
        return

    logger.info(f"Found {len(files)} files to merge in {directory}")

    combined = AudioSegment.empty()
    for f in files:
        logger.info(f"Adding {os.path.basename(f)} ...")
        # Attempt to infer format from extension or default to the requested output_format
        audio_ext = f.split(".")[-1]
        try:
            chunk = AudioSegment.from_file(f, format=audio_ext)
            combined += chunk
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
            return

    # Output file name: book_name.mp3
    book_name = os.path.basename(os.path.normpath(directory))
    parent_dir = os.path.dirname(os.path.normpath(directory))
    output_filename = os.path.join(parent_dir, f"{book_name}_full.{output_format}")

    logger.info(f"Exporting merged audio to {output_filename}...")
    combined.export(output_filename, format=output_format)
    logger.info(f"Export complete: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Merge audio chunks into a single audiobook file.")
    parser.add_argument("directory", help="The directory containing the audio chunks (e.g., output/audio/gold-rush-adventures)")
    parser.add_argument("--format", default="mp3", help="Output format (mp3, wav, m4a). Default is mp3.")

    args = parser.parse_args()
    merge_audio(args.directory, args.format)

if __name__ == "__main__":
    main()
