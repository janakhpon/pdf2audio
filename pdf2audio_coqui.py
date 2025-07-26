import os
import sys
import shutil
import tempfile
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Iterator

import torch
from PyPDF2 import PdfReader
from TTS.api import TTS
from pydub import AudioSegment
from tqdm import tqdm

# === CONFIG ===
CHUNK_SIZE = 4
MAX_TEXT_LENGTH = 5000
WORKERS = min(os.cpu_count() or 2, 6)
BITRATE = "128k"
SAMPLE_RATE = 24000
SILENCE_PADDING = 200
MODEL = "tts_models/en/vctk/vits"
SPEAKER = "p230"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("CoquiTTS")

class TTSWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
        self.tts = TTS(MODEL, progress_bar=False).to(self.device)

    def synthesize(self, text: str, path: str) -> bool:
        for _ in range(2):
            try:
                self.tts.tts_to_file(text=text, speaker=SPEAKER, file_path=path)
                if Path(path).exists() and os.path.getsize(path) > 1024:
                    return True
            except Exception as e:
                log.warning(f"TTS error: {e}")
                time.sleep(1.5)
        return False

def extract_chunks(pdf_path: Path) -> Iterator[Tuple[str, str]]:
    reader = PdfReader(str(pdf_path))
    for start in range(0, len(reader.pages), CHUNK_SIZE):
        pages = reader.pages[start:start + CHUNK_SIZE]
        chunk_id = f"{start+1}-{start+len(pages)}"
        text = "\n".join(p.extract_text() or "" for p in pages).strip()
        if not text:
            continue
        if len(text) <= MAX_TEXT_LENGTH:
            yield chunk_id, text
        else:
            words, chunk, count = text.split(), "", 0
            for word in words:
                if len(chunk) + len(word) + 1 > MAX_TEXT_LENGTH:
                    yield f"{chunk_id}.{count}", chunk.strip()
                    chunk, count = word + " ", count + 1
                else:
                    chunk += word + " "
            if chunk:
                yield f"{chunk_id}.{count}", chunk.strip()

def synthesize_chunk(cid: str, text: str, out_dir: Path, tts: TTSWorker) -> Optional[str]:
    if not text:
        return None
    path = out_dir / f"chunk_{cid}.mp3"
    return str(path) if tts.synthesize(text, str(path)) else None

def merge_audio(paths: List[str], output: Path):
    audio = AudioSegment.silent(duration=300)
    for i, p in enumerate(tqdm(paths, desc="Merging audio")):
        try:
            seg = AudioSegment.from_file(p)
            if len(seg) > 0:
                audio += seg
                if i < len(paths) - 1:
                    audio += AudioSegment.silent(duration=SILENCE_PADDING)
        except Exception as e:
            log.warning(f"Skipping {p}: {e}")
    audio.set_frame_rate(SAMPLE_RATE).export(output, format="mp3", bitrate=BITRATE)
    return output

def sort_key(p: str):
    parts = Path(p).stem.replace("chunk_", "").replace(".", "-").split("-")
    return list(map(int, parts))

def convert_pdf(pdf_file: str):
    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_file}")
    
    # Create audio directory if it doesn't exist
    audio_dir = Path("audio")
    audio_dir.mkdir(exist_ok=True)
    
    out_file = audio_dir / f"{pdf_path.stem}_audiobook.mp3"
    temp_dir = Path(tempfile.mkdtemp())

    try:
        chunks = list(extract_chunks(pdf_path))
        tts = TTSWorker()
        results = []

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(synthesize_chunk, cid, text, temp_dir, tts): cid for cid, text in chunks}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Synthesizing"):
                r = f.result()
                if r:
                    results.append(r)

        if not results:
            raise RuntimeError("No audio was generated.")
        results.sort(key=sort_key)
        merge_audio(results, out_file)
        log.info(f"✅ Audiobook saved to {out_file}")
    finally:
        shutil.rmtree(temp_dir)

def main():
    if len(sys.argv) != 2:
        print("Usage: python pdf2audio_coqui.py <file.pdf>")
        return
    convert_pdf(sys.argv[1])

if __name__ == "__main__":
    main()
