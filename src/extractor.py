import re
from pathlib import Path
from pypdf import PdfReader
from .logger import logger

class PDFExtractor:
    def __init__(self, chunk_size: int = 5):
        self.chunk_size = chunk_size

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'-\s+', '', text)
        return text.strip()

    def process_file(self, pdf_path: Path):
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        logger.info(f"Extracting: {pdf_path.name}")
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        current_chunk = []
        count = 0
        
        for i, page in enumerate(reader.pages):
            if text := page.extract_text():
                current_chunk.append(self._clean_text(text))
            
            if (i + 1) % self.chunk_size == 0 or (i + 1) == total_pages:
                if chunk_text := " ".join(current_chunk).strip():
                    count += 1
                    yield chunk_text
                current_chunk.clear()
                
        logger.info(f"Extracted {count} chunks.")
