import re
from pathlib import Path
from pypdf import PdfReader
from .logger import logger

class PDFExtractor:
    """Handles extracting and cleaning text from PDF files."""

    def __init__(self, chunk_size: int = 5):
        self.chunk_size = chunk_size

    def _clean_text(self, text: str) -> str:
        """Removes excessive whitespace, newlines, and common artifacts from PDF extraction."""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'-\s+', '', text)
        
        return text.strip()

    def process_file(self, pdf_path: Path):
        """
        Extracts content from a PDF and cleans it, yielding chunks of text 
        corresponding to configured chunk sizes. This generator ensures O(1) memory usage 
        regardless of PDF size.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting text from PDF: {pdf_path.name}")
        
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        current_chunk = []
        chunks_yielded = 0
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                current_chunk.append(self._clean_text(text))
            
            if (i + 1) % self.chunk_size == 0 or (i + 1) == total_pages:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks_yielded += 1
                    yield chunk_text
                # Clear the memory for the next chunk
                current_chunk.clear()
                
        logger.info(f"Completed extraction. Yielded {chunks_yielded} total chunks.")
