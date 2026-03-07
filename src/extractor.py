import re
from pathlib import Path

from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter
import ebooklib
from ebooklib import epub

from src.logger import logger

class DocumentExtractor:
    def __init__(self, chunk_size: int = 5):
        self.chunk_size = chunk_size
        self._docling_converter = None

    def _get_docling_converter(self):
        if self._docling_converter is None:
            # Lazy load the Docling model only when needed
            logger.info("Initializing Docling DocumentConverter...")
            self._docling_converter = DocumentConverter()
        return self._docling_converter

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Remove consecutive whitespaces, including newlines
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def process_file(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Extracting: {file_path.name}")
        
        if file_path.is_dir():
            if list(file_path.glob("*.html")):
                yield from self._process_html_dir(file_path)
                return
            else:
                raise ValueError(f"Directory {file_path} does not contain HTML files.")

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            yield from self._process_pdf(file_path)
        elif suffix == ".epub":
            yield from self._process_epub(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _process_pdf(self, file_path: Path):
        # Using Docling to extract natively as Markdown
        converter = self._get_docling_converter()
        result = converter.convert(str(file_path))
        markdown_text = result.document.export_to_markdown()
        
        # Split markdown text into paragraphs or blocks
        blocks = [block.strip() for block in markdown_text.split('\n\n') if block.strip()]
        
        current_chunk = []
        count = 0
        
        for i, block in enumerate(blocks):
            current_chunk.append(self._clean_text(block))
            
            # Since Docling exports a single string, we use the logical chunks
            # as an analogue to pages, but we scale chunk size up since a markdown
            # block is usually much smaller than a PDF page.
            if (i + 1) % (self.chunk_size * 10) == 0 or (i + 1) == len(blocks):
                if chunk_text := " ".join(current_chunk).strip():
                    count += 1
                    yield chunk_text
                current_chunk.clear()
                
        logger.info(f"Extracted {count} chunks from PDF.")

    def _process_epub(self, file_path: Path):
        book = epub.read_epub(str(file_path))
        
        current_chunk = []
        count = 0
        chapter_idx = 0
        
        # Generator for clean text from epub items
        def extract_epub_text():
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    if text:
                        yield text

        for chapter_text in extract_epub_text():
            chapter_idx += 1
            current_chunk.append(self._clean_text(chapter_text))
            
            if chapter_idx % self.chunk_size == 0:
                if chunk_text := " ".join(current_chunk).strip():
                    count += 1
                    yield chunk_text
                current_chunk.clear()
                
        # Yield remainder
        if current_chunk:
            if chunk_text := " ".join(current_chunk).strip():
                count += 1
                yield chunk_text

        logger.info(f"Extracted {count} chunks from EPUB.")

    def _process_html_dir(self, dir_path: Path):
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s.name)]
            
        html_files = sorted(list(dir_path.glob("*.html")), key=natural_sort_key)
        
        current_chunk = []
        count = 0
        file_idx = 0
        
        def extract_html_text():
            for html_file in html_files:
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    
                    # Remove non-content elements
                    for tag in soup(['nav', 'header', 'footer', 'script', 'style', 'aside', 'noscript', 'form']):
                        tag.decompose()
                        
                    text = soup.get_text(separator=' ', strip=True)
                    if text:
                        yield text

        for chapter_text in extract_html_text():
            file_idx += 1
            current_chunk.append(self._clean_text(chapter_text))
            
            if file_idx % self.chunk_size == 0:
                if chunk_text := " ".join(current_chunk).strip():
                    count += 1
                    yield chunk_text
                current_chunk.clear()
                
        # Yield remainder
        if current_chunk:
            if chunk_text := " ".join(current_chunk).strip():
                count += 1
                yield chunk_text

        logger.info(f"Extracted {count} chunks from HTML directory.")
