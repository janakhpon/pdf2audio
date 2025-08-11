import re
import gc
import logging
import time
from pathlib import Path
from typing import List, Optional
from functools import wraps

import torch
from PyPDF2 import PdfReader
from TTS.api import TTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# optional fallback
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

def retry(max_retries=3, delay=1.0):
    """simple retry decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

def cleanup_memory():
    # cleanup only unused memory
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

class Config:
    def __init__(self, summary_style="medium", force_cpu=False):
        self.llm_model = "facebook/bart-large-cnn"
        self.tts_model = "tts_models/en/vctk/vits"
        self.speaker = "p230"
        self.summary_style = summary_style
        self.bitrate = "128k"
        self.force_cpu = force_cpu
        
        # summary ratios
        self.ratios = {"short": 0.4, "medium": 0.6, "long": 0.8}
        self.ratio = self.ratios.get(summary_style, 0.6)

class Chapter:
    def __init__(self, num, content, pages):
        self.num = num
        self.content = content
        self.pages = pages
        self.summary = None

class PDFProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract_chapters(self, pdf_path):
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        
        chunk_size = 4 if total_pages <= 200 else 6
        self.logger.info(f"processing {total_pages} pages")
        
        chapters = []
        for start in range(0, total_pages, chunk_size):
            end = min(start + chunk_size, total_pages)
            text_parts = []
            
            for i in range(start, end):
                try:
                    text = reader.pages[i].extract_text()
                    if text.strip():
                        cleaned = self._clean_text(text)
                        if cleaned:
                            text_parts.append(cleaned)
                except Exception:
                    continue
            
            if text_parts:
                content = "\n\n".join(text_parts)
                chapter = Chapter(len(chapters) + 1, content, (start + 1, end))
                chapters.append(chapter)
        
        return chapters
    
    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()

class Summarizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load_model(self):
        if self._loaded:
            return True
        
        try:
            self.logger.info("loading llm model")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.llm_model,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            self._loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"llm model loading failed: {e}")
            return False
    
    def summarize_chapter(self, chapter, target_length):
        if self.model:
            chapter.summary = self._llm_summarize(chapter.content, target_length)
        
        if not chapter.summary:
            chapter.summary = self._extractive_summary(chapter.content, target_length)
        
        return chapter
    
    def _llm_summarize(self, text, target_length):
        try:
            # limit input length
            if len(text) > 4000:
                text = text[:4000] + "..."
            
            prompt = f"Summarize the key points: {text}"
            
            inputs = self.tokenizer.encode(
                prompt, return_tensors="pt", max_length=1024, truncation=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=min(target_length, 512),
                    min_length=100,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = self._clean_summary(summary)
            
            return summary if len(summary) > 50 else ""
            
        except Exception as e:
            self.logger.warning(f"llm summarization failed: {e}")
            return ""
    
    def _extractive_summary(self, text, target_length):
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 30]
        
        if not sentences:
            return text[:target_length] + "..."
        
        num_sentences = min(len(sentences), target_length // 50)
        selected = sentences[:num_sentences]
        
        if len(sentences) > num_sentences:
            mid_idx = len(sentences) // 2
            end_idx = len(sentences) - 1
            
            if mid_idx not in range(len(selected)):
                selected.append(sentences[mid_idx])
            if end_idx not in range(len(selected)):
                selected.append(sentences[end_idx])
        
        return ". ".join(selected) + "."
    
    def _clean_summary(self, text):
        if not text:
            return ""
        
        text = text.strip()
        prefixes = ["summary:", "summarize:", "overview:"]
        
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text

class AudioGenerator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tts = None
        self.device = "cpu" if config.force_cpu else self._get_device()
        self._loaded = False
    
    def _get_device(self):
        try:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    
    @retry(max_retries=2, delay=2.0)
    def load_model(self):
        if self._loaded:
            return True
        
        try:
            cleanup_memory()
            
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="jieba")
            
            self.tts = TTS(self.config.tts_model, progress_bar=False).to(self.device)
            self._loaded = True
            
            self.logger.info(f"tts model loaded on {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"tts model loading failed: {e}")
            raise
    
    @retry(max_retries=2, delay=1.0)
    def generate_audio(self, text, output_path):
        if not text.strip():
            return False
        
        # limit text length
        if len(text) > 10000:
            self.logger.warning("text too long, truncating")
            text = text[:10000] + "..."
        
        try:
            cleanup_memory()
            
            self.tts.tts_to_file(
                text=text, 
                speaker=self.config.speaker, 
                file_path=str(output_path)
            )
            
            if output_path.exists() and output_path.stat().st_size > 1024:
                self.logger.info(f"audio saved: {output_path}")
                return True
            else:
                self.logger.warning("invalid output file")
                return False
                
        except Exception as e:
            self.logger.warning(f"audio generation failed: {e}")
            return self._fallback_tts(text, output_path)
    
    def _fallback_tts(self, text, output_path):
        if not GTTS_AVAILABLE:
            return False
        
        try:
            self.logger.info("trying google tts fallback")
            
            if len(text) > 5000:
                text = text[:5000] + "..."
            
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(str(output_path))
            
            if output_path.exists() and output_path.stat().st_size > 1000:
                self.logger.info("google tts audio saved")
                return True
                
        except Exception as e:
            self.logger.warning(f"google tts failed: {e}")
        
        return False

class PDFSummarizer:
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        self.pdf_processor = PDFProcessor(config)
        self.summarizer = Summarizer(config)
        self.audio_generator = AudioGenerator(config)
    
    def _setup_logging(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )
        logger.handlers.clear()
        logger.addHandler(handler)
        return logger
    
    def process(self, pdf_path):
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"file not found: {pdf_path}")
        
        # create output directories
        summaries_dir = Path("summaries")
        audio_dir = Path("audio_summaries")
        summaries_dir.mkdir(exist_ok=True)
        audio_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"processing {pdf_file.name}")
        
        try:
            # extract chapters
            chapters = self.pdf_processor.extract_chapters(pdf_file)
            if not chapters:
                raise RuntimeError("no content extracted")
            
            # calculate target length
            total_pages = sum(ch.pages[1] - ch.pages[0] + 1 for ch in chapters)
            target_length = int(total_pages * self.config.ratio * 400)
            
            self.logger.info(f"target length: {target_length} tokens")
            
            # load models
            if not self.summarizer.load_model():
                self.logger.warning("using extractive summaries")
            
            # summarize chapters
            self.logger.info(f"summarizing {len(chapters)} chapters")
            successful = 0
            
            for chapter in tqdm(chapters, desc="summarizing"):
                try:
                    self.summarizer.summarize_chapter(chapter, target_length // len(chapters))
                    if chapter.summary and len(chapter.summary.strip()) > 50:
                        successful += 1
                    
                    # cleanup every 10 chapters
                    if chapter.num % 10 == 0:
                        cleanup_memory()
                        
                except Exception as e:
                    self.logger.error(f"chapter {chapter.num} failed: {e}")
                    continue
            
            self.logger.info(f"processed {successful}/{len(chapters)} chapters")
            
            # save summaries
            text_output = self._save_summaries(chapters, summaries_dir, pdf_file.stem)
            
            # generate audio
            audio_output = self._generate_audio(chapters, audio_dir, pdf_file.stem)
            
            return str(text_output), str(audio_output) if audio_output else "audio generation failed"
            
        except Exception as e:
            self.logger.error(f"processing failed: {e}")
            raise
    
    def _save_summaries(self, chapters, output_dir, name):
        output_file = output_dir / f"{name}_summaries.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"BOOK SUMMARY: {name}\n")
            f.write("="*60 + "\n\n")
            
            total_chapters = len([ch for ch in chapters if ch.summary and len(ch.summary.strip()) > 50])
            f.write(f"Total Chapters: {len(chapters)}\n")
            f.write(f"Summarized: {total_chapters}\n")
            f.write(f"Style: {self.config.summary_style}\n\n")
            f.write("="*60 + "\n\n")
            
            for chapter in chapters:
                f.write(f"CHAPTER {chapter.num} (Pages {chapter.pages[0]}-{chapter.pages[1]})\n")
                f.write("-"*50 + "\n")
                if chapter.summary and len(chapter.summary.strip()) > 50:
                    f.write(f"{chapter.summary}\n")
                else:
                    f.write("Summary not available.\n")
                f.write("\n" + "="*60 + "\n\n")
        
        self.logger.info(f"summaries saved: {output_file}")
        return output_file
    
    def _generate_audio(self, chapters, audio_dir, name):
        # create audio content
        audio_text = self._create_audio_content(chapters)
        
        if not audio_text:
            self.logger.warning("no content for audio")
            return None
        
        # save transcript
        transcript_file = audio_dir / f"{name}_transcript.txt"
        try:
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(f"TRANSCRIPT: {name}\n")
                f.write("="*60 + "\n\n")
                f.write(audio_text)
            self.logger.info(f"transcript saved: {transcript_file}")
        except Exception as e:
            self.logger.warning(f"transcript save failed: {e}")
        
        # load tts model
        if not self.audio_generator.load_model():
            self.logger.error("tts model failed to load")
            return None
        
        # generate audio
        output_file = audio_dir / f"{name}_audiobook.mp3"
        
        if self.audio_generator.generate_audio(audio_text, output_file):
            return output_file
        
        return None
    
    def _create_audio_content(self, chapters):
        valid_chapters = [ch for ch in chapters if ch.summary and len(ch.summary.strip()) > 100]
        
        if not valid_chapters:
            return ""
        
        self.logger.info(f"creating audio from {len(valid_chapters)} chapters")
        
        parts = []
        parts.append(f"Welcome to this book summary. This book contains {len(chapters)} chapters with insights and takeaways.")
        
        for i, chapter in enumerate(valid_chapters):
            clean_summary = self._clean_summary(chapter.summary)
            if clean_summary:
                if i == 0:
                    parts.append(f"Chapter {chapter.num}, pages {chapter.pages[0]} to {chapter.pages[1]}: {clean_summary}")
                else:
                    parts.append(f"Chapter {chapter.num}, pages {chapter.pages[0]} to {chapter.pages[1]}: {clean_summary}")
        
        parts.append("This concludes our book summary. We've covered the essential concepts and insights from each chapter.")
        
        transcript = ". ".join(parts)
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        
        return transcript
    
    def _clean_summary(self, text):
        if not text:
            return ""
        
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text

def main():
    import sys
    
    pdf_path = None
    for arg in sys.argv[1:]:
        if arg.endswith(".pdf"):
            pdf_path = arg
            break
    
    if not pdf_path:
        pdf_files = list(Path(".").glob("*.pdf"))
        if not pdf_files:
            print("usage: python pdf2audio_summarizer.py <file.pdf> [--short|--medium|--long] [--cpu]")
            print("or place a pdf file in current directory")
            sys.exit(1)
        pdf_path = pdf_files[0]
        print(f"using: {pdf_path}")
    
    if not Path(pdf_path).exists():
        print(f"error: {pdf_path} not found")
        sys.exit(1)
    
    # config
    args = sys.argv[1:]
    summary_style = "medium"
    if "--short" in args:
        summary_style = "short"
    elif "--long" in args:
        summary_style = "long"
    
    force_cpu = "--cpu" in args
    
    config = Config(summary_style, force_cpu)
    
    print(f"processing {Path(pdf_path).name} ({summary_style} summaries)")
    
    summarizer = PDFSummarizer(config)
    
    try:
        start = time.time()
        text_output, audio_output = summarizer.process(pdf_path)
        elapsed = time.time() - start
        
        print(f"\ncompleted in {elapsed:.1f}s")
        print(f"Summaries: {text_output}")
        print(f"Audio: {audio_output}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
