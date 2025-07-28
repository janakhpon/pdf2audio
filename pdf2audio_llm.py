import os
import sys
import shutil
import tempfile
import logging
import time
import re
import json
import signal
import gc
import psutil
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import List, Tuple, Optional, Iterator, NamedTuple, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import lru_cache, wraps

import torch
from PyPDF2 import PdfReader
from TTS.api import TTS
from pydub import AudioSegment
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

class PerformanceMonitor:
    
    def __init__(self):
        self.start_time = time.time()
        self.memory_peak = 0
        self.cpu_peak = 0
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        while self._monitoring:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                self.memory_peak = max(self.memory_peak, memory_mb)
                self.cpu_peak = max(self.cpu_peak, cpu_percent)
                time.sleep(5)
            except Exception:
                break
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            process = psutil.Process()
            return {
                "elapsed_time": time.time() - self.start_time,
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "memory_peak_mb": self.memory_peak,
                "cpu_peak_percent": self.cpu_peak
            }
        except Exception:
            return {"error": "could not retrieve stats"}

@dataclass
class Config:
    chunk_size: int = 5
    max_text_length: int = 6000
    chapter_min_length: int = 200
    
    bitrate: str = "128k"
    sample_rate: int = 24000
    silence_padding: int = 200
    
    tts_model: str = "tts_models/en/vctk/vits"
    speaker: str = "p230"
    
    llm_model: str = "gpt2-medium"
    max_new_tokens: int = 300
    temperature: float = 0.7
    
    workers: int = field(default_factory=lambda: min(os.cpu_count() or 2, 6))
    add_book_announcements: bool = True
    add_intro_outro: bool = True
    
    enable_llm_enhancement: bool = True
    memory_management_interval: int = 30
    progress_log_interval: int = 5
    
    model_cache_dir: str = "./models"
    max_memory_mb: int = 8192
    timeout_seconds: int = 600
    
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """validate config"""
        if not (1 <= self.chunk_size <= 20):
            raise ValueError(f"chunk_size must be 1-20, got {self.chunk_size}")
        if not (1000 <= self.max_text_length <= 15000):
            raise ValueError(f"max_text_length must be 1000-15000, got {self.max_text_length}")
        if not (8000 <= self.sample_rate <= 48000):
            raise ValueError(f"sample_rate must be 8000-48000, got {self.sample_rate}")
        if not (50 <= self.max_new_tokens <= 1000):
            raise ValueError(f"max_new_tokens must be 50-1000, got {self.max_new_tokens}")
        if not (0.1 <= self.temperature <= 1.0):
            raise ValueError(f"temperature must be 0.1-1.0, got {self.temperature}")
        if not (10 <= self.memory_management_interval <= 100):
            raise ValueError(f"memory_management_interval must be 10-100, got {self.memory_management_interval}")
        if not (1 <= self.progress_log_interval <= 20):
            raise ValueError(f"progress_log_interval must be 1-20, got {self.progress_log_interval}")
        if not (1024 <= self.max_memory_mb <= 32768):
            raise ValueError(f"max_memory_mb must be 1024-32768, got {self.max_memory_mb}")
        if not (60 <= self.timeout_seconds <= 3600):
            raise ValueError(f"timeout_seconds must be 60-3600, got {self.timeout_seconds}")
        if not (1 <= self.max_retries <= 10):
            raise ValueError(f"max_retries must be 1-10, got {self.max_retries}")
        if not (0.1 <= self.retry_delay <= 10.0):
            raise ValueError(f"retry_delay must be 0.1-10.0, got {self.retry_delay}")

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level.upper()))

    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

@contextmanager
def resource_manager(monitor: PerformanceMonitor, logger: logging.Logger):
    """context manager for resource monitoring."""
    monitor.start_monitoring()
    start_stats = monitor.get_stats()
    
    try:
        yield
    finally:
        monitor.stop_monitoring()
        end_stats = monitor.get_stats()
        
        logger.info(f"Resource usage: Memory: {end_stats.get('memory_mb', 0):.1f}MB, "
                   f"CPU: {end_stats.get('cpu_percent', 0):.1f}%, "
                   f"Time: {end_stats.get('elapsed_time', 0):.1f}s")

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """decorator for retrying operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logging.getLogger(__name__).warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logging.getLogger(__name__).error(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator

class MemoryManager:
    @staticmethod
    def check_memory_usage(max_memory_mb: int) -> bool:
        """check if memory usage is within limits."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb < max_memory_mb
        except Exception:
            return True  # assume ok if we can't check
    
    @staticmethod
    def force_cleanup():
        """force garbage collection and memory cleanup."""
        gc.collect()
        if hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """get detailed memory information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except Exception:
            return {"error": "could not retrieve memory info"}

class TextChunk(NamedTuple):
    """processed text chunk with metadata."""
    id: str
    title: str
    content: str
    page_range: Optional[Tuple[int, int]] = None
    processing_time: Optional[float] = None
    word_count: Optional[int] = None
    
    def __post_init__(self):
        """calculate word count if not provided."""
        if self.word_count is None:
            self.word_count = len(self.content.split()) if self.content else 0

class ProcessingStats:
    """tracks processing statistics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.chunks_processed = 0
        self.chunks_failed = 0
        self.total_words = 0
        self.total_chars = 0
        self.processing_times = []
        self.errors = []
    
    def add_chunk(self, chunk: TextChunk, processing_time: float, success: bool = True):
        """add chunk processing result."""
        if success:
            self.chunks_processed += 1
            self.total_words += chunk.word_count or 0
            self.total_chars += len(chunk.content)
        else:
            self.chunks_failed += 1
        
        self.processing_times.append(processing_time)
    
    def add_error(self, error: Exception, context: str = ""):
        """add error information."""
        self.errors.append({
            "error": str(error),
            "type": type(error).__name__,
            "context": context,
            "timestamp": time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """get processing summary."""
        elapsed_time = time.time() - self.start_time
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            "elapsed_time": elapsed_time,
            "chunks_processed": self.chunks_processed,
            "chunks_failed": self.chunks_failed,
            "success_rate": self.chunks_processed / (self.chunks_processed + self.chunks_failed) if (self.chunks_processed + self.chunks_failed) > 0 else 0,
            "total_words": self.total_words,
            "total_chars": self.total_chars,
            "avg_processing_time": avg_processing_time,
            "words_per_second": self.total_words / elapsed_time if elapsed_time > 0 else 0,
            "error_count": len(self.errors)
        }

config = Config()

logger = setup_logging("INFO", "pdf2audio_llm.log")

class ModelManager:
    
    def __init__(self, cache_dir: str, config: Config):
        self.model_name = config.llm_model
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.context_length = 1024
        self._model_loaded = False
        self._load_lock = threading.Lock()
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_device(self) -> str:
        try:
            if torch.cuda.is_available():
                if torch.cuda.get_device_properties(0).total_memory > 4 * 1024**3:  # 4gb
                    return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    
    @retry_with_backoff(max_retries=2, base_delay=2.0)
    def load(self) -> bool:
        """load the model and tokenizer."""
        with self._load_lock:
            if self._model_loaded:
                return True
            
            try:
                logging.getLogger(__name__).info(f"Loading LLM model: {self.model_name}")
                
                if not MemoryManager.check_memory_usage(self.config.max_memory_mb):
                    raise RuntimeError(f"Insufficient memory. Required: {self.config.max_memory_mb}MB")
                
                logging.getLogger(__name__).info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    cache_dir=str(self.cache_dir),
                    local_files_only=False
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                logging.getLogger(__name__).info("Loading model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    cache_dir=str(self.cache_dir),
                    local_files_only=False,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
                
                self.context_length = self._get_context_length()
                
                self._model_loaded = True
                logging.getLogger(__name__).info(
                    f"Model loaded successfully on {self.device}, "
                    f"context length: {self.context_length}"
                )
                return True
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to load model: {e}")
                self._cleanup()
                return False
    
    def _cleanup(self):
        """clean up model resources"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        MemoryManager.force_cleanup()
        self._model_loaded = False
    
    def _get_context_length(self) -> int:
        """get model context length"""
        if not self.model:
            return 1024
        
        config = self.model.config
        for attr in ['max_position_embeddings', 'n_positions', 'max_sequence_length']:
            if hasattr(config, attr):
                return getattr(config, attr)
        return 1024
    
    def get_max_input_length(self, max_new_tokens: int = 300) -> int:
        """calculate max input length"""
        max_length = self.context_length - max_new_tokens - 100
        return max(512, max_length)
    
    def __del__(self):
        """cleanup on destruction"""
        self._cleanup()


class TextProcessor:
    
    def __init__(self, model_manager: ModelManager, config: Config):
        self.model_manager = model_manager
        self.config = config
        self._cleaning_patterns = self._compile_cleaning_patterns()
        self._cache = {}
        self._cache_size = 1000
        self._cache_lock = threading.Lock()
    
    def _compile_cleaning_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """compile regex patterns"""
        return [
            (re.compile(r'\b\d+\s*$', re.MULTILINE), ''),  # page numbers
            (re.compile(r'^\d+\s*', re.MULTILINE), ''),     # page numbers
            (re.compile(r'[A-Z]{2,}\s*\d+'), ''),          # headers
            (re.compile(r'©\s*\d{4}.*?\.'), ''),           # copyright
            (re.compile(r'www\.[^\s]+'), ''),               # urls
            (re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)]'), ''), # special chars
            (re.compile(r'\s+'), ' '),                      # spaces
            (re.compile(r'\n\s*\n'), '\n\n'),              # newlines
        ]
    
    @lru_cache(maxsize=1000)
    def _rule_based_cleaning(self, text: str) -> str:
        """apply rule-based cleaning"""
        if not text or not text.strip():
            return ""
        
        cleaned = text
        for pattern, replacement in self._cleaning_patterns:
            cleaned = pattern.sub(replacement, cleaned)
        
        return re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE).strip()
    
    def process_text(self, text: str) -> str:
        """process text with llm enhancement"""
        if not text or not text.strip():
            return ""
        
        cache_key = hash(text)
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        cleaned_text = self._rule_based_cleaning(text)
        
        if not self.model_manager.model or not self.config.enable_llm_enhancement:
            result = cleaned_text
        else:
            try:
                max_chars = self.model_manager.get_max_input_length() * 4
                result = (self._process_single_chunk(cleaned_text) if len(cleaned_text) <= max_chars 
                         else self._process_chunked_text(cleaned_text, max_chars))
            except Exception as e:
                logging.getLogger(__name__).warning(f"llm failed: {e}")
                result = cleaned_text
        
        with self._cache_lock:
            if len(self._cache) >= self._cache_size:
                for key in list(self._cache.keys())[:100]:
                    del self._cache[key]
            self._cache[cache_key] = result
        
        return result
    
    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _process_single_chunk(self, text: str) -> str:
        """process single chunk"""
        try:
            cleaned = self._llm_clean(text)
            if not cleaned or len(cleaned.strip()) < 10:
                cleaned = self._rule_based_cleaning(text)
            
            if len(cleaned) > 20:
                enhanced = self._llm_enhance(cleaned)
                if enhanced and len(enhanced.strip()) > 10:
                    return enhanced
            
            return cleaned
        except Exception as e:
            logging.getLogger(__name__).warning(f"chunk processing failed: {e}")
            return self._rule_based_cleaning(text)
    
    def _process_chunked_text(self, text: str, max_chars: int) -> str:
        """process text in chunks"""
        sentences = text.split('. ')
        chunks = self._create_overlapping_chunks(sentences, max_chars)
        
        if not chunks:
            return self._rule_based_cleaning(text)
        
        processed_chunks = []
        for chunk in chunks:
            processed = self._process_single_chunk(chunk)
            if processed:
                processed_chunks.append(processed)
        
        return self._merge_chunks(processed_chunks) if processed_chunks else self._rule_based_cleaning(text)
    
    def _create_overlapping_chunks(self, sentences: List[str], max_chars: int) -> List[str]:
        """create overlapping chunks"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) + 2 > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_sentences = current_chunk.split('. ')[-2:]
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
            else:
                current_chunk += sentence + '. '
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _merge_chunks(self, chunks: List[str]) -> str:
        """merge chunks, remove overlaps"""
        if len(chunks) == 1:
            return chunks[0]
        
        merged = chunks[0]
        for chunk in chunks[1:]:
            overlap = self._find_overlap(merged, chunk)
            if overlap:
                merged += chunk[len(overlap):]
            else:
                merged += ' ' + chunk
        
        return merged
    
    def _find_overlap(self, text1: str, text2: str) -> str:
        """find overlapping content"""
        min_overlap = 50
        if len(text1) < min_overlap or len(text2) < min_overlap:
            return ""
        
        sentences1 = text1.split('. ')
        sentences2 = text2.split('. ')
        
        for i in range(min(3, len(sentences1))):
            end_sentences = '. '.join(sentences1[-(i+1):])
            for j in range(min(3, len(sentences2))):
                start_sentences = '. '.join(sentences2[:j+1])
                if end_sentences == start_sentences and len(end_sentences) >= min_overlap:
                    return end_sentences
        
        return ""
    
    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def _llm_clean(self, text: str) -> str:
        """clean text using llm"""
        if not text or len(text) < 10:
            return text
            
        prompt = f"Clean this text for audiobook narration. Remove formatting artifacts, fix punctuation, ensure smooth reading flow, and maintain the original meaning:\n\n{text[:1800]}"
        return self._generate_text(prompt, max_new_tokens=200, temperature=0.4)
    
    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def _llm_enhance(self, text: str) -> str:
        """enhance text using llm"""
        if not text or len(text) < 10:
            return text
            
        prompt = f"Enhance this text for audiobook narration. Make it engaging, clear, well-paced for listening, and maintain the technical accuracy:\n\n{text[:1800]}"
        return self._generate_text(prompt, max_new_tokens=250, temperature=0.7)
    
    def _generate_text(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """generate text using llm"""
        try:
            max_input_length = self.model_manager.get_max_input_length(max_new_tokens)
            
            inputs = self.model_manager.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=max_input_length,
                add_special_tokens=True
            )
            
            attention_mask = torch.ones_like(inputs)
            inputs = inputs.to(self.model_manager.device)
            attention_mask = attention_mask.to(self.model_manager.device)
            
            with torch.no_grad():
                try:
                    outputs = self.model_manager.model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.model_manager.tokenizer.pad_token_id,
                        eos_token_id=self.model_manager.tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                        use_cache=True,
                        num_return_sequences=1
                    )
                except Exception as e:
                    logging.getLogger(__name__).warning(f"generation failed: {e}")
                    return ""
            
            generated = self.model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if ":" in generated:
                parts = generated.split(":", 1)
                if len(parts) > 1:
                    generated = parts[-1].strip()
            
            generated = generated.replace('\n', ' ').replace('  ', ' ')
            
            sentences = [s.strip() for s in generated.split('. ') if s.strip()]
            unique_sentences = list(dict.fromkeys(sentences))
            
            result = '. '.join(unique_sentences) if unique_sentences else ""
            
            return result if len(result) >= 10 else ""
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"generation failed: {e}")
            return ""
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """truncate prompt to fit token limit"""
        instruction_end = prompt.find('\n\n')
        if instruction_end == -1:
            return self.model_manager.tokenizer.decode(
                self.model_manager.tokenizer.encode(prompt, max_length=max_tokens, truncation=True),
                skip_special_tokens=True
            )
        
        instruction = prompt[:instruction_end + 2]
        text_part = prompt[instruction_end + 2:]
        
        truncated_text = self.model_manager.tokenizer.decode(
            self.model_manager.tokenizer.encode(text_part, max_length=max_tokens, truncation=True),
            skip_special_tokens=True
        )
        
        return instruction + truncated_text


class TTSProcessor:
    """tts processing"""
    
    def __init__(self, model_name: str, speaker: str, config: Config):
        self.model_name = model_name
        self.speaker = speaker
        self.config = config
        self.device = self._get_device()
        self.tts = None
        self._model_loaded = False
        self._load_lock = threading.Lock()
        self._synthesis_cache = {}
        self._cache_size = 100
        self._cache_lock = threading.Lock()
    
    def _get_device(self) -> str:
        try:
            if torch.cuda.is_available():
                if torch.cuda.get_device_properties(0).total_memory > 2 * 1024**3:  # 2gb
                    return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    
    @retry_with_backoff(max_retries=2, base_delay=2.0)
    def _load_model(self):
        """load the tts model"""
        with self._load_lock:
            if self._model_loaded:
                return
            
            try:
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning, module="jieba")
                
                self.tts = TTS(self.model_name, progress_bar=False).to(self.device)
                self._model_loaded = True
                logging.getLogger(__name__).info(f"TTS model loaded on {self.device}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to load TTS model: {e}")
                raise
    
    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def synthesize(self, text: str, output_path: str) -> bool:
        """synthesize text to speech"""
        if not text.strip():
            logging.getLogger(__name__).warning("empty text")
            return False
        
        cache_key = hash(text)
        with self._cache_lock:
            if cache_key in self._synthesis_cache:
                cached_path = self._synthesis_cache[cache_key]
                if Path(cached_path).exists():
                    shutil.copy2(cached_path, output_path)
                    return True
        
        if not self._model_loaded:
            self._load_model()
        
        if len(text) > 10000:
            logging.getLogger(__name__).warning(f"text too long ({len(text)} chars)")
            text = text[:10000] + "..."
        
        for attempt in range(self.config.max_retries):
            try:
                self.tts.tts_to_file(text=text, speaker=self.speaker, file_path=output_path)
                
                if Path(output_path).exists() and os.path.getsize(output_path) > 1024:
                    with self._cache_lock:
                        if len(self._synthesis_cache) >= self._cache_size:
                            for key in list(self._synthesis_cache.keys())[:20]:
                                del self._synthesis_cache[key]
                        self._synthesis_cache[cache_key] = output_path
                    return True
                else:
                    logging.getLogger(__name__).warning(f"invalid file: {output_path}")
                    
            except Exception as e:
                logging.getLogger(__name__).warning(f"tts error: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
        
        return False
    
    def __del__(self):
        """cleanup on destruction"""
        if self.tts:
            del self.tts
            self.tts = None
        MemoryManager.force_cleanup()


class PDFProcessor:    
    def __init__(self, text_processor: TextProcessor, config: Config):
        self.text_processor = text_processor
        self.config = config
        self._page_cache = {}
        self._cache_size = 500
        self._cache_lock = threading.Lock()
    
    def extract_chunks(self, pdf_path: Path) -> Iterator[TextChunk]:
        """extract and process text chunks from pdf"""
        try:
            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            logging.getLogger(__name__).info(f"Processing {total_pages} pages")
            
            processed_pages = 0
            total_text_length = 0
            stats = ProcessingStats()
            
            for start in range(0, total_pages, self.config.chunk_size):
                end = min(start + self.config.chunk_size, total_pages)
                pages = reader.pages[start:end]
                chunk_id = f"{start+1}-{end}"
                
                text_parts = []
                for i, page in enumerate(pages):
                    try:
                        page_text = self._extract_page_text(page, start + i + 1)
                        if page_text and page_text.strip():
                            text_parts.append(page_text.strip())
                            processed_pages += 1
                        else:
                            page_num = start + i + 1
                            if page_num > 1:  # skip warning for cover pages
                                logging.getLogger(__name__).debug(
                                    f"Page {page_num} has no extractable text (likely image/diagram)"
                                )
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Failed to extract text from page {start + i + 1}: {e}")
                        stats.add_error(e, f"page_extraction_{start + i + 1}")
                        continue
                
                text = "\n".join(text_parts).strip()
                if not text:
                    logging.getLogger(__name__).warning(f"Chunk {chunk_id} has no text")
                    continue
                
                total_text_length += len(text)
                logging.getLogger(__name__).info(
                    f"Processing chunk {chunk_id} ({len(text)} chars)"
                )
                
                start_time = time.time()
                try:
                    processed_text = self.text_processor.process_text(text)
                    if not processed_text:
                        logging.getLogger(__name__).warning(
                            f"Empty processing for chunk {chunk_id}, using original"
                        )
                        processed_text = text
                    
                    processing_time = time.time() - start_time
                    chunk = TextChunk(
                        chunk_id, 
                        f"Chunk {chunk_id}", 
                        processed_text,
                        page_range=(start+1, end),
                        processing_time=processing_time,
                        word_count=len(processed_text.split())
                    )
                    stats.add_chunk(chunk, processing_time, success=True)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    logging.getLogger(__name__).warning(f"Processing failed for chunk {chunk_id}: {e}")
                    stats.add_error(e, f"text_processing_{chunk_id}")
                    chunk = TextChunk(
                        chunk_id, 
                        f"Chunk {chunk_id}", 
                        text,
                        page_range=(start+1, end),
                        processing_time=processing_time,
                        word_count=len(text.split())
                    )
                    stats.add_chunk(chunk, processing_time, success=False)
                
                if len(processed_text) <= self.config.max_text_length:
                    yield chunk
                else:
                    yield from self._split_long_text(chunk_id, processed_text, (start+1, end))
                
                if processed_pages % self.config.memory_management_interval == 0:
                    MemoryManager.force_cleanup()
            
            logging.getLogger(__name__).info(
                f"Extraction complete: {processed_pages}/{total_pages} pages processed, "
                f"{total_text_length:,} total characters"
            )
            
            summary = stats.get_summary()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"PDF processing failed: {e}")
            raise
    
    def _extract_page_text(self, page, page_num: int) -> str:
        """extract text from page"""
        cache_key = f"page_{page_num}"
        
        with self._cache_lock:
            if cache_key in self._page_cache:
                return self._page_cache[cache_key]
        
        try:
            text = page.extract_text()
            
            with self._cache_lock:
                if len(self._page_cache) >= self._cache_size:
                    for key in list(self._page_cache.keys())[:100]:
                        del self._page_cache[key]
                self._page_cache[cache_key] = text
            
            return text
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to extract text from page {page_num}")
            return ""
    
    def _split_long_text(self, chunk_id: str, text: str, page_range: Tuple[int, int]) -> Iterator[TextChunk]:
        """split long text into chunks"""
        sentences = text.split('. ')
        current_chunk = ""
        count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if not sentence.endswith('.'):
                sentence += '.'
            
            if len(current_chunk) + len(sentence) + 1 > self.config.max_text_length and current_chunk:
                yield TextChunk(
                    f"{chunk_id}.{count}",
                    f"Chunk {chunk_id}.{count}",
                    current_chunk.strip(),
                    page_range=page_range,
                    word_count=len(current_chunk.split())
                )
                count += 1
                current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "
        
        if current_chunk.strip():
            yield TextChunk(
                f"{chunk_id}.{count}",
                f"Chunk {chunk_id}.{count}",
                current_chunk.strip(),
                page_range=page_range,
                word_count=len(current_chunk.split())
            )


class AudioProcessor:    
    def __init__(self, tts_processor: TTSProcessor, config: Config):
        self.tts = tts_processor
        self.config = config
        self._audio_cache = {}
        self._cache_size = 50
        self._cache_lock = threading.Lock()
    
    def synthesize_chunk(self, chunk: TextChunk, output_dir: Path, book_title: str = "", is_first: bool = False) -> Optional[str]:
        """synthesize chunk to audio"""
        if not chunk.content:
            return None
        
        cache_key = hash(chunk.content + book_title + str(is_first))
        with self._cache_lock:
            if cache_key in self._audio_cache:
                cached_path = self._audio_cache[cache_key]
                if Path(cached_path).exists():
                    output_path = output_dir / f"chunk_{chunk.id}.mp3"
                    shutil.copy2(cached_path, output_path)
                    return str(output_path)
        
        if self.config.add_book_announcements and is_first:
            narration_text = f"{book_title}. {chunk.title}. {chunk.content}"
        elif self.config.add_book_announcements:
            narration_text = f"{chunk.title}. {chunk.content}"
        else:
            narration_text = chunk.content
        
        output_path = output_dir / f"chunk_{chunk.id}.mp3"
        
        if self.tts.synthesize(narration_text, str(output_path)):
            with self._cache_lock:
                if len(self._audio_cache) >= self._cache_size:
                    for key in list(self._audio_cache.keys())[:10]:
                        del self._audio_cache[key]
                self._audio_cache[cache_key] = str(output_path)
            return str(output_path)
        
        return None
    
    def merge_files(self, audio_files: List[str], output_path: Path, book_title: str = ""):
        """merge audio files with intro/outro."""
        try:
            audio = AudioSegment.silent(duration=200)
            
            if self.config.add_intro_outro and book_title:
                intro_audio = self._create_intro(book_title)
                if intro_audio:
                    audio += intro_audio
                    audio += AudioSegment.silent(duration=1000)
            
            successful_files = 0
            failed_files = 0
            
            for i, file_path in enumerate(tqdm(audio_files, desc="Merging audio")):
                try:
                    if not Path(file_path).exists():
                        logging.getLogger(__name__).warning(f"Audio file not found: {file_path}")
                        failed_files += 1
                        continue
                    
                    segment = AudioSegment.from_file(file_path)
                    if len(segment) > 0:
                        audio += segment
                        successful_files += 1
                        if i < len(audio_files) - 1:
                            audio += AudioSegment.silent(duration=self.config.silence_padding)
                    else:
                        logging.getLogger(__name__).warning(f"Empty audio segment: {file_path}")
                        failed_files += 1
                        
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Skipping {file_path}: {e}")
                    failed_files += 1
            
            if self.config.add_intro_outro and book_title:
                audio += AudioSegment.silent(duration=1000)
                outro_audio = self._create_outro(book_title)
                if outro_audio:
                    audio += outro_audio
            
            audio += AudioSegment.silent(duration=500)
            
            try:
                audio.set_frame_rate(self.config.sample_rate).export(
                    output_path, format="mp3", bitrate=self.config.bitrate
                )
                logging.getLogger(__name__).info(
                    f"Merge complete: {successful_files} successful, {failed_files} failed"
                )
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to export merged audio: {e}")
                raise
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Audio merging failed: {e}")
            raise
    
    def _create_intro(self, book_title: str) -> Optional[AudioSegment]:
        """create intro audio with error handling."""
        try:
            intro_text = f"{book_title}."
            temp_path = Path(tempfile.mktemp(suffix=".mp3"))
            
            if self.tts.synthesize(intro_text, str(temp_path)):
                audio = AudioSegment.from_file(temp_path)
                temp_path.unlink()
                return audio
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not generate intro")
            return None
    
    def _create_outro(self, book_title: str) -> Optional[AudioSegment]:
        """create outro audio with error handling."""
        try:
            outro_text = f"End of {book_title}."
            temp_path = Path(tempfile.mktemp(suffix=".mp3"))
            
            if self.tts.synthesize(outro_text, str(temp_path)):
                audio = AudioSegment.from_file(temp_path)
                temp_path.unlink()
                return audio
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not generate outro")
            return None


class PDFToAudioConverter:
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.monitor = PerformanceMonitor()
        self.stats = ProcessingStats()
        
        # Initialize components with configuration
        self.model_manager = ModelManager(self.config.model_cache_dir, self.config)
        self.text_processor = TextProcessor(self.model_manager, self.config)
        self.tts_processor = TTSProcessor(self.config.tts_model, self.config.speaker, self.config)
        self.pdf_processor = PDFProcessor(self.text_processor, self.config)
        self.audio_processor = AudioProcessor(self.tts_processor, self.config)
    
    def convert(self, pdf_path: str) -> str:
        """convert pdf to audio"""
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not pdf_file.suffix.lower() == '.pdf':
            raise ValueError(f"File must be a PDF: {pdf_path}")
        
        file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:
            self.config.memory_management_interval = 20
            self.config.progress_log_interval = 3
            if file_size_mb > 100:
                self.config.chunk_size = 10
        
        audio_dir = Path("audio")
        text_dir = Path("cleaned_texts_llm")
        audio_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)
        
        output_file = audio_dir / f"{pdf_file.stem}_audiobook.mp3"
        
        if output_file.exists():
            logger.warning(f"Output exists: {output_file}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                if self.config.enable_llm_enhancement:
                    try:
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError("Model loading timed out")
                        
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)
                        
                        try:
                            if not self.model_manager.load():
                                self.config.enable_llm_enhancement = False
                        finally:
                            signal.alarm(0)
                            
                    except (TimeoutError, Exception):
                        self.config.enable_llm_enhancement = False
                
                chunks = []
                chunk_count = 0
                
                for chunk in self.pdf_processor.extract_chunks(pdf_file):
                    chunks.append(chunk)
                    chunk_count += 1
                    
                    if chunk_count % self.config.memory_management_interval == 0:
                        MemoryManager.force_cleanup()
                
                if not chunks:
                    raise RuntimeError("No text content could be extracted from PDF")
                
                self._save_text_files(chunks, text_dir)
                
                audio_files = self._generate_audio(chunks, temp_path, pdf_file.stem)
                if not audio_files:
                    raise RuntimeError("No audio was generated")
                
                audio_files.sort(key=self._sort_chunk_key)
                self.audio_processor.merge_files(audio_files, output_file, pdf_file.stem)
                
                self._save_metadata(pdf_file, chunks, text_dir, output_file)
                
                if not output_file.exists() or output_file.stat().st_size < 1024:
                    raise RuntimeError("Generated audio file is invalid or too small")
                
                return str(output_file)
                
            except Exception as e:
                if output_file.exists():
                    output_file.unlink()
                raise
    
    def _save_text_files(self, chunks: List[TextChunk], text_dir: Path):
        for chunk in chunks:
            safe_title = re.sub(r'[^\w\s-]', '', chunk.title).strip()
            safe_title = re.sub(r'\s+', '_', safe_title)
            filename = f"{chunk.id}_{safe_title}.txt"
            filepath = text_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {chunk.title}\n")
                f.write(f"Chapter ID: {chunk.id}\n")
                f.write(f"Processing: LLM-Enhanced\n")
                f.write("-" * 50 + "\n\n")
                f.write(chunk.content)
    
    def _generate_audio(self, chunks: List[TextChunk], temp_dir: Path, book_title: str) -> List[str]:
        """generate audio files in parallel"""
        results = []
        book_title_clean = book_title.replace('-', ' ').replace('_', ' ').title()
        
        max_workers = min(self.config.workers, max(1, len(chunks) // 10))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(
                self.audio_processor.synthesize_chunk,
                chunk, temp_dir, book_title_clean, i == 0
            ): chunk.id for i, chunk in enumerate(chunks)}
            
            completed = failed = 0
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="synthesizing"):
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    if result:
                        results.append(result)
                        completed += 1
                    else:
                        failed += 1
                        logger.warning(f"empty synthesis for chunk {futures[future]}")
                except TimeoutError:
                    failed += 1
                    logger.warning(f"timeout for chunk {futures[future]}")
                except Exception as e:
                    failed += 1
                    logger.warning(f"failed for chunk {futures[future]}: {e}")
                
        return results
    
    def _save_metadata(self, pdf_file: Path, chunks: List[TextChunk], text_dir: Path, output_file: Path):
        metadata = {
            "pdf_file": str(pdf_file),
            "chapters": len(chunks),
            "audio_file": str(output_file),
            "tts_model": self.config.tts_model,
            "llm_model": self.config.llm_model,
            "llm_enhanced": self.model_manager.model is not None,
            "config": {
                "chunk_size": self.config.chunk_size,
                "max_text_length": self.config.max_text_length,
                "workers": self.config.workers
            }
        }
        
        metadata_file = text_dir / f"{pdf_file.stem}_metadata_llm.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def _sort_chunk_key(file_path: str) -> Tuple[int, int]:
        """sort chunks by page numbers"""
        stem = Path(file_path).stem.replace("chunk_", "")
        
        if "-" in stem:
            page_part, subpart = (stem.split(".", 1) if "." in stem else (stem, "0"))
            start_page = int(page_part.split("-")[0])
            return (start_page, int(subpart))
        
        return (0, 0)


def signal_handler(signum, frame):
    """handle interrupt signals"""
    print("\ninterrupted")
    sys.exit(1)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    monitor = PerformanceMonitor()
    
    if len(sys.argv) < 2:
        print("Usage: python pdf2audio_llm.py <pdf_file>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    config = Config()
    if "--no-llm" in sys.argv:
        config.enable_llm_enhancement = False
    
    if not pdf_file.lower().endswith('.pdf'):
        print("error: input must be pdf file")
        sys.exit(1)
    
    if not Path(pdf_file).exists():
        print(f"error: file '{pdf_file}' not found")
        sys.exit(1)
    
    file_size = Path(pdf_file).stat().st_size
    if file_size == 0:
        print("error: pdf file is empty")
        sys.exit(1)
    
    print(f"Processing: {pdf_file} ({file_size / 1024 / 1024:.1f} MB)")
    
    start_time = time.time()
    
    logger = setup_logging()
    
    try:
        with resource_manager(monitor, logger):
            converter = PDFToAudioConverter(config)
            output_file = converter.convert(pdf_file)
            
            elapsed_time = time.time() - start_time
            print(f"completed in {elapsed_time:.1f}s: {output_file}")
        
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 