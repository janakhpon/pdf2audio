#!/usr/bin/env python3
"""
PDF to Audio Converter with LLM-Enhanced Text Processing
A production-ready implementation for converting PDFs to audiobooks.

Author: Senior Software Engineer
Features:
- Robust error handling and recovery
- Memory-efficient processing for large documents
- Comprehensive logging and monitoring
- Resource management and cleanup
- Performance optimizations
- Graceful degradation and fallbacks
"""

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
from contextlib import contextmanager, ExitStack
from functools import lru_cache, wraps

import torch
from PyPDF2 import PdfReader
from TTS.api import TTS
from pydub import AudioSegment
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Performance monitoring
class PerformanceMonitor:
    """Monitors system performance and resource usage."""
    
    def __init__(self):
        self.start_time = time.time()
        self.memory_usage = []
        self.cpu_usage = []
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start background monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                process = psutil.Process()
                self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
                self.cpu_usage.append(process.cpu_percent())
                time.sleep(5)  # Sample every 5 seconds
            except Exception:
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        try:
            process = psutil.Process()
            return {
                "elapsed_time": time.time() - self.start_time,
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "memory_peak_mb": max(self.memory_usage) if self.memory_usage else 0,
                "cpu_peak_percent": max(self.cpu_usage) if self.cpu_usage else 0
            }
        except Exception:
            return {"error": "Could not retrieve performance stats"}

# Enhanced configuration with validation
@dataclass
class Config:
    """Application configuration with validation and sensible defaults."""
    # PDF processing
    chunk_size: int = 5
    max_text_length: int = 6000
    chapter_min_length: int = 200
    
    # Audio settings
    bitrate: str = "128k"
    sample_rate: int = 24000
    silence_padding: int = 200
    
    # TTS configuration
    tts_model: str = "tts_models/en/vctk/vits"
    speaker: str = "p230"
    
    # LLM settings
    llm_model: str = "gpt2-medium"
    max_new_tokens: int = 300
    temperature: float = 0.7
    
    # Performance settings
    workers: int = field(default_factory=lambda: min(os.cpu_count() or 2, 6))
    add_book_announcements: bool = True
    add_intro_outro: bool = True
    
    # Large document handling
    enable_llm_enhancement: bool = True
    memory_management_interval: int = 30
    progress_log_interval: int = 5
    
    # Resource management
    model_cache_dir: str = "./models"
    max_memory_mb: int = 8192
    timeout_seconds: int = 600
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Validate configuration values."""
        # Validate chunk_size
        if self.chunk_size < 1 or self.chunk_size > 20:
            raise ValueError(f"chunk_size must be between 1 and 20, got {self.chunk_size}")
        
        # Validate max_text_length
        if self.max_text_length < 1000 or self.max_text_length > 15000:
            raise ValueError(f"max_text_length must be between 1000 and 15000, got {self.max_text_length}")
        
        # Validate sample_rate
        if self.sample_rate < 8000 or self.sample_rate > 48000:
            raise ValueError(f"sample_rate must be between 8000 and 48000, got {self.sample_rate}")
        
        # Validate max_new_tokens
        if self.max_new_tokens < 50 or self.max_new_tokens > 1000:
            raise ValueError(f"max_new_tokens must be between 50 and 1000, got {self.max_new_tokens}")
        
        # Validate temperature
        if self.temperature < 0.1 or self.temperature > 1.0:
            raise ValueError(f"temperature must be between 0.1 and 1.0, got {self.temperature}")
        
        # Validate memory_management_interval
        if self.memory_management_interval < 10 or self.memory_management_interval > 100:
            raise ValueError(f"memory_management_interval must be between 10 and 100, got {self.memory_management_interval}")
        
        # Validate progress_log_interval
        if self.progress_log_interval < 1 or self.progress_log_interval > 20:
            raise ValueError(f"progress_log_interval must be between 1 and 20, got {self.progress_log_interval}")
        
        # Validate max_memory_mb
        if self.max_memory_mb < 1024 or self.max_memory_mb > 32768:
            raise ValueError(f"max_memory_mb must be between 1024 and 32768, got {self.max_memory_mb}")
        
        # Validate timeout_seconds
        if self.timeout_seconds < 60 or self.timeout_seconds > 3600:
            raise ValueError(f"timeout_seconds must be between 60 and 3600, got {self.timeout_seconds}")
        
        # Validate max_retries
        if self.max_retries < 1 or self.max_retries > 10:
            raise ValueError(f"max_retries must be between 1 and 10, got {self.max_retries}")
        
        # Validate retry_delay
        if self.retry_delay < 0.1 or self.retry_delay > 10.0:
            raise ValueError(f"retry_delay must be between 0.1 and 10.0, got {self.retry_delay}")

# Enhanced logging setup
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging with file and console handlers."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with color
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler for detailed logging
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Resource management context manager
@contextmanager
def resource_manager(monitor: PerformanceMonitor, logger: logging.Logger):
    """Context manager for resource monitoring and cleanup."""
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

# Retry decorator with exponential backoff
def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying operations with exponential backoff."""
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

# Memory management utilities
class MemoryManager:
    """Manages memory usage and garbage collection."""
    
    @staticmethod
    def check_memory_usage(max_memory_mb: int) -> bool:
        """Check if memory usage is within limits."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb < max_memory_mb
        except Exception:
            return True  # Assume OK if we can't check
    
    @staticmethod
    def force_cleanup():
        """Force garbage collection and memory cleanup."""
        gc.collect()
        if hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get detailed memory information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except Exception:
            return {"error": "Could not retrieve memory info"}

# Enhanced data structures
class TextChunk(NamedTuple):
    """Represents a processed text chunk with metadata."""
    id: str
    title: str
    content: str
    page_range: Optional[Tuple[int, int]] = None
    processing_time: Optional[float] = None
    word_count: Optional[int] = None
    
    def __post_init__(self):
        """Calculate word count if not provided."""
        if self.word_count is None:
            self.word_count = len(self.content.split()) if self.content else 0

class ProcessingStats:
    """Tracks processing statistics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.chunks_processed = 0
        self.chunks_failed = 0
        self.total_words = 0
        self.total_chars = 0
        self.processing_times = []
        self.errors = []
    
    def add_chunk(self, chunk: TextChunk, processing_time: float, success: bool = True):
        """Add chunk processing result."""
        if success:
            self.chunks_processed += 1
            self.total_words += chunk.word_count or 0
            self.total_chars += len(chunk.content)
        else:
            self.chunks_failed += 1
        
        self.processing_times.append(processing_time)
    
    def add_error(self, error: Exception, context: str = ""):
        """Add error information."""
        self.errors.append({
            "error": str(error),
            "type": type(error).__name__,
            "context": context,
            "timestamp": time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
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

# Global configuration instance
config = Config()

# Setup logging
logger = setup_logging("INFO", "pdf2audio_llm.log")

# Enhanced Model Manager with better resource management
class ModelManager:
    """Manages LLM model loading and context with enhanced error handling."""
    
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
        """Get the best available device with fallback."""
        try:
            if torch.cuda.is_available():
                # Check CUDA memory
                if torch.cuda.get_device_properties(0).total_memory > 4 * 1024**3:  # 4GB
                    return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    
    @retry_with_backoff(max_retries=2, base_delay=2.0)
    def load(self) -> bool:
        """Load the model and tokenizer with enhanced error handling."""
        with self._load_lock:
            if self._model_loaded:
                return True
            
            try:
                logging.getLogger(__name__).info(f"Loading LLM model: {self.model_name}")
                
                # Check available memory
                if not MemoryManager.check_memory_usage(self.config.max_memory_mb):
                    raise RuntimeError(f"Insufficient memory. Required: {self.config.max_memory_mb}MB")
                
                # Load tokenizer with timeout
                logging.getLogger(__name__).info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    cache_dir=str(self.cache_dir),
                    local_files_only=False
                )
                
                # Handle pad token properly
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                # Load model with memory optimization
                logging.getLogger(__name__).info("Loading model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    cache_dir=str(self.cache_dir),
                    local_files_only=False,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
                
                # Calculate context length
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
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        MemoryManager.force_cleanup()
        self._model_loaded = False
    
    def _get_context_length(self) -> int:
        """Get the model's context length."""
        if not self.model:
            return 1024
        
        config = self.model.config
        for attr in ['max_position_embeddings', 'n_positions', 'max_sequence_length']:
            if hasattr(config, attr):
                return getattr(config, attr)
        return 1024
    
    def get_max_input_length(self, max_new_tokens: int = 300) -> int:
        """Calculate maximum input length for generation."""
        max_length = self.context_length - max_new_tokens - 100
        return max(512, max_length)
    
    def __del__(self):
        """Cleanup on destruction."""
        self._cleanup()


class TextProcessor:
    """Efficient text processing with LLM enhancement and robust error handling."""
    
    def __init__(self, model_manager: ModelManager, config: Config):
        self.model_manager = model_manager
        self.config = config
        self._cleaning_patterns = self._compile_cleaning_patterns()
        self._cache = {}
        self._cache_size = 1000
        self._cache_lock = threading.Lock()
    
    def _compile_cleaning_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Compile regex patterns for efficient cleaning."""
        patterns = [
            (re.compile(r'\b\d+\s*$', re.MULTILINE), ''),  # Page numbers at end
            (re.compile(r'^\d+\s*', re.MULTILINE), ''),     # Page numbers at start
            (re.compile(r'[A-Z]{2,}\s*\d+'), ''),          # Headers with numbers
            (re.compile(r'©\s*\d{4}.*?\.'), ''),           # Copyright notices
            (re.compile(r'www\.[^\s]+'), ''),               # URLs
            (re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)]'), ''), # Special chars
            (re.compile(r'\s+'), ' '),                      # Multiple spaces
            (re.compile(r'\n\s*\n'), '\n\n'),              # Multiple newlines
        ]
        return patterns
    
    @lru_cache(maxsize=1000)
    def _rule_based_cleaning(self, text: str) -> str:
        """Apply rule-based text cleaning with caching."""
        if not text or not text.strip():
            return ""
        
        cleaned = text
        for pattern, replacement in self._cleaning_patterns:
            cleaned = pattern.sub(replacement, cleaned)
        
        # Final cleanup
        cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)
        return cleaned.strip()
    
    def process_text(self, text: str) -> str:
        """Process text with LLM enhancement if available."""
        # Validate input
        if not text or not text.strip():
            return ""
        
        # Check cache first
        cache_key = hash(text)
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Always start with rule-based cleaning for consistency
        cleaned_text = self._rule_based_cleaning(text)
        
        if not self.model_manager.model or not self.config.enable_llm_enhancement:
            result = cleaned_text
        else:
            try:
                max_input_length = self.model_manager.get_max_input_length()
                estimated_chars_per_token = 4
                max_chars = max_input_length * estimated_chars_per_token
                
                logging.getLogger(__name__).debug(
                    f"Processing text of {len(cleaned_text)} characters "
                    f"with max chunk size of {max_chars} characters"
                )
                
                if len(cleaned_text) <= max_chars:
                    result = self._process_single_chunk(cleaned_text)
                else:
                    result = self._process_chunked_text(cleaned_text, max_chars)
                    
            except Exception as e:
                logging.getLogger(__name__).warning(f"LLM processing failed: {e}, using rule-based cleaning")
                result = cleaned_text
        
        # Cache result
        with self._cache_lock:
            if len(self._cache) >= self._cache_size:
                # Remove oldest entries
                oldest_keys = list(self._cache.keys())[:100]
                for key in oldest_keys:
                    del self._cache[key]
            self._cache[cache_key] = result
        
        return result
    
    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def _process_single_chunk(self, text: str) -> str:
        """Process a single chunk of text with retry logic."""
        try:
            # First, clean the text
            cleaned = self._llm_clean(text)
            if not cleaned or len(cleaned.strip()) < 10:
                logging.getLogger(__name__).debug("LLM cleaning returned empty or too short, using rule-based")
                cleaned = self._rule_based_cleaning(text)
            
            # Then enhance if we have a good base
            if len(cleaned) > 20:
                enhanced = self._llm_enhance(cleaned)
                if enhanced and len(enhanced.strip()) > 10:
                    return enhanced
            
            return cleaned
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Single chunk processing failed: {e}")
            return self._rule_based_cleaning(text)
    
    def _process_chunked_text(self, text: str, max_chars: int) -> str:
        """Process text in overlapping chunks."""
        sentences = text.split('. ')
        chunks = self._create_overlapping_chunks(sentences, max_chars)
        
        if not chunks:
            return self._rule_based_cleaning(text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            logging.getLogger(__name__).debug(f"Processing chunk {i+1}/{len(chunks)}")
            processed = self._process_single_chunk(chunk)
            if processed:
                processed_chunks.append(processed)
        
        return self._merge_chunks(processed_chunks) if processed_chunks else self._rule_based_cleaning(text)
    
    def _create_overlapping_chunks(self, sentences: List[str], max_chars: int) -> List[str]:
        """Create overlapping chunks from sentences."""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) + 2 > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                # Create overlap with last 2 sentences for context continuity
                overlap_sentences = current_chunk.split('. ')[-2:]
                current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
            else:
                current_chunk += sentence + '. '
        
        # Add the final chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _merge_chunks(self, chunks: List[str]) -> str:
        """Merge processed chunks, removing overlaps."""
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
        """Find overlapping content between two text chunks."""
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
        """Clean text using LLM with retry logic."""
        if not text or len(text) < 10:
            return text
            
        prompt = f"Clean this text for audiobook narration. Remove formatting artifacts, fix punctuation, ensure smooth reading flow, and maintain the original meaning:\n\n{text[:1800]}"
        return self._generate_text(prompt, max_new_tokens=200, temperature=0.4)
    
    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def _llm_enhance(self, text: str) -> str:
        """Enhance text using LLM with retry logic."""
        if not text or len(text) < 10:
            return text
            
        prompt = f"Enhance this text for audiobook narration. Make it engaging, clear, well-paced for listening, and maintain the technical accuracy:\n\n{text[:1800]}"
        return self._generate_text(prompt, max_new_tokens=250, temperature=0.7)
    
    def _generate_text(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate text using the LLM model with optimized performance."""
        try:
            max_input_length = self.model_manager.get_max_input_length(max_new_tokens)
            
            # Encode and check length with proper attention mask
            inputs = self.model_manager.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=max_input_length,
                add_special_tokens=True
            )
            
            # Create proper attention mask
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
                    logging.getLogger(__name__).warning(f"Generation failed: {e}")
                    return ""
            
            generated = self.model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated content
            if ":" in generated:
                parts = generated.split(":", 1)
                if len(parts) > 1:
                    generated = parts[-1].strip()
            
            # Clean up the generated text
            generated = generated.replace('\n', ' ').replace('  ', ' ')
            
            # Remove duplicates while preserving order
            sentences = [s.strip() for s in generated.split('. ') if s.strip()]
            unique_sentences = list(dict.fromkeys(sentences))
            
            result = '. '.join(unique_sentences) if unique_sentences else ""
            
            # Validate result quality
            if len(result) < 10:
                logging.getLogger(__name__).debug("Generated text too short, using original")
                return ""
            
            return result
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Text generation failed: {e}")
            return ""
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """intelligently truncate prompt to fit token limit."""
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
    """Efficient TTS processing with enhanced error handling and resource management."""
    
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
        """Get the best available device with fallback."""
        try:
            if torch.cuda.is_available():
                # Check CUDA memory
                if torch.cuda.get_device_properties(0).total_memory > 2 * 1024**3:  # 2GB
                    return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    
    @retry_with_backoff(max_retries=2, base_delay=2.0)
    def _load_model(self):
        """Load the TTS model with retry logic."""
        with self._load_lock:
            if self._model_loaded:
                return
            
            try:
                # Suppress deprecation warnings from jieba
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
        """Synthesize text to speech with enhanced error handling."""
        if not text.strip():
            logging.getLogger(__name__).warning("Empty text provided for synthesis")
            return False
        
        # Check cache first
        cache_key = hash(text)
        with self._cache_lock:
            if cache_key in self._synthesis_cache:
                cached_path = self._synthesis_cache[cache_key]
                if Path(cached_path).exists():
                    shutil.copy2(cached_path, output_path)
                    return True
        
        # Ensure model is loaded
        if not self._model_loaded:
            self._load_model()
        
        # Limit text length to prevent TTS crashes
        max_text_length = 10000  # Conservative limit
        if len(text) > max_text_length:
            logging.getLogger(__name__).warning(
                f"Text too long ({len(text)} chars), truncating to {max_text_length}"
            )
            text = text[:max_text_length] + "..."
        
        for attempt in range(self.config.max_retries):
            try:
                self.tts.tts_to_file(text=text, speaker=self.speaker, file_path=output_path)
                
                if Path(output_path).exists() and os.path.getsize(output_path) > 1024:
                    # Cache successful synthesis
                    with self._cache_lock:
                        if len(self._synthesis_cache) >= self._cache_size:
                            # Remove oldest entries
                            oldest_keys = list(self._synthesis_cache.keys())[:20]
                            for key in oldest_keys:
                                del self._synthesis_cache[key]
                        self._synthesis_cache[cache_key] = output_path
                    return True
                else:
                    logging.getLogger(__name__).warning(f"Generated file is invalid: {output_path}")
                    
            except Exception as e:
                logging.getLogger(__name__).warning(f"TTS error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
        
        return False
    
    def __del__(self):
        """Cleanup on destruction."""
        if self.tts:
            del self.tts
            self.tts = None
        MemoryManager.force_cleanup()


class PDFProcessor:
    """Efficient PDF text extraction and processing with enhanced error handling."""
    
    def __init__(self, text_processor: TextProcessor, config: Config):
        self.text_processor = text_processor
        self.config = config
        self._page_cache = {}
        self._cache_size = 500
        self._cache_lock = threading.Lock()
    
    def extract_chunks(self, pdf_path: Path) -> Iterator[TextChunk]:
        """Extract and process text chunks from PDF with memory optimization."""
        try:
            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            logging.getLogger(__name__).info(f"Processing {total_pages} pages")
            
            # Track processed pages to ensure nothing is missed
            processed_pages = 0
            total_text_length = 0
            stats = ProcessingStats()
            
            for start in range(0, total_pages, self.config.chunk_size):
                # Process pages in chunks to manage memory
                end = min(start + self.config.chunk_size, total_pages)
                pages = reader.pages[start:end]
                chunk_id = f"{start+1}-{end}"
                
                # Extract text efficiently
                text_parts = []
                for i, page in enumerate(pages):
                    try:
                        page_text = self._extract_page_text(page, start + i + 1)
                        if page_text and page_text.strip():
                            text_parts.append(page_text.strip())
                            processed_pages += 1
                        else:
                            # Only warn for pages that might be expected to have text
                            page_num = start + i + 1
                            if page_num > 1:  # Skip warning for cover pages
                                logging.getLogger(__name__).debug(
                                    f"Page {page_num} has no extractable text (likely image/diagram)"
                                )
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Failed to extract text from page {start + i + 1}: {e}")
                        stats.add_error(e, f"page_extraction_{start + i + 1}")
                        continue
                
                text = "\n".join(text_parts).strip()
                if not text:
                    logging.getLogger(__name__).warning(f"Chunk {chunk_id} has no text content")
                    continue
                
                total_text_length += len(text)
                logging.getLogger(__name__).info(
                    f"Processing chunk {chunk_id} ({len(text)} chars, pages {start+1}-{end})"
                )
                
                # Process text with timing
                start_time = time.time()
                try:
                    processed_text = self.text_processor.process_text(text)
                    if not processed_text:
                        logging.getLogger(__name__).warning(
                            f"Text processing returned empty for chunk {chunk_id}, using original"
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
                    logging.getLogger(__name__).warning(f"Text processing failed for chunk {chunk_id}: {e}")
                    stats.add_error(e, f"text_processing_{chunk_id}")
                    # Use original text as fallback
                    chunk = TextChunk(
                        chunk_id, 
                        f"Chunk {chunk_id}", 
                        text,
                        page_range=(start+1, end),
                        processing_time=processing_time,
                        word_count=len(text.split())
                    )
                    stats.add_chunk(chunk, processing_time, success=False)
                
                # Split if too long for TTS
                if len(processed_text) <= self.config.max_text_length:
                    yield chunk
                else:
                    # Split into smaller pieces
                    yield from self._split_long_text(chunk_id, processed_text, (start+1, end))
                
                # Memory management for large documents
                if processed_pages % self.config.memory_management_interval == 0:
                    MemoryManager.force_cleanup()
            
            logging.getLogger(__name__).info(
                f"Extraction complete: {processed_pages}/{total_pages} pages processed, "
                f"{total_text_length:,} total characters"
            )
            
            if processed_pages < total_pages:
                skipped_pages = total_pages - processed_pages
                logging.getLogger(__name__).info(
                    f"Skipped {skipped_pages} pages (likely images, diagrams, or blank pages)"
                )
            
            # Log final statistics
            summary = stats.get_summary()
            logging.getLogger(__name__).info(f"Processing summary: {summary}")
            
        except Exception as e:
            logging.getLogger(__name__).error(f"PDF processing failed: {e}")
            raise
    
    def _extract_page_text(self, page, page_num: int) -> str:
        """Extract text from a single page with caching."""
        cache_key = f"page_{page_num}"
        
        with self._cache_lock:
            if cache_key in self._page_cache:
                return self._page_cache[cache_key]
        
        try:
            text = page.extract_text()
            
            # Cache the result
            with self._cache_lock:
                if len(self._page_cache) >= self._cache_size:
                    # Remove oldest entries
                    oldest_keys = list(self._page_cache.keys())[:100]
                    for key in oldest_keys:
                        del self._page_cache[key]
                self._page_cache[cache_key] = text
            
            return text
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to extract text from page {page_num}: {e}")
            return ""
    
    def _split_long_text(self, chunk_id: str, text: str, page_range: Tuple[int, int]) -> Iterator[TextChunk]:
        """Split long text into smaller chunks with sentence boundary awareness."""
        sentences = text.split('. ')
        current_chunk = ""
        count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add period back if it was removed by split
            if not sentence.endswith('.'):
                sentence += '.'
            
            # Check if adding this sentence would exceed the limit
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
        
        # Add the final chunk if it has content
        if current_chunk.strip():
            yield TextChunk(
                f"{chunk_id}.{count}",
                f"Chunk {chunk_id}.{count}",
                current_chunk.strip(),
                page_range=page_range,
                word_count=len(current_chunk.split())
            )


class AudioProcessor:
    """Efficient audio processing and merging with enhanced error handling."""
    
    def __init__(self, tts_processor: TTSProcessor, config: Config):
        self.tts = tts_processor
        self.config = config
        self._audio_cache = {}
        self._cache_size = 50
        self._cache_lock = threading.Lock()
    
    def synthesize_chunk(self, chunk: TextChunk, output_dir: Path, book_title: str = "", is_first: bool = False) -> Optional[str]:
        """Synthesize a text chunk to audio with caching."""
        if not chunk.content:
            return None
        
        # Check cache first
        cache_key = hash(chunk.content + book_title + str(is_first))
        with self._cache_lock:
            if cache_key in self._audio_cache:
                cached_path = self._audio_cache[cache_key]
                if Path(cached_path).exists():
                    output_path = output_dir / f"chunk_{chunk.id}.mp3"
                    shutil.copy2(cached_path, output_path)
                    return str(output_path)
        
        # Prepare narration text
        if self.config.add_book_announcements and is_first:
            narration_text = f"{book_title}. {chunk.title}. {chunk.content}"
        elif self.config.add_book_announcements:
            narration_text = f"{chunk.title}. {chunk.content}"
        else:
            narration_text = chunk.content
        
        output_path = output_dir / f"chunk_{chunk.id}.mp3"
        
        if self.tts.synthesize(narration_text, str(output_path)):
            # Cache successful synthesis
            with self._cache_lock:
                if len(self._audio_cache) >= self._cache_size:
                    # Remove oldest entries
                    oldest_keys = list(self._audio_cache.keys())[:10]
                    for key in oldest_keys:
                        del self._audio_cache[key]
                self._audio_cache[cache_key] = str(output_path)
            return str(output_path)
        
        return None
    
    def merge_files(self, audio_files: List[str], output_path: Path, book_title: str = ""):
        """Merge audio files with intro/outro and enhanced error handling."""
        try:
            audio = AudioSegment.silent(duration=200)
            
            # Add intro
            if self.config.add_intro_outro and book_title:
                intro_audio = self._create_intro(book_title)
                if intro_audio:
                    audio += intro_audio
                    audio += AudioSegment.silent(duration=1000)
            
            # Add main content with progress tracking
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
            
            # Add outro
            if self.config.add_intro_outro and book_title:
                audio += AudioSegment.silent(duration=1000)
                outro_audio = self._create_outro(book_title)
                if outro_audio:
                    audio += outro_audio
            
            audio += AudioSegment.silent(duration=500)
            
            # Export with error handling
            try:
                audio.set_frame_rate(self.config.sample_rate).export(
                    output_path, format="mp3", bitrate=self.config.bitrate
                )
                logging.getLogger(__name__).info(
                    f"Audio merge complete: {successful_files} successful, {failed_files} failed"
                )
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to export merged audio: {e}")
                raise
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Audio merging failed: {e}")
            raise
    
    def _create_intro(self, book_title: str) -> Optional[AudioSegment]:
        """Create intro audio with error handling."""
        try:
            intro_text = f"{book_title}."
            temp_path = Path(tempfile.mktemp(suffix=".mp3"))
            
            if self.tts.synthesize(intro_text, str(temp_path)):
                audio = AudioSegment.from_file(temp_path)
                temp_path.unlink()
                return audio
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not generate intro: {e}")
        return None
    
    def _create_outro(self, book_title: str) -> Optional[AudioSegment]:
        """Create outro audio with error handling."""
        try:
            outro_text = f"End of {book_title}."
            temp_path = Path(tempfile.mktemp(suffix=".mp3"))
            
            if self.tts.synthesize(outro_text, str(temp_path)):
                audio = AudioSegment.from_file(temp_path)
                temp_path.unlink()
                return audio
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not generate outro: {e}")
        return None


class PDFToAudioConverter:
    """main converter class with clean architecture."""
    
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
        """convert pdf to audio with comprehensive error handling and validation."""
        pdf_file = Path(pdf_path)
        
        # Validate input
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not pdf_file.suffix.lower() == '.pdf':
            raise ValueError(f"File must be a PDF: {pdf_path}")
        
        # Check file size and optimize for large documents
        file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:
            logger.info(f"Large PDF detected ({file_size_mb:.1f} MB). Optimizing for large document processing.")
            # Optimize settings for large documents
            self.config.memory_management_interval = 20  # More frequent GC
            self.config.progress_log_interval = 3        # More frequent updates
            if file_size_mb > 100:
                logger.warning(f"Very large PDF ({file_size_mb:.1f} MB). This may take significant time.")
                self.config.chunk_size = 10              # Larger chunks for efficiency
        
        # Setup directories
        audio_dir = Path("audio")
        text_dir = Path("cleaned_texts_llm")
        audio_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)
        
        output_file = audio_dir / f"{pdf_file.stem}_audiobook.mp3"
        
        # Check if output already exists
        if output_file.exists():
            logger.warning(f"Output file already exists: {output_file}")
            # Could add option to overwrite or skip
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # Load models with timeout (only if LLM enhancement is enabled)
                if self.config.enable_llm_enhancement:
                    logger.info("Loading AI models...")
                    try:
                        # Add timeout for model loading
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError("Model loading timed out")
                        
                        # Set 60 second timeout for model loading
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)
                        
                        try:
                            if not self.model_manager.load():
                                logger.warning("LLM model failed to load, using rule-based processing")
                                self.config.enable_llm_enhancement = False
                        finally:
                            signal.alarm(0)  # Cancel the alarm
                            
                    except TimeoutError:
                        logger.warning("LLM model loading timed out, using rule-based processing")
                        self.config.enable_llm_enhancement = False
                    except Exception as e:
                        logger.warning(f"LLM model loading failed: {e}, using rule-based processing")
                        self.config.enable_llm_enhancement = False
                else:
                    logger.info("Skipping LLM model loading (disabled)")
                
                # Extract and process chunks with memory management
                logger.info("Extracting and processing text...")
                chunks = []
                chunk_count = 0
                
                for chunk in self.pdf_processor.extract_chunks(pdf_file):
                    chunks.append(chunk)
                    chunk_count += 1
                    
                    # Log progress for large documents
                    if chunk_count % self.config.progress_log_interval == 0:
                        logger.info(f"Processed {chunk_count} chunks so far...")
                    
                    # Force garbage collection for large documents
                    if chunk_count % self.config.memory_management_interval == 0:
                        MemoryManager.force_cleanup()
                
                if not chunks:
                    raise RuntimeError("No text content could be extracted from PDF")
                
                logger.info(f"Processed {len(chunks)} text chunks")
                
                # Save processed text
                self._save_text_files(chunks, text_dir)
                
                # Generate audio
                logger.info("Generating audio...")
                audio_files = self._generate_audio(chunks, temp_path, pdf_file.stem)
                if not audio_files:
                    raise RuntimeError("No audio was generated")
                
                logger.info(f"Generated {len(audio_files)} audio files")
                
                # Merge audio files
                logger.info("Merging audio files...")
                audio_files.sort(key=self._sort_chunk_key)
                self.audio_processor.merge_files(audio_files, output_file, pdf_file.stem)
                
                # Save metadata
                self._save_metadata(pdf_file, chunks, text_dir, output_file)
                
                # Validate output
                if not output_file.exists() or output_file.stat().st_size < 1024:
                    raise RuntimeError("Generated audio file is invalid or too small")
                
                logger.info(f"Audiobook successfully created: {output_file}")
                return str(output_file)
                
            except Exception as e:
                logger.error(f"Conversion failed: {e}")
                # Clean up partial output
                if output_file.exists():
                    output_file.unlink()
                raise
    
    def _save_text_files(self, chunks: List[TextChunk], text_dir: Path):
        """save processed text files."""
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
        """Generate audio files in parallel with better error handling."""
        results = []
        book_title_clean = book_title.replace('-', ' ').replace('_', ' ').title()
        
        # Adjust workers based on chunk count to prevent overwhelming the system
        max_workers = min(self.config.workers, max(1, len(chunks) // 10))
        logger.info(f"Using {max_workers} workers for {len(chunks)} chunks")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for i, chunk in enumerate(chunks):
                is_first = i == 0
                future = executor.submit(
                    self.audio_processor.synthesize_chunk,
                    chunk, temp_dir, book_title_clean, is_first
                )
                futures[future] = chunk.id
            
            completed = 0
            failed = 0
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Synthesizing"):
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    if result:
                        results.append(result)
                        completed += 1
                    else:
                        failed += 1
                        logger.warning(f"Audio synthesis returned empty for chunk {futures[future]}")
                except TimeoutError:
                    failed += 1
                    logger.warning(f"Audio synthesis timeout for chunk {futures[future]}")
                except Exception as e:
                    failed += 1
                    logger.warning(f"Audio synthesis failed for chunk {futures[future]}: {e}")
                
                # Log progress
                if (completed + failed) % 10 == 0:
                    logger.info(f"Audio progress: {completed} completed, {failed} failed")
        
        logger.info(f"Audio generation complete: {completed} successful, {failed} failed")
        return results
    
    def _save_metadata(self, pdf_file: Path, chunks: List[TextChunk], text_dir: Path, output_file: Path):
        """Save conversion metadata."""
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
        """sort chunks by page numbers."""
        stem = Path(file_path).stem.replace("chunk_", "")
        
        if "-" in stem:
            if "." in stem:
                page_part, subpart = stem.split(".", 1)
            else:
                page_part, subpart = stem, "0"
            
            start_page = int(page_part.split("-")[0])
            subpart = int(subpart)
            return (start_page, subpart)
        
        return (0, 0)


def signal_handler(signum, frame):
    """handle interrupt signals gracefully."""
    print("\ninterrupted by user, cleaning up...")
    sys.exit(1)

def main():
    """Main entry point with comprehensive error handling and performance monitoring."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize performance monitoring
    monitor = PerformanceMonitor()
    
    if len(sys.argv) < 2:
        print("Usage: python pdf2audio_llm.py <pdf_file> [OPTIONS]")
        print("Options:")
        print("  --no-llm    Disable LLM enhancement for faster processing")
        print("  --verbose   Enable verbose logging")
        print("  --debug     Enable debug mode")
        print("")
        print("The script uses GPT-2 Medium for high-quality text enhancement.")
        print("Use --no-llm for fastest processing without LLM enhancement.")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    # Parse command line options
    if "--no-llm" in sys.argv:
        config.enable_llm_enhancement = False
        logger.info("LLM enhancement disabled for faster processing")
    else:
        logger.info("Using GPT-2 Medium for text enhancement")
    
    if "--verbose" in sys.argv:
        logger.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    if "--debug" in sys.argv:
        config.max_retries = 1
        config.timeout_seconds = 300
        logger.info("Debug mode enabled - reduced timeouts and retries")
    
    # Skip LLM loading entirely if disabled
    if not config.enable_llm_enhancement:
        logger.info("Skipping LLM model loading for faster processing")
    
    # Validate input
    if not pdf_file.lower().endswith('.pdf'):
        print("Error: Input file must be a PDF file.")
        sys.exit(1)
    
    if not Path(pdf_file).exists():
        print(f"Error: File '{pdf_file}' not found.")
        sys.exit(1)
    
    # Check file size
    file_size = Path(pdf_file).stat().st_size
    if file_size == 0:
        print("Error: PDF file is empty.")
        sys.exit(1)
    
    # Check available memory
    memory_info = MemoryManager.get_memory_info()
    if "error" not in memory_info:
        logger.info(f"Available memory: {memory_info['rss_mb']:.1f}MB RSS, {memory_info['vms_mb']:.1f}MB VMS")
    
    print(f"Processing PDF: {pdf_file} ({file_size / 1024 / 1024:.1f} MB)")
    
    start_time = time.time()
    
    try:
        with resource_manager(monitor, logger):
            print("Initializing PDF to Audio converter...")
            converter = PDFToAudioConverter(config)
            
            print("Starting conversion...")
            print("⏳ This may take a while for large documents...")
            output_file = converter.convert(pdf_file)
            
            elapsed_time = time.time() - start_time
            print(f"Conversion completed successfully in {elapsed_time:.1f} seconds!")
            print(f"Output: {output_file}")
            
            # Show output file size
            if Path(output_file).exists():
                output_size = Path(output_file).stat().st_size
                print(f"Output size: {output_size / 1024 / 1024:.1f} MB")
            
            # Show performance statistics
            stats = monitor.get_stats()
            if "error" not in stats:
                print(f"Performance: Peak memory: {stats.get('memory_peak_mb', 0):.1f}MB, "
                      f"Peak CPU: {stats.get('cpu_peak_percent', 0):.1f}%")
        
    except KeyboardInterrupt:
        print("\nConversion interrupted by user")
        monitor.stop_monitoring()
        sys.exit(1)
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error during conversion after {elapsed_time:.1f} seconds: {e}")
        logger.error(f"Conversion failed: {e}", exc_info=True)
        monitor.stop_monitoring()
        sys.exit(1)


if __name__ == "__main__":
    main() 