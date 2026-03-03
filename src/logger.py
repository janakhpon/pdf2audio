import logging
import sys

def setup_logger(name: str = "pdf2audio") -> logging.Logger:
    """
    Configures and returns a professional, standardized logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger()
