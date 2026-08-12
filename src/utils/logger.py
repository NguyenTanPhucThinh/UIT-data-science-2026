import logging
import sys

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a standardized logger for the project.
    
    Args:
        name: Name of the module requesting the logger (typically __name__)
        level: Logging level (default INFO)
        
    Returns:
        Configured logging.Logger object
    """
    logger = logging.getLogger(name)
    
    # Only configure if it doesn't already have handlers to avoid duplication
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handler
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)
        
    return logger
