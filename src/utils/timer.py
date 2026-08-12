import time
from contextlib import contextmanager
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

@contextmanager
def timeit(task_name: str, log_level: int = 20):  # 20 is logging.INFO
    """
    Context manager to time a block of code and log its duration.
    
    Usage:
        with timeit("Loading Vectorstore"):
            # slow code here
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.log(log_level, f"{task_name} completed in {elapsed:.3f} seconds.")

def format_time(seconds: float) -> str:
    """Formats seconds into a human-readable HH:MM:SS string."""
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"
