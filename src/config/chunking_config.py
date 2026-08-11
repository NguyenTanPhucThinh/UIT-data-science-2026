# src/config/chunking_config.py

# Default max sequence length for chunking.
# Most BERT-based embedding models use 512.
MAX_SEQ_LENGTH = 512

# Overlap for sliding window fallback
OVERLAP_TOKENS = 50

# Target length for short docs that don't need semantic splitting
SHORT_DOC_MAX_CHARS = 5000

# We aggregate chunks back to document level.
# Options: 'max-score', 'mean-score'
AGGREGATION_METHOD = 'max-score'
