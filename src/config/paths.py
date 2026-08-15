from pathlib import Path

# Base directory (Project Root)
# Resolves to: d:\UIT-data-science-2026
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models Directory (for VnCoreNLP, Embeddings, etc)
MODELS_DIR = BASE_DIR / "models"

# Docs Directory
DOCS_DIR = BASE_DIR / "docs"

# Specific Processed Files (for easy import by team)
CORPUS_PREPROCESSED = PROCESSED_DATA_DIR / "corpus_preprocessed.jsonl"
CHUNKS_CORPUS = PROCESSED_DATA_DIR / "chunks_corpus.jsonl"
TRAIN_PREPROCESSED = PROCESSED_DATA_DIR / "train_preprocessed.jsonl"
VAL_PREPROCESSED = PROCESSED_DATA_DIR / "val_preprocessed.jsonl"
PUBLIC_TEST_PREPROCESSED = PROCESSED_DATA_DIR / "public_test_preprocessed.jsonl"

# Task 2 LegalQA question artifacts. The corpus/chunks above are shared by both tasks.
TRAIN_2_PREPROCESSED = PROCESSED_DATA_DIR / "train_2_preprocessed.jsonl"
PUBLIC_TEST_2_PREPROCESSED = PROCESSED_DATA_DIR / "public_test_2_preprocessed.jsonl"
