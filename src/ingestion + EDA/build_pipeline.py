import os
import sys
import zipfile
from pathlib import Path

# Set up project root in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger
from src.utils.timer import timeit

logger = get_logger("build_pipeline")

def ensure_raw_data():
    """Unzips selected-contexts.zip if selected-contexts directory is missing."""
    zip_path = RAW_DATA_DIR / "selected-contexts.zip"
    extract_dir = RAW_DATA_DIR / "selected-contexts"
    
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        if zip_path.exists():
            logger.info(f"Unzipping {zip_path} to {extract_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            logger.info("Unzipping complete.")
        else:
            logger.warning(f"Neither {extract_dir} nor {zip_path} found! Please place selected-contexts.zip in data/raw/")

def run_all():
    logger.info("==================================================")
    logger.info("  STARTING AUTOMATED DATA PIPELINE BUILD (PHASES 1-4)")
    logger.info("==================================================")
    
    # 0. Ensure raw data exists
    ensure_raw_data()
    
    # Ensure src/ingestion + EDA is in sys.path
    eda_dir = project_root / "src" / "ingestion + EDA"
    if str(eda_dir) not in sys.path:
        sys.path.insert(0, str(eda_dir))
        
    import run_phase1
    import run_phase2
    import run_phase4
    
    # Phase 1: Audit & Corpus Unification
    with timeit("Phase 1: Audit & Corpus Unification"):
        logger.info("\n---> Running Phase 1...")
        run_phase1.run_audit()
        
    # Phase 2: Normalization & VnCoreNLP Segmentation
    with timeit("Phase 2: Normalization & VnCoreNLP Segmentation"):
        logger.info("\n---> Running Phase 2...")
        run_phase2.run_phase2()
        
    # Phase 4: Structural Chunking (Điều/Chương)
    with timeit("Phase 4: Adaptive Structural Chunking"):
        logger.info("\n---> Running Phase 4...")
        run_phase4.run_phase4()
        
    logger.info("==================================================")
    logger.info("  ALL DATASETS GENERATED SUCCESSFULLY IN data/processed/")
    logger.info("==================================================")

if __name__ == "__main__":
    run_all()
