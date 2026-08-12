import os
import sys
import zipfile
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Set up project root in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger
from src.utils.timer import timeit

logger = get_logger("build_pipeline")

def ensure_raw_data():
    """
    Automatically detects and extracts the master competition zip file 
    (e.g., 'LegalIR - Public Test-*.zip') as well as the nested 'selected-contexts.zip'.
    """
    extract_dir = RAW_DATA_DIR / "selected-contexts"
    
    # Check if raw context files already exist
    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info("Raw context files already unzipped in data/raw/selected-contexts/")
        return

    # 1. Look for Master Zip file (e.g. LegalIR*.zip or any zip in data/raw except selected-contexts.zip)
    master_zips = list(RAW_DATA_DIR.glob("LegalIR*.zip")) + list(RAW_DATA_DIR.glob("*.zip"))
    master_zips = [z for z in master_zips if z.name != "selected-contexts.zip"]
    
    if master_zips:
        master_zip = master_zips[0]
        logger.info(f"Detected master competition zip: {master_zip.name}. Extracting...")
        with zipfile.ZipFile(master_zip, 'r') as zip_ref:
            # Extract contents directly into RAW_DATA_DIR
            for member in zip_ref.namelist():
                filename = Path(member).name
                if not filename:
                    continue
                # Copy file out directly to RAW_DATA_DIR
                target_path = RAW_DATA_DIR / filename
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())
        logger.info("Master zip extraction complete.")

    # 2. Extract nested selected-contexts.zip if needed
    zip_path = RAW_DATA_DIR / "selected-contexts.zip"
    if zip_path.exists():
        logger.info(f"Extracting nested context archive: {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        logger.info("Context files extraction complete.")
    else:
        logger.warning(f"No selected-contexts.zip found in {RAW_DATA_DIR}! Please place the competition zip file in data/raw/")

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
