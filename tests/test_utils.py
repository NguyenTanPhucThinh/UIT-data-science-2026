from src.utils.logger import get_logger
from src.utils.file_io import load_jsonl, save_jsonl
from src.utils.timer import timeit, format_time
from src.utils.metrics import calculate_recall, calculate_mrr, calculate_ndcg
from src.config.paths import PROCESSED_DATA_DIR
import time

def test_all_utils():
    # 1. Test Logger
    logger = get_logger("test_utils")
    logger.info("Starting utility tests...")

    # 2. Test Time Utils
    with timeit("Sleeping for 0.5 seconds"):
        time.sleep(0.5)
    
    logger.info(f"Format time test (130s): {format_time(130)}")

    # 3. Test File IO
    test_file = PROCESSED_DATA_DIR / "test_io.jsonl"
    dummy_data = [{"id": 1, "text": "Hello"}, {"id": 2, "text": "World"}]
    
    logger.info(f"Saving JSONL to {test_file}")
    save_jsonl(dummy_data, test_file)
    
    logger.info("Loading JSONL...")
    loaded_data = load_jsonl(test_file)
    assert len(loaded_data) == 2
    logger.info("File IO verified successfully.")
    
    # Cleanup
    test_file.unlink()

    # 4. Test Metrics
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    gold = ["doc2"] # Gold is at rank 2
    
    r5 = calculate_recall(retrieved, gold, k=5)
    m5 = calculate_mrr(retrieved, gold, k=5)
    n5 = calculate_ndcg(retrieved, gold, k=5)
    
    logger.info(f"Metrics Test (Gold rank 2) - Recall@5: {r5:.2f}, MRR@5: {m5:.2f}, NDCG@5: {n5:.2f}")
    assert r5 == 1.0
    assert m5 == 0.5
    
    logger.info("All utilities are working perfectly!")

if __name__ == "__main__":
    test_all_utils()
