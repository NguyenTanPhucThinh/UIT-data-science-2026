import argparse
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_dir))

# Dùng đúng công thức Recall@5 / Precision@5 chính thức của BTC (partial-credit theo tập hợp,
# ràng buộc >5 document -> 0 điểm), thay vì công thức hit-rate nhị phân cũ - để so sánh được
# 1-1 với baseline_retrieval.py (dense) và các model đã fine-tune, tất cả cùng đi qua metrics.py.
from src.evaluation.metrics import evaluate_predictions

# Optional: import dedup utility if available
try:
    eda_dir = Path(__file__).resolve().parent.parent / 'ingestion + EDA'
    sys.path.insert(0, str(eda_dir))
    from dedup_corpus import apply_dedup_to_predictions
    DEDUP_AVAILABLE = True
except ImportError:
    DEDUP_AVAILABLE = False

TOP_K = 5


def load_chunks(filepath):
    print(f"Loading chunks from {filepath}...")
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_queries(filepath):
    print(f"Loading queries from {filepath}...")
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def evaluate_bm25(chunk_file=None, query_file=None, limit=None, report_path=None, use_dedup=False):
    chunk_file = Path(chunk_file) if chunk_file else base_dir / 'data' / 'processed' / 'chunks_corpus.jsonl'
    query_file = Path(query_file) if query_file else base_dir / 'data' / 'processed' / 'val_preprocessed.jsonl'
    dedup_mapping_file = base_dir / 'data' / 'processed' / 'dedup_mapping.json'

    dedup_mapping = {}
    if use_dedup:
        if dedup_mapping_file.exists():
            with open(dedup_mapping_file, 'r', encoding='utf-8') as f:
                dedup_mapping = json.load(f)
            print(f"Loaded dedup_mapping.json: {len(dedup_mapping)} alias entries")
        else:
            print("WARNING: --dedup flag set but dedup_mapping.json not found. Run build_pipeline.py first.")

    if not chunk_file.exists() or not query_file.exists():
        print("Missing processed data files! Please run Phase 1-4 first.")
        return

    chunks = load_chunks(chunk_file)
    queries = load_queries(query_file)
    if limit is not None:
        queries = queries[:limit]

    print("Tokenizing chunk passages for BM25 (using pre-segmented text)...")
    t0 = time.time()
    tokenized_corpus = []
    chunk_doc_ids = []

    for c in chunks:
        # The text is already segmented by VnCoreNLP, so space splits word tokens
        text = c.get('passage_segmented', '')
        tokenized_corpus.append(text.split())
        chunk_doc_ids.append(c['doc_id'])

    print(f"Tokenization complete in {time.time() - t0:.2f}s. Building BM25 index...")
    t0 = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built in {time.time() - t0:.2f}s.")

    print(f"\nRunning Evaluation on Validation Set ({len(queries)} queries)...")

    ground_truth = {}
    predictions = {}
    mrr_5_sum = 0.0

    for q in tqdm(queries, desc="Evaluating"):
        qid = str(q.get('question_id'))
        q_tokens = q.get('question_segmented', '').split()
        gold_docs = list(q.get('answer_doc_ids') or [])
        if not gold_docs:
            # Không có ground truth -> bỏ qua câu này khỏi cả tử số lẫn mẫu số, giống
            # data_utils.load_labeled_questions() lọc answer rỗng ở phía dense baseline.
            continue

        # Get chunk-level scores
        chunk_scores = bm25.get_scores(q_tokens)

        # Max-Score Aggregation mapping chunk_scores to doc_ids
        doc_scores = {}
        for doc_id, score in zip(chunk_doc_ids, chunk_scores):
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

        # Sort documents by max score descending
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [doc_id for doc_id, score in ranked_docs[:TOP_K]]

        # Apply dedup remapping if enabled
        if use_dedup and dedup_mapping:
            top_docs = apply_dedup_to_predictions(top_docs, dedup_mapping)

        ground_truth[qid] = gold_docs
        predictions[qid] = top_docs

        # MRR@5 - chỉ số bổ sung để debug thứ hạng, KHÔNG phải công thức chính thức BTC
        # (BTC chỉ quan tâm tập hợp top-5, không quan tâm rank bên trong).
        for rank, doc_id in enumerate(top_docs, 1):
            if doc_id in set(gold_docs):
                mrr_5_sum += 1.0 / rank
                break

    result = evaluate_predictions(ground_truth, predictions)

    print("=" * 50)
    print("      BM25 BASELINE RESULTS")
    print(f"      (Validation Set - {result['n_questions']} Queries)")
    print("      Aggregation: Max-Score")
    if use_dedup:
        print(f"      Deduplication: ON ({len(dedup_mapping)} alias remappings)")
    print("      Cong thuc: Recall/Precision chinh thuc BTC (metrics.py)")
    print("=" * 50)
    print(f"Recall@5    : {result['avg_recall']:.4f}")
    print(f"Precision@5 : {result['avg_precision']:.4f}")
    print(f"Thiếu dự đoán            : {result['n_missing_predictions']}")
    print(f"Vi phạm (>5 document_id) : {result['n_violations_over_5_docs']}")
    if result['n_questions']:
        print(f"MRR@5 (bổ sung, không thuộc công thức BTC): {mrr_5_sum / result['n_questions']:.4f}")
    print("=" * 50)

    if report_path:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu báo cáo chi tiết: {report_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="BM25 baseline retrieval, chấm theo công thức BTC")
    parser.add_argument("--chunks", default=None, help="Đường dẫn chunks_corpus.jsonl (mặc định: data/processed/chunks_corpus.jsonl)")
    parser.add_argument("--queries", default=None, help="Đường dẫn val_preprocessed.jsonl (mặc định: data/processed/val_preprocessed.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Chi chay N cau hoi dau (debug nhanh). Mac dinh: chay het")
    parser.add_argument("--report", default=None, help="Neu chi dinh, luu bao cao chi tiet (JSON) ra duong dan nay")
    parser.add_argument("--dedup", action="store_true", help="Ap dung dedup_mapping.json de remap alias doc_ids truoc khi tinh metrics")
    args = parser.parse_args()

    evaluate_bm25(chunk_file=args.chunks, query_file=args.queries, limit=args.limit, report_path=args.report, use_dedup=args.dedup)


if __name__ == "__main__":
    main()
