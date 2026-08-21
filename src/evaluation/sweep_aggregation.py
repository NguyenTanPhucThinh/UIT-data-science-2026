"""
So sánh CÁCH GỘP ĐIỂM từ chunk lên document (`aggregate_to_documents`) trên val_split_20, giữ NGUYÊN
mọi tham số khác ở đỉnh đã có bằng chứng (dense-top-n=100, bm25-top-n=100, rrf-k=60, bm25-weight=0.1
— xem dòng #11b và mục "sweep LƯỚI THẬT dense-top-n/bm25-top-n/rrf-k" trong submission-log-task1.md).

ĐỘNG LỰC (2026-08-17): `aggregate_to_documents()` trong `full_pipeline_eval.py` LUÔN lấy chunk có RRF
score CAO NHẤT làm đại diện cho cả document (max-score), CHƯA TỪNG thử cách khác trong suốt log dự
án. Với câu hỏi có n_relevant>=2 (nhóm ĐANG YẾU NHẤT trong mọi lần đo, Recall chỉ ~0.74-0.79 dù n=1 đã
~0.94-0.95), 1 document có thể có NHIỀU chunk cùng liên quan (vd văn bản luật dài, câu trả lời trải ra
nhiều điều khoản) - max-score chỉ dùng đúng 1 chunk mạnh nhất, bỏ qua tín hiệu từ các chunk khác cùng
tài liệu. Giả thuyết: cộng dồn điểm từ NHIỀU chunk cùng doc có thể giúp document "nhiều bằng chứng"
được xếp hạng cao hơn.

CÁC CÁCH GỘP được so sánh (giữ nguyên `max` làm baseline đối chứng):
  - max        : baseline hiện tại - lấy chunk RRF score cao nhất/doc (KHÔNG đổi gì).
  - sum_top2   : cộng 2 chunk RRF score cao nhất/doc (nếu doc chỉ có 1 chunk ứng viên thì = max).
  - sum_top3   : cộng 3 chunk RRF score cao nhất/doc.
  - mean_top2  : trung bình 2 chunk RRF score cao nhất/doc (khác sum_top2 ở chỗ KHÔNG thiên vị
                 document có nhiều chunk ứng viên hơn - sum có thể vô tình ưu ái doc dài/nhiều chunk).

MẸO TỐC ĐỘ: dùng lại đúng cơ chế tính dense/BM25 score 1 LẦN DUY NHẤT + fuse RRF 1 LẦN (tham số fusion
CỐ ĐỊNH ở đỉnh đã biết) - chỉ lặp lại bước GỘP CHUNK->DOC (rẻ) cho từng cách gộp.

Cách dùng (mặc định đã trỏ đúng AITeamVN v1 + cache có sẵn):
    python -m src.evaluation.sweep_aggregation --report outputs/predictions/sweep_aggregation_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from .evaluate import load_ground_truth
    from .metrics import evaluate_predictions
    from .full_pipeline_eval import (
        load_chunks, load_questions, resolve_device, encode_chunks, build_bm25_index, _naive_tokenize,
        build_bm25_fast_matrix, bm25_scores_batch_fast,
    )
except ImportError:  # cho phép chạy trực tiếp
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.metrics import evaluate_predictions
    from src.evaluation.full_pipeline_eval import (
        load_chunks, load_questions, resolve_device, encode_chunks, build_bm25_index, _naive_tokenize,
        build_bm25_fast_matrix, bm25_scores_batch_fast,
    )


def top_n_indices(scores, top_n: int):
    import numpy as np

    n = len(scores)
    top_n = min(top_n, n)
    idx = np.argpartition(-scores, top_n - 1)[:top_n]
    return idx[np.argsort(-scores[idx])]


def fuse_from_ranks(dense_top_idx, bm25_top_idx, rrf_k: int, bm25_weight: float) -> Dict[int, float]:
    rrf_score: Dict[int, float] = defaultdict(float)
    for rank, idx in enumerate(dense_top_idx, start=1):
        rrf_score[int(idx)] += 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(bm25_top_idx, start=1):
        rrf_score[int(idx)] += bm25_weight / (rrf_k + rank)
    return rrf_score


def aggregate_max(rrf_score: Dict[int, float], doc_ids: Sequence[str], top_k: int) -> List[str]:
    """Baseline HIỆN TẠI (copy hành vi của aggregate_to_documents() trong full_pipeline_eval.py)."""
    best_by_doc: Dict[str, float] = {}
    for chunk_idx, score in rrf_score.items():
        doc_id = doc_ids[chunk_idx]
        if doc_id not in best_by_doc or score > best_by_doc[doc_id]:
            best_by_doc[doc_id] = score
    ranked = sorted(best_by_doc.items(), key=lambda x: -x[1])
    return [doc_id for doc_id, _ in ranked[:top_k]]


def aggregate_topk(rrf_score: Dict[int, float], doc_ids: Sequence[str], top_k: int, k: int, mode: str) -> List[str]:
    """mode='sum' hoặc 'mean' - gộp k chunk RRF score CAO NHẤT của mỗi document."""
    by_doc: Dict[str, List[float]] = defaultdict(list)
    for chunk_idx, score in rrf_score.items():
        by_doc[doc_ids[chunk_idx]].append(score)

    doc_scores: Dict[str, float] = {}
    for doc_id, scores in by_doc.items():
        scores.sort(reverse=True)
        top_scores = scores[:k]
        doc_scores[doc_id] = sum(top_scores) if mode == "sum" else (sum(top_scores) / len(top_scores))

    ranked = sorted(doc_scores.items(), key=lambda x: -x[1])
    return [doc_id for doc_id, _ in ranked[:top_k]]


AGGREGATORS = {
    "max": lambda rrf_score, doc_ids, top_k: aggregate_max(rrf_score, doc_ids, top_k),
    "sum_top2": lambda rrf_score, doc_ids, top_k: aggregate_topk(rrf_score, doc_ids, top_k, 2, "sum"),
    "sum_top3": lambda rrf_score, doc_ids, top_k: aggregate_topk(rrf_score, doc_ids, top_k, 3, "sum"),
    "mean_top2": lambda rrf_score, doc_ids, top_k: aggregate_topk(rrf_score, doc_ids, top_k, 2, "mean"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-model", default="AITeamVN/Vietnamese_Embedding")
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    parser.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    parser.add_argument("--embedding-cache", default="outputs/cache/embcompare_aiteamvn.npz")
    parser.add_argument("--dense-top-n", type=int, default=100)
    parser.add_argument("--bm25-top-n", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--question-batch-size", type=int, default=64)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--passage-prefix", default="")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--fp16-inference", dest="fp16_inference", action="store_true", default=True)
    parser.add_argument("--no-fp16-inference", dest="fp16_inference", action="store_false")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[lỗi] Thiếu sentence-transformers.", file=sys.stderr)
        sys.exit(1)
    try:
        import rank_bm25, scipy, pandas  # noqa: F401
    except ImportError:
        print("[lỗi] Thiếu rank_bm25/scipy/pandas.", file=sys.stderr)
        sys.exit(1)
    import numpy as np

    device = resolve_device(args.device)
    if device == "cuda":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ground_truth = load_ground_truth(args.ground_truth)
    chunks = load_chunks(args.chunks)
    questions = load_questions(args.questions)
    if args.limit_questions:
        questions = questions[: args.limit_questions]
    doc_ids = [c["doc_id"] for c in chunks]

    print(f"Load embedding model: {args.embedding_model}")
    embedding_model = SentenceTransformer(args.embedding_model, device=device)
    if args.max_seq_length is not None:
        embedding_model.max_seq_length = args.max_seq_length
        print(f"Đã ghi đè embedding_model.max_seq_length = {args.max_seq_length}")
    if device == "cuda" and args.fp16_inference:
        try:
            embedding_model = embedding_model.half()
        except Exception as e:  # noqa: BLE001
            print(f"[cảnh báo] Không cast fp16 ({e}).", file=sys.stderr)

    chunk_embeddings = encode_chunks(
        embedding_model, chunks, args.batch_size, args.embedding_cache, passage_prefix=args.passage_prefix
    )
    bm25_index = build_bm25_index(chunks)
    print("Xây ma trận BM25 vector hoá (--bm25-fast)...")
    bm25_fast = build_bm25_fast_matrix(bm25_index)

    print(f"Tính RRF fusion (dense-top-n={args.dense_top_n}, bm25-top-n={args.bm25_top_n}, "
          f"rrf-k={args.rrf_k}, bm25-weight={args.bm25_weight}) cho {len(questions)} câu hỏi (1 lần)...")
    per_question_rrf: List[Tuple[str, Dict[int, float]]] = []
    for start in range(0, len(questions), args.question_batch_size):
        batch = questions[start : start + args.question_batch_size]
        batch_texts = [args.query_prefix + q["question"] for q in batch]
        batch_dense_emb = embedding_model.encode(
            batch_texts, batch_size=args.question_batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)
        dense_sims_batch = batch_dense_emb @ chunk_embeddings.T

        q_tokens_batch = [
            q["question_segmented"].split() if q.get("question_segmented") else _naive_tokenize(q["question"])
            for q in batch
        ]
        bm25_scores_batch = bm25_scores_batch_fast(bm25_fast, q_tokens_batch)

        for i, q in enumerate(batch):
            dense_top_idx = top_n_indices(dense_sims_batch[i], args.dense_top_n)
            bm25_top_idx = top_n_indices(bm25_scores_batch[i], args.bm25_top_n)
            rrf_score = fuse_from_ranks(dense_top_idx, bm25_top_idx, args.rrf_k, args.bm25_weight)
            per_question_rrf.append((q["question_id"], rrf_score))
        print(f"  đã xử lý {min(start + args.question_batch_size, len(questions))}/{len(questions)} câu hỏi")

    print(f"\nSo sánh {len(AGGREGATORS)} cách gộp chunk->document...")
    results = {}
    for agg_name, agg_fn in AGGREGATORS.items():
        predictions: Dict[str, List[str]] = {}
        for qid, rrf_score in per_question_rrf:
            predictions[qid] = agg_fn(rrf_score, doc_ids, args.top_k)
        results[agg_name] = evaluate_predictions(ground_truth, predictions)

    print("\n" + "=" * 60)
    print(f"{'Cách gộp':>12s} {'Recall':>10s} {'Precision':>11s} {'Δ vs max':>10s}")
    print("-" * 60)
    baseline_recall = results["max"]["avg_recall"]
    for agg_name, r in sorted(results.items(), key=lambda kv: -kv[1]["avg_recall"]):
        delta = r["avg_recall"] - baseline_recall
        sign = "+" if delta >= 0 else ""
        marker = "  <-- baseline (hiện tại)" if agg_name == "max" else ""
        print(f"{agg_name:>12s} {r['avg_recall']:>10.4f} {r['avg_precision']:>11.4f} {sign}{delta:>9.4f}{marker}")
    print("=" * 60)

    best_agg = max(results, key=lambda a: results[a]["avg_recall"])
    best_delta = results[best_agg]["avg_recall"] - baseline_recall
    if best_agg != "max" and best_delta >= 0.005:
        print(f"\n-> '{best_agg}' cải thiện Recall +{best_delta:.4f} so với 'max' (baseline hiện tại) - "
              "ĐÁNG đổi cấu hình + xác nhận lại trên public test trước khi chốt.")
    else:
        print(f"\n-> Không cách gộp nào cải thiện rõ rệt (>=+0.005) so với 'max' hiện tại - "
              "giữ nguyên aggregate_to_documents() như cũ.")

    # Breakdown theo n_relevant cho cách tốt nhất (nếu khác max) và max, để soi kỹ nhóm n>=2
    def n_relevant_breakdown(predictions_by_agg):
        pass  # (không cần thiết cho báo cáo tóm tắt - report JSON đã đủ chi tiết nếu cần đào sâu thêm)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            agg_name: {"avg_recall": r["avg_recall"], "avg_precision": r["avg_precision"]}
            for agg_name, r in results.items()
        }
        report_data["_best"] = best_agg
        report_data["_baseline"] = "max"
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu báo cáo so sánh: {args.report}")


if __name__ == "__main__":
    main()
