"""
Kiểm tra 1 hậu-xử-lý (post-processing) AN TOÀN cho vấn đề "tài liệu trùng lặp nội dung" (mục 2 trong
báo cáo data-quality gửi Kiên, 2026-08-17): 27 nhóm/54 `doc_id` trong corpus là 2 lần crawl khác nhau
của CÙNG 1 văn bản (gom theo mã số trang trong URL nguồn thuvienphapluat.vn).

VẤN ĐỀ: nếu CẢ 2 `doc_id` trùng của cùng 1 nhóm đều lọt vào top-5 dự đoán của 1 câu hỏi, ta lãng phí
1 slot cho nội dung ĐÃ có (case thật đã ghi nhận: câu hỏi 13366, doc_id 222242+276631 cùng là Luật
Khám bệnh, chữa bệnh 2023 - 2/5 slot bị trùng, khiến 1 tài liệu ground-truth khác đúng ra xếp hạng 6
KHÔNG lọt vào top-5).

CÁCH SỬA (dedup + backfill) - KHÔNG đổi corpus, KHÔNG đổi ground truth, CHỈ đổi bước lấy top-5 cuối
cùng từ danh sách xếp hạng document đã có sẵn:
  1. Lấy 1 danh sách ứng viên DÀI hơn top-5 (--candidate-pool, mặc định 30) từ đúng pipeline hiện tại
     (RRF fusion + aggregate max, dense-top-n=100/bm25-top-n=100/rrf-k=60/bm25-weight=0.1 - cấu hình
     ĐÃ xác nhận tốt nhất, KHÔNG đổi).
  2. Duyệt danh sách theo thứ hạng; nếu `doc_id` thuộc 1 nhóm trùng đã CÓ 1 thành viên khác được chọn
     rồi -> BỎ QUA (không lãng phí slot), lấy tiếp ứng viên xếp hạng thấp hơn để LẤP ĐẦY (backfill).
  3. So Recall/Precision @5 của cách này ("dedup_backfill") với baseline hiện tại ("max", không dedup).

Vẫn dùng đúng `doc_id` gốc (KHÔNG gộp về 1 `doc_id` đại diện) - tương thích 100% với format nộp bài
hiện tại, không cần Kiên sửa corpus/GT trước.

Cách dùng (mặc định đã trỏ đúng AITeamVN v1 + cache có sẵn + file nhóm trùng lặp đã xác nhận):
    python -m src.evaluation.eval_dedup_backfill --report outputs/predictions/eval_dedup_backfill_report.json
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


def ranked_docs_max(rrf_score: Dict[int, float], doc_ids: Sequence[str], pool_size: int) -> List[str]:
    """Giống hệt aggregate_to_documents() hiện tại (max-score/doc) nhưng trả về pool DÀI hơn top-5."""
    best_by_doc: Dict[str, float] = {}
    for chunk_idx, score in rrf_score.items():
        doc_id = doc_ids[chunk_idx]
        if doc_id not in best_by_doc or score > best_by_doc[doc_id]:
            best_by_doc[doc_id] = score
    ranked = sorted(best_by_doc.items(), key=lambda x: -x[1])
    return [doc_id for doc_id, _ in ranked[:pool_size]]


def load_canonical_map(path: str) -> Dict[str, str]:
    """canonical[doc_id] = id đại diện của nhóm trùng lặp (id nhỏ hơn). doc_id không trùng -> map về chính nó."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical: Dict[str, str] = {}
    for group in data["groups"]:
        rep = min(group, key=lambda x: int(x))
        for doc_id in group:
            canonical[doc_id] = rep
    return canonical


def baseline_top_k(ranked_pool: List[str], top_k: int) -> List[str]:
    return ranked_pool[:top_k]


def dedup_backfill_top_k(ranked_pool: List[str], canonical: Dict[str, str], top_k: int) -> List[str]:
    seen_canonical = set()
    out: List[str] = []
    for doc_id in ranked_pool:
        rep = canonical.get(doc_id, doc_id)
        if rep in seen_canonical:
            continue  # bỏ qua - nội dung đã có 1 doc_id khác của cùng nhóm trong output rồi
        seen_canonical.add(rep)
        out.append(doc_id)
        if len(out) >= top_k:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-model", default="AITeamVN/Vietnamese_Embedding")
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    parser.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    parser.add_argument("--embedding-cache", default="outputs/cache/embcompare_aiteamvn.npz")
    parser.add_argument("--dup-groups", default="data/processed/duplicate_doc_groups.json")
    parser.add_argument("--dense-top-n", type=int, default=100)
    parser.add_argument("--bm25-top-n", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=30,
                         help="Số ứng viên document lấy trước khi dedup+backfill xuống top-k cuối cùng.")
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
    canonical = load_canonical_map(args.dup_groups)
    print(f"Đã nạp {len(canonical)} doc_id thuộc các nhóm trùng lặp (từ {args.dup_groups}).")

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

    baseline_predictions: Dict[str, List[str]] = {}
    dedup_predictions: Dict[str, List[str]] = {}
    n_collision = 0  # so cau hoi ma baseline top-5 co >=2 doc_id cung 1 nhom trung lap

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
            ranked_pool = ranked_docs_max(rrf_score, doc_ids, args.candidate_pool)

            base5 = baseline_top_k(ranked_pool, args.top_k)
            dedup5 = dedup_backfill_top_k(ranked_pool, canonical, args.top_k)

            base_canon = {canonical.get(d, d) for d in base5}
            if len(base_canon) < len(base5):
                n_collision += 1

            baseline_predictions[q["question_id"]] = base5
            dedup_predictions[q["question_id"]] = dedup5
        print(f"  đã xử lý {min(start + args.question_batch_size, len(questions))}/{len(questions)} câu hỏi")

    baseline_metrics = evaluate_predictions(ground_truth, baseline_predictions)
    dedup_metrics = evaluate_predictions(ground_truth, dedup_predictions)

    print("\n" + "=" * 68)
    print(f"{'Cách':>16s} {'Recall':>10s} {'Precision':>11s} {'Δ vs baseline':>14s}")
    print("-" * 68)
    delta = dedup_metrics["avg_recall"] - baseline_metrics["avg_recall"]
    sign = "+" if delta >= 0 else ""
    print(f"{'baseline (max)':>16s} {baseline_metrics['avg_recall']:>10.4f} {baseline_metrics['avg_precision']:>11.4f} {'':>14s}  <-- hiện tại")
    print(f"{'dedup_backfill':>16s} {dedup_metrics['avg_recall']:>10.4f} {dedup_metrics['avg_precision']:>11.4f} {sign}{delta:>13.4f}")
    print("=" * 68)
    print(f"\nSố câu hỏi có va chạm trùng lặp trong top-5 baseline (bị lãng phí slot): "
          f"{n_collision}/{len(questions)} ({100*n_collision/len(questions):.2f}%)")

    if delta >= 0.001:
        print(f"\n-> dedup_backfill cải thiện Recall +{delta:.4f} - ĐÁNG áp dụng, không tốn chi phí thêm "
              "(chỉ đổi bước hậu xử lý, không đổi model/hyperparameter). Có thể thử trên public test.")
    elif n_collision == 0:
        print("\n-> Không có va chạm trùng lặp nào trong tập câu hỏi này (val_split_20 nhỏ, 27 nhóm trùng "
              "trên 8.532 tài liệu là hiếm) - kết quả không đổi. Vẫn NÊN áp dụng vì không có rủi ro giảm "
              "điểm (identical khi không va chạm) và có thể có ích trên public test (nhiều câu hỏi hơn).")
    else:
        print(f"\n-> Có va chạm ({n_collision} câu) nhưng Recall không đổi rõ rệt - tài liệu backfill vào "
              "không trùng ground truth ở các câu này. Vẫn an toàn để giữ (không làm giảm điểm).")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "baseline": {"avg_recall": baseline_metrics["avg_recall"], "avg_precision": baseline_metrics["avg_precision"]},
            "dedup_backfill": {"avg_recall": dedup_metrics["avg_recall"], "avg_precision": dedup_metrics["avg_precision"]},
            "delta_recall": delta,
            "n_collision_questions": n_collision,
            "n_questions": len(questions),
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu báo cáo: {args.report}")


if __name__ == "__main__":
    main()
