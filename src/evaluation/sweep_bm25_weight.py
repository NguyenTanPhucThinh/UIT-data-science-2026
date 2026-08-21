"""
Dò (sweep) nhiều giá trị --bm25-weight CÙNG LÚC trên val_split_20, để trả lời câu hỏi "tăng trọng
số BM25 trong RRF lên bao nhiêu là hợp lý?" bằng SỐ ĐO THỰC TẾ thay vì đoán mò 1 giá trị rồi chạy
lại full_pipeline_eval.py nhiều lần (rất tốn - mỗi lần chạy lại phải encode corpus + build BM25
index + encode câu hỏi).

Động lực (xem outputs/predictions/error_analysis_recall0.md, mục "Pattern A" - 2026-08-14): nhiều
case recall=0 là do dense embedding nhầm lẫn các văn bản CÙNG KHUÔN MẪU (chỉ khác tên riêng/số
hiệu - vd Điều lệ Kiểm soát viên các Tổng công ty khác nhau, chuỗi tiêu chuẩn TCVN cùng series).
BM25 (khớp từ khóa chính xác) về lý thuyết phân biệt tốt các case này hơn dense (khớp ngữ nghĩa) -
giả thuyết: tăng trọng số BM25 trong RRF fusion có thể sửa được một phần Pattern A, RẺ hơn nhiều
so với đổi hẳn embedding model (không cần fine-tune/tải model mới).

MẸO TỐC ĐỘ: dense_scores và bm25_scores của MỖI câu hỏi chỉ phụ thuộc model/corpus, KHÔNG phụ
thuộc --bm25-weight - nên script này chỉ encode + build BM25 index + tính điểm 1 LẦN DUY NHẤT,
rồi thử lại phép RRF fusion (rất rẻ, chỉ là cộng số) cho TỪNG giá trị --weights truyền vào. Sweep
7 giá trị chỉ chậm hơn chạy full_pipeline_eval.py 1 lần một chút (không phải 7 lần).

QUAN TRỌNG: script này KHÔNG dùng reranker - đo đúng giai đoạn "hybrid_fused" (sau RRF, TRƯỚC
rerank), vì đây là giai đoạn ĐANG DÙNG THẬT trong cấu hình production hiện tại (đã tắt reranker từ
2026-08-14, xem claude/submission-log-task1.md mục "QUYẾT ĐỊNH HIỆN TẠI"). Không cần tải/chạy
reranker model - đỡ thời gian và VRAM.

Cách dùng (mặc định đã trỏ đúng model/data tốt nhất hiện có - config dòng #9 trong
claude/submission-log-task1.md):
    python -m src.evaluation.sweep_bm25_weight \\
        --weights 0.5,1,1.5,2,3,5,8 \\
        --report outputs/predictions/sweep_bm25_weight_report.json

Sau khi có bảng kết quả, ĐỌC KỸ trước khi chốt:
  - Nếu Recall tăng đơn điệu rồi giảm khi weight quá lớn (có đỉnh rõ ràng) -> chọn weight ở đỉnh,
    có thể sweep tinh hơn quanh giá trị đó (vd nếu đỉnh ở weight=2, thử thêm 1.7/1.8/2.2/2.3).
  - Nếu Recall gần như không đổi ở mọi weight -> giả thuyết "BM25 sửa được Pattern A" SAI (hoặc
    yếu hơn kỳ vọng) trên tập val này - đừng cố ép, quay lại hướng so sánh embedding model khác.
  - Nếu Recall CAO NHẤT lại ở weight=0.5 (giảm BM25) -> ngược hoàn toàn giả thuyết, đáng nghi ngờ
    lại phân tích Pattern A hoặc kiểm tra lỗi code trước khi tin kết quả.
  - Precision đi kèm để tham khảo (đo phụ theo thể lệ), KHÔNG dùng để chọn weight nếu ngược hướng
    với Recall (Recall là độ đo chính BTC dùng chấm điểm).
  - Đây là đo trên val_split_20 (dev, không leak) - vẫn nên xác nhận lại trên leaderboard công
    khai trước khi coi là quyết định cuối cùng, đúng quy trình đã áp dụng khi bỏ reranker.
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
        aggregate_to_documents,
    )
except ImportError:  # cho phép chạy trực tiếp
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.metrics import evaluate_predictions
    from src.evaluation.full_pipeline_eval import (
        load_chunks, load_questions, resolve_device, encode_chunks, build_bm25_index, _naive_tokenize,
        aggregate_to_documents,
    )


def top_n_indices(scores, top_n: int):
    """Trả về chỉ số top-N (giảm dần theo điểm) - tách riêng từ rrf_fuse() để tính 1 LẦN, dùng lại
    cho mọi giá trị --bm25-weight (bước tốn thời gian nhất của RRF là sort/argpartition, không
    phải phép cộng trọng số)."""
    import numpy as np

    n = len(scores)
    top_n = min(top_n, n)
    idx = np.argpartition(-scores, top_n - 1)[:top_n]
    return idx[np.argsort(-scores[idx])]


def fuse_from_ranks(
    dense_top_idx, bm25_top_idx, rrf_k: int, bm25_weight: float
) -> Dict[int, float]:
    """Giống phần sau của rrf_fuse() trong full_pipeline_eval.py, nhưng nhận thẳng top-N index đã
    tính sẵn (KHÔNG tính lại top-N mỗi lần đổi weight) - đây là bước RẺ, chạy lại được nhiều lần."""
    rrf_score: Dict[int, float] = defaultdict(float)
    for rank, idx in enumerate(dense_top_idx, start=1):
        rrf_score[int(idx)] += 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(bm25_top_idx, start=1):
        rrf_score[int(idx)] += bm25_weight / (rrf_k + rank)
    return rrf_score


def parse_weights(raw: str) -> List[float]:
    weights = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        weights.append(float(token))
    if not weights:
        raise ValueError(f"--weights rỗng hoặc không hợp lệ: {raw!r}")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-model", default="models/public_train/vietnamese-legal-embedding-ft-full",
                         help="Mặc định = model TỐT NHẤT hiện có, ĐÚNG model dùng cho config dòng #9 (no-rerank, Score PUBLIC 0.8472)")
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--questions", default="data/processed/val_preprocessed.jsonl",
                         help="Nên dùng file *_preprocessed.jsonl (có question_segmented) để BM25 tokenize đúng - KHÔNG dùng val_split_20.json ở đây (đó là --ground-truth)")
    parser.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    parser.add_argument("--embedding-cache", default="outputs/cache/chunks_corpus_embeddings__ft_full.npz",
                         help="Mặc định tái dùng cache đã có sẵn (model -ft-full) - KHÔNG cần encode lại corpus")
    parser.add_argument("--weights", default="0.5,1,1.5,2,3,5,8",
                         help="Danh sách giá trị --bm25-weight để dò, cách nhau bởi dấu phẩy. 1.0 = công thức RRF gốc (baseline để so sánh), LUÔN nên có mặt trong danh sách.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-n", type=int, default=100)
    parser.add_argument("--bm25-top-n", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size encode corpus (dense)")
    parser.add_argument("--question-batch-size", type=int, default=64)
    parser.add_argument("--limit-questions", type=int, default=None, help="Chỉ chạy N câu hỏi đầu (debug nhanh)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--max-seq-length", type=int, default=None,
                         help="Ghi đè embedding_model.max_seq_length sau khi load (vd 2048 cho "
                              "AITeamVN/Vietnamese_Embedding, thêm 2026-08-14) - PHẢI khớp giá trị "
                              "đã dùng lúc fine-tune/đo Recall gốc của model này, nếu không kết quả "
                              "sweep sẽ không nhất quán với số liệu đã có.")
    parser.add_argument("--fp16-inference", dest="fp16_inference", action="store_true", default=True)
    parser.add_argument("--no-fp16-inference", dest="fp16_inference", action="store_false")
    parser.add_argument("--output-dir", default="outputs/predictions/sweep_bm25_weight",
                         help="Lưu dự đoán hybrid_fused riêng cho từng weight (để soi lại sau nếu cần) - đặt rỗng \"\" để tắt.")
    parser.add_argument("--report", default=None, help="Nếu chỉ định, lưu bảng so sánh đầy đủ (JSON) ra đường dẫn này")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[lỗi] Thiếu sentence-transformers. Chạy: pip install sentence-transformers torch", file=sys.stderr)
        sys.exit(1)
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        print("[lỗi] Thiếu rank_bm25. Chạy: pip install rank_bm25", file=sys.stderr)
        sys.exit(1)
    import numpy as np

    weights = parse_weights(args.weights)
    if 1.0 not in weights:
        print("[cảnh báo] --weights không có 1.0 (baseline RRF gốc) - thêm 1.0 để so sánh có ý nghĩa.", file=sys.stderr)

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
            print("Đã cast embedding model sang fp16 (dùng --no-fp16-inference để tắt).")
        except Exception as e:  # noqa: BLE001
            print(f"[cảnh báo] Không cast được model sang fp16 ({e}) - encode ở fp32.", file=sys.stderr)

    chunk_embeddings = encode_chunks(
        embedding_model, chunks, args.batch_size, args.embedding_cache, passage_prefix=args.passage_prefix
    )
    bm25_index = build_bm25_index(chunks)

    # ---- Bước TỐN THỜI GIAN NHẤT: tính dense/BM25 score + top-N rank cho từng câu hỏi, 1 LẦN DUY
    # NHẤT (không phụ thuộc --bm25-weight) ----
    print(f"Tính điểm dense + BM25 cho {len(questions)} câu hỏi (1 lần, dùng lại cho mọi weight)...")
    per_question_ranks: List[Tuple[str, "np.ndarray", "np.ndarray"]] = []
    for start in range(0, len(questions), args.question_batch_size):
        batch = questions[start : start + args.question_batch_size]
        batch_texts = [args.query_prefix + q["question"] for q in batch]
        batch_dense_emb = embedding_model.encode(
            batch_texts, batch_size=args.question_batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)
        dense_sims_batch = batch_dense_emb @ chunk_embeddings.T

        for i, q in enumerate(batch):
            dense_scores = dense_sims_batch[i]
            q_tokens = q["question_segmented"].split() if q.get("question_segmented") else _naive_tokenize(q["question"])
            bm25_scores = np.asarray(bm25_index.get_scores(q_tokens))

            dense_top_idx = top_n_indices(dense_scores, args.dense_top_n)
            bm25_top_idx = top_n_indices(bm25_scores, args.bm25_top_n)
            per_question_ranks.append((q["question_id"], dense_top_idx, bm25_top_idx))
        print(f"  đã xử lý {min(start + args.question_batch_size, len(questions))}/{len(questions)} câu hỏi")

    # ---- Bước RẺ: thử từng --bm25-weight, chỉ tổ hợp lại RRF từ rank đã tính sẵn ----
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for weight in weights:
        predictions: Dict[str, List[str]] = {}
        for qid, dense_top_idx, bm25_top_idx in per_question_ranks:
            rrf_score = fuse_from_ranks(dense_top_idx, bm25_top_idx, args.rrf_k, weight)
            candidates = aggregate_to_documents(rrf_score, doc_ids, args.top_k)
            predictions[qid] = [doc_id for doc_id, _, _ in candidates]

        result = evaluate_predictions(ground_truth, predictions)
        results[weight] = result

        if output_dir:
            weight_path = output_dir / f"bm25_weight_{weight}.json"
            with open(weight_path, "w", encoding="utf-8") as f:
                json.dump({qid: {"answer": docs} for qid, docs in predictions.items()}, f, ensure_ascii=False, indent=2)

    # ---- In bảng so sánh ----
    print("\n" + "=" * 66)
    print(f"{'bm25_weight':>12s} {'Recall':>10s} {'Precision':>11s} {'Δ Recall vs 1.0':>18s}")
    print("-" * 66)
    baseline_recall = results.get(1.0, {}).get("avg_recall")
    best_weight = max(results, key=lambda w: results[w]["avg_recall"])
    for weight in sorted(results):
        r = results[weight]
        delta_str = ""
        if baseline_recall is not None:
            delta = r["avg_recall"] - baseline_recall
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.4f}"
        marker = "  <-- TỐT NHẤT" if weight == best_weight else ("  <-- baseline (RRF gốc)" if weight == 1.0 else "")
        print(f"{weight:>12g} {r['avg_recall']:>10.4f} {r['avg_precision']:>11.4f} {delta_str:>18s}{marker}")
    print("=" * 66)

    if baseline_recall is not None and results[best_weight]["avg_recall"] > baseline_recall + 0.005:
        print(
            f"\n-> weight={best_weight:g} cải thiện Recall +{results[best_weight]['avg_recall'] - baseline_recall:.4f} "
            "so với RRF gốc (weight=1.0) trên val_split_20 - đáng thử sweep tinh hơn quanh giá trị này, "
            "rồi xác nhận lại trên leaderboard công khai trước khi chốt (đúng quy trình đã áp dụng khi bỏ reranker)."
        )
    elif baseline_recall is not None and results[best_weight]["avg_recall"] <= baseline_recall + 0.005:
        print(
            "\n-> Không có weight nào cải thiện Recall rõ rệt (>+0.005) so với RRF gốc trên val_split_20 - "
            "giả thuyết \"tăng BM25 sửa Pattern A\" có vẻ KHÔNG mạnh như kỳ vọng trên tập val này. "
            "Cân nhắc chuyển sang hướng so sánh embedding model khác (AITeamVN/Vietnamese_Embedding, "
            "CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3) thay vì tiếp tục dò weight."
        )

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            str(weight): {"avg_recall": r["avg_recall"], "avg_precision": r["avg_precision"]}
            for weight, r in results.items()
        }
        report_data["_best_weight"] = best_weight
        report_data["_baseline_weight"] = 1.0
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu báo cáo so sánh: {args.report}")


if __name__ == "__main__":
    main()
