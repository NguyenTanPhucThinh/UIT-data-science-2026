"""
Sweep (dò lưới) 3 tham số RRF fusion CÙNG LÚC trên val_split_20: --dense-top-n, --bm25-top-n,
--rrf-k. Giữ NGUYÊN --bm25-weight=0.1 (đỉnh đã tìm được riêng, xem sweep_bm25_weight_aiteamvn.ps1 /
dòng #11b trong claude/submission-log-task1.md) - KHÔNG sweep lại đồng thời cả 4 tham số (không gian
tổ hợp quá lớn, và bm25-weight đã có bằng chứng riêng khá chắc là đỉnh mượt/đơn điệu).

ĐỘNG LỰC (2026-08-17): soát lại log dự án phát hiện dòng đầu file `submission-log-task1.md` ghi
"tune RRF hyperparameter (dense-top-n/bm25-top-n/rrf-k) đã ĐÓNG", nhưng thực ra chỉ có 1 lần nhắc
thoáng qua "+0.0004" KHÔNG có bảng sweep/dòng log riêng - kiểm tra code (`full_pipeline_eval.py`)
xác nhận 3 tham số này giữ NGUYÊN giá trị mặc định (`dense_top_n=100`, `bm25_top_n=100`, `rrf_k=60`)
xuyên suốt MỌI lần chạy kể từ dòng #4 - tức CHƯA thật sự sweep. Script này lấp khoảng trống đó.

Ý NGHĨA của 3 tham số (xem docstring `rrf_fuse()` trong full_pipeline_eval.py):
  - --dense-top-n / --bm25-top-n: số chunk TOP theo từng nhánh (dense/BM25 riêng) được đưa vào bước
    fusion RRF. Tăng lên => nhánh đó có nhiều ứng viên hơn để đóng góp điểm, nhưng cũng đưa vào
    nhiều ứng viên yếu hơn (rank thấp, đóng góp RRF score rất nhỏ do công thức 1/(rrf_k+rank)).
  - --rrf-k: hằng số "giảm xóc" trong công thức RRF (score += 1/(rrf_k+rank)). rrf_k NHỎ => chênh
    lệch điểm giữa rank 1 và rank 20 LỚN hơn (ưu tiên mạnh các rank cao nhất). rrf_k LỚN => các rank
    khác nhau đóng góp điểm gần bằng nhau hơn (dàn đều, giảm ảnh hưởng của rank tuyệt đối).

MẸO TỐC ĐỘ (giống sweep_bm25_weight.py): dense_scores/bm25_scores ĐẦY ĐỦ (toàn bộ 351K chunk) của
MỖI câu hỏi chỉ cần tính 1 LẦN DUY NHẤT (không phụ thuộc bất kỳ tham số sweep nào). Với TỪNG câu hỏi,
tính sẵn top-N_MAX chỉ số đã SẮP XẾP theo rank (N_MAX = giá trị lớn nhất trong lưới --dense-top-n-grid
/ --bm25-top-n-grid) cho cả 2 nhánh - sau đó với MỌI tổ hợp (dense_top_n, bm25_top_n, rrf_k) trong
lưới, chỉ cần CẮT (slice) mảng đã sắp xếp sẵn này (rất rẻ) rồi tổ hợp lại RRF, không cần tính lại
dense/BM25 score hay sắp xếp lại. Dùng `--bm25-fast` (mặc định BẬT, verify đúng+an toàn từ
2026-08-17) để tính BM25 CẢ BATCH câu hỏi bằng 1 phép nhân ma trận thưa duy nhất.

Cách dùng (mặc định đã trỏ đúng AITeamVN/Vietnamese_Embedding v1 + cache có sẵn, KHÔNG cần encode
lại corpus):
    python -m src.evaluation.sweep_rrf_hyperparams \\
        --dense-top-n-grid 50,100,200 --bm25-top-n-grid 50,100,200 --rrf-k-grid 10,30,60,100 \\
        --report outputs/predictions/sweep_rrf_hyperparams_report.json

Sau khi có bảng kết quả:
  - Nếu tổ hợp tốt nhất TRÙNG với mặc định hiện tại (100,100,60) hoặc chênh lệch < +0.005 Recall so
    với mặc định -> xác nhận nhãn "đã đóng" trong log là ĐÚNG trên thực tế (dù trước đó thiếu bằng
    chứng chi tiết) - không cần đổi gì, không cần sweep tinh hơn.
  - Nếu tổ hợp tốt nhất cải thiện RÕ RỆT (>= +0.005, cùng ngưỡng đã dùng cho sweep bm25-weight) ->
    đáng sweep tinh hơn quanh tổ hợp đó, rồi xác nhận lại trên public test trước khi đổi cấu hình
    chính thức (đúng quy trình đã áp dụng nhất quán trong dự án).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from .evaluate import load_ground_truth
    from .metrics import evaluate_predictions
    from .full_pipeline_eval import (
        load_chunks, load_questions, resolve_device, encode_chunks, build_bm25_index, _naive_tokenize,
        aggregate_to_documents, build_bm25_fast_matrix, bm25_scores_batch_fast,
    )
except ImportError:  # cho phép chạy trực tiếp
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.metrics import evaluate_predictions
    from src.evaluation.full_pipeline_eval import (
        load_chunks, load_questions, resolve_device, encode_chunks, build_bm25_index, _naive_tokenize,
        aggregate_to_documents, build_bm25_fast_matrix, bm25_scores_batch_fast,
    )


def top_n_indices(scores, top_n: int):
    """Trả về chỉ số top-N (giảm dần theo điểm), ĐÃ SẮP XẾP theo rank - tính 1 lần với N_MAX, sau
    đó cắt (slice) ra top-N nhỏ hơn bất kỳ lúc nào mà KHÔNG cần sắp xếp lại."""
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


def parse_int_list(raw: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Danh sách rỗng hoặc không hợp lệ: {raw!r}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-model", default="AITeamVN/Vietnamese_Embedding",
                         help="Mặc định = v1 pretrained, cấu hình CHÍNH THỨC hiện tại (dòng #11b)")
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    parser.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    parser.add_argument("--embedding-cache", default="outputs/cache/embcompare_aiteamvn.npz",
                         help="Mặc định tái dùng cache đã có sẵn của AITeamVN v1 - KHÔNG cần encode lại corpus")
    parser.add_argument("--dense-top-n-grid", default="50,100,200",
                         help="Danh sách giá trị --dense-top-n để dò, cách nhau dấu phẩy. Mặc định hiện tại của pipeline là 100 - LUÔN nên có mặt để so sánh.")
    parser.add_argument("--bm25-top-n-grid", default="50,100,200",
                         help="Danh sách giá trị --bm25-top-n để dò. Mặc định hiện tại là 100 - LUÔN nên có mặt.")
    parser.add_argument("--rrf-k-grid", default="10,30,60,100",
                         help="Danh sách giá trị --rrf-k để dò. Mặc định hiện tại là 60 - LUÔN nên có mặt.")
    parser.add_argument("--bm25-weight", type=float, default=0.1,
                         help="GIỮ NGUYÊN đỉnh đã tìm được (dòng #11b) - không sweep lại trong script này.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size encode corpus (dense) - chỉ dùng nếu cache chưa có")
    parser.add_argument("--question-batch-size", type=int, default=64)
    parser.add_argument("--limit-questions", type=int, default=None, help="Chỉ chạy N câu hỏi đầu (debug nhanh)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--passage-prefix", default="")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--fp16-inference", dest="fp16_inference", action="store_true", default=True)
    parser.add_argument("--no-fp16-inference", dest="fp16_inference", action="store_false")
    parser.add_argument("--top-results", type=int, default=15, help="Số tổ hợp tốt nhất in ra bảng cuối (đầy đủ vẫn lưu trong --report)")
    parser.add_argument("--report", default=None, help="Nếu chỉ định, lưu bảng so sánh ĐẦY ĐỦ (JSON, mọi tổ hợp) ra đường dẫn này")
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
    try:
        import scipy, pandas  # noqa: F401
    except ImportError:
        print("[lỗi] --bm25-fast (luôn bật trong script này) cần scipy + pandas. Chạy: pip install scipy pandas", file=sys.stderr)
        sys.exit(1)
    import numpy as np

    dense_top_n_grid = parse_int_list(args.dense_top_n_grid)
    bm25_top_n_grid = parse_int_list(args.bm25_top_n_grid)
    rrf_k_grid = parse_int_list(args.rrf_k_grid)
    dense_n_max = max(dense_top_n_grid)
    bm25_n_max = max(bm25_top_n_grid)

    n_combos = len(dense_top_n_grid) * len(bm25_top_n_grid) * len(rrf_k_grid)
    print(f"Lưới sweep: {len(dense_top_n_grid)} dense-top-n x {len(bm25_top_n_grid)} bm25-top-n x "
          f"{len(rrf_k_grid)} rrf-k = {n_combos} tổ hợp (bm25-weight giữ cố định = {args.bm25_weight}).")

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
    print("Xây ma trận BM25 vector hoá (--bm25-fast, 1 lần duy nhất)...")
    bm25_fast = build_bm25_fast_matrix(bm25_index)

    # ---- Bước TỐN THỜI GIAN NHẤT: tính dense/BM25 score đầy đủ + top-N_MAX rank cho từng câu hỏi,
    # 1 LẦN DUY NHẤT (không phụ thuộc tổ hợp nào trong lưới sweep) ----
    print(f"Tính điểm dense + BM25 cho {len(questions)} câu hỏi (1 lần, dùng lại cho mọi tổ hợp)...")
    per_question_ranks: List[Tuple[str, "np.ndarray", "np.ndarray"]] = []
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
            dense_top_idx = top_n_indices(dense_sims_batch[i], dense_n_max)
            bm25_top_idx = top_n_indices(bm25_scores_batch[i], bm25_n_max)
            per_question_ranks.append((q["question_id"], dense_top_idx, bm25_top_idx))
        print(f"  đã xử lý {min(start + args.question_batch_size, len(questions))}/{len(questions)} câu hỏi")

    # ---- Bước RẺ: thử từng tổ hợp (dense_top_n, bm25_top_n, rrf_k), chỉ CẮT mảng rank đã tính sẵn
    # rồi tổ hợp lại RRF (không tính lại dense/BM25 score) ----
    print(f"\nDò {n_combos} tổ hợp...")
    results = {}
    for dense_top_n, bm25_top_n, rrf_k in product(dense_top_n_grid, bm25_top_n_grid, rrf_k_grid):
        predictions: Dict[str, List[str]] = {}
        for qid, dense_top_idx_full, bm25_top_idx_full in per_question_ranks:
            dense_top_idx = dense_top_idx_full[:dense_top_n]
            bm25_top_idx = bm25_top_idx_full[:bm25_top_n]
            rrf_score = fuse_from_ranks(dense_top_idx, bm25_top_idx, rrf_k, args.bm25_weight)
            candidates = aggregate_to_documents(rrf_score, doc_ids, args.top_k)
            predictions[qid] = [doc_id for doc_id, _, _ in candidates]

        result = evaluate_predictions(ground_truth, predictions)
        results[(dense_top_n, bm25_top_n, rrf_k)] = result

    # ---- In bảng so sánh (top N tốt nhất theo Recall) ----
    baseline_key = (100, 100, 60)
    baseline = results.get(baseline_key)
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["avg_recall"])

    print("\n" + "=" * 78)
    print(f"{'dense_top_n':>12s} {'bm25_top_n':>11s} {'rrf_k':>6s} {'Recall':>9s} {'Precision':>10s} {'Δ vs mặc định':>15s}")
    print("-" * 78)
    for (dense_top_n, bm25_top_n, rrf_k), r in ranked[: args.top_results]:
        delta_str = ""
        if baseline is not None:
            delta = r["avg_recall"] - baseline["avg_recall"]
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.4f}"
        marker = "  <-- MẶC ĐỊNH HIỆN TẠI" if (dense_top_n, bm25_top_n, rrf_k) == baseline_key else ""
        print(f"{dense_top_n:>12d} {bm25_top_n:>11d} {rrf_k:>6d} {r['avg_recall']:>9.4f} {r['avg_precision']:>10.4f} {delta_str:>15s}{marker}")
    if baseline_key not in {k for k, _ in ranked[: args.top_results]} and baseline is not None:
        print("-" * 78)
        print(f"{100:>12d} {100:>11d} {60:>6d} {baseline['avg_recall']:>9.4f} {baseline['avg_precision']:>10.4f} "
              f"{'0.0000':>15s}  <-- MẶC ĐỊNH HIỆN TẠI (ngoài top {args.top_results})")
    print("=" * 78)

    best_key, best_result = ranked[0]
    if baseline is not None:
        delta = best_result["avg_recall"] - baseline["avg_recall"]
        if delta >= 0.005:
            print(f"\n-> Tổ hợp tốt nhất {best_key} cải thiện Recall +{delta:.4f} so với mặc định (100,100,60) - "
                  "ĐÁNG sweep tinh hơn quanh tổ hợp này, rồi xác nhận lại trên public test trước khi đổi "
                  "cấu hình chính thức (đúng quy trình đã áp dụng cho bm25-weight).")
        else:
            print(f"\n-> Tổ hợp tốt nhất {best_key} chỉ chênh +{delta:.4f} so với mặc định (100,100,60) - "
                  "DƯỚI ngưỡng +0.005 coi là cải thiện đáng kể. Xác nhận: mặc định hiện tại ĐÃ gần tối ưu "
                  "cho 3 tham số này, nhãn \"đã đóng\" trong log giờ có bằng chứng chi tiết thật sự.")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            f"dense{dense_top_n}_bm25{bm25_top_n}_k{rrf_k}": {
                "dense_top_n": dense_top_n, "bm25_top_n": bm25_top_n, "rrf_k": rrf_k,
                "avg_recall": r["avg_recall"], "avg_precision": r["avg_precision"],
            }
            for (dense_top_n, bm25_top_n, rrf_k), r in results.items()
        }
        report_data["_best"] = {"dense_top_n": best_key[0], "bm25_top_n": best_key[1], "rrf_k": best_key[2],
                                 "avg_recall": best_result["avg_recall"]}
        report_data["_baseline"] = {"dense_top_n": 100, "bm25_top_n": 100, "rrf_k": 60,
                                     "avg_recall": baseline["avg_recall"] if baseline else None}
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu báo cáo so sánh ĐẦY ĐỦ ({len(results)} tổ hợp): {args.report}")


if __name__ == "__main__":
    main()
