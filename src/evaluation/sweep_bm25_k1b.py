"""
SWEEP k1/b của BM25Okapi — hướng #3 (cuối) trong 3 hướng đã chốt thứ tự
(fusion theo score -> DEk21 -> sweep BM25 k1/b), 2026-08-28.

VÌ SAO HƯỚNG NÀY, VÀ VÌ SAO KHÁC HẲN `bm25-weight` (đã sweep, +0.0115, dòng #? trong
submission-log-task1.md): `bm25-weight` chỉnh TRỌNG SỐ ĐÓNG GÓP của cả nhánh BM25 vào RRF tổng —
đã sweep rồi, đã dùng. `k1`/`b` KHÔNG đụng tới trọng số đó — chúng chỉnh CHẤT LƯỢNG XẾP HẠNG NỘI BỘ
của BM25Okapi (k1: độ bão hoà tần suất từ; b: mức phạt theo độ dài tài liệu), thứ mà pipeline này
CHƯA TỪNG sweep — luôn dùng nguyên giá trị mặc định của thư viện rank_bm25 (k1=1.5, b=0.75), vốn
được hiệu chỉnh cho corpus tiếng Anh kiểu TREC, không có gì đảm bảo tối ưu cho corpus luật tiếng Việt
đã tách từ (passage_segmented) hiện tại.

⚠️ RỦI RO PHA LOÃNG (đã biết TRƯỚC khi chạy, không phải suy đoán): dù nội bộ BM25 xếp hạng tốt hơn
bao nhiêu, khi fuse vào RRF nó vẫn chỉ đóng góp qua đúng `bm25-weight=0.1` hiện tại — cải thiện xếp
hạng nội bộ NHỎ vẫn có thể bị "hoà loãng" tới mức không đổi được top-5 cuối cùng. Đây là lý do hướng
này được xếp ƯU TIÊN THẤP NHẤT trong 3 hướng (dev rẻ nhất, nhưng trần cải thiện kỳ vọng cũng thấp
nhất — xem bảng so sánh 3 hướng đã trình bày cho Hoàng 2026-08-28).

CƠ CHẾ TÍNH RẺ (tránh xây lại ma trận ~4GB RAM mỗi lần đổi k1/b): trong công thức BM25Okapi,
    idf(từ), doc_freqs, doc_len, avgdl  ĐỘC LẬP HOÀN TOÀN với k1/b (xem rank_bm25.BM25Okapi._calc_idf
    — chỉ dùng self.corpus_size/nd, KHÔNG bao giờ đọc self.k1/self.b). CHỈ phần
        K(doc) = k1 * (1 - b + b * doc_len / avgdl)
        transformed(doc, từ) = tf * (k1+1) / (tf + K[doc])
    phụ thuộc k1/b. Nên: build `bm25_index` (BM25Okapi) + ma trận tần suất thô (tf_matrix, CHƯA nhân
    k1/b) ĐÚNG 1 LẦN (`build-raw`, bước tốn RAM/thời gian như build_bm25_fast_matrix() gốc trong
    full_pipeline_eval.py) — MỌI combo (k1,b) sau đó chỉ cần tính lại K (1 vector, rẻ) rồi transform
    LẠI .data của ma trận thưa CÓ SẴN (KHÔNG dựng lại rows/cols từ đầu, bước ĐẮT NHẤT — đã đo ~7s ở
    quy mô 50 triệu phần tử cho riêng bước dựng chỉ số cột bằng pandas). `apply_k1_b()` bên dưới tái
    dùng NGUYÊN VĂN công thức của build_bm25_fast_matrix() (full_pipeline_eval.py dòng ~373-412),
    chỉ tách phần phụ thuộc k1/b ra khỏi phần dựng ma trận.

⚠️ CẢNH BÁO RAM (kế thừa từ build_bm25_fast_matrix() gốc): bước `build-raw` ước tính đỉnh ~4GB RAM
    cho corpus 351.394 chunk / ~31,2 triệu phần tử khác-0 (xem log lúc chạy để biết số chính xác).
    Nếu máy ít RAM trống, đóng bớt ứng dụng khác trước khi chạy.

ĐÃ TEST TRƯỚC KHI GIAO (`selftest`, không cần corpus thật/GPU — dùng corpus tổng hợp nhỏ):
  1. apply_k1_b() với ĐÚNG k1=1.5/b=0.75 mặc định phải cho điểm KHỚP TUYỆT ĐỐI với
     rank_bm25.BM25Okapi.get_scores() gốc (đối chiếu trực tiếp, không phải suy luận công thức đúng).
  2. apply_k1_b() với k1/b KHÁC mặc định (vd k1=1.2, b=0.5) phải khớp với 1 BM25Okapi THỨ HAI dựng
     lại từ đầu với đúng k1/b đó và gọi get_scores() — xác nhận việc "chỉ tính lại K rồi transform"
     tương đương TOÁN HỌC với việc xây lại toàn bộ index bằng k1/b mới, không chỉ đúng ở giá trị
     mặc định.
  3. build_raw_tf_matrix() + apply_k1_b() + bm25_scores_batch_fast() (hàm dùng LẠI, KHÔNG viết mới,
     từ full_pipeline_eval.py) cho kết quả khớp get_scores() ở CẢ 2 case trên, xác nhận toàn bộ
     đường ống vector hoá (không chỉ hàm apply_k1_b riêng lẻ) đúng.

CÁCH DÙNG (chạy từ thư mục GỐC repo):
  python -m src.evaluation.sweep_bm25_k1b selftest              # vài giây, không cần corpus thật
  python -m src.evaluation.sweep_bm25_k1b build-raw              # ~như --bm25-fast gốc, 1 lần
  python -m src.evaluation.sweep_bm25_k1b sweep                  # tức thì mỗi combo, không GPU
  python -m src.evaluation.sweep_bm25_k1b export --k1 .. --b ..  # xuất .npz cho combo tốt nhất,
                                                                   # dùng làm --rank bm25tuned=... cho
                                                                   # ensemble_fusion.py fuse (weight=0.1)
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

try:
    from .evaluate import load_ground_truth
    from .metrics import evaluate_predictions
    from .full_pipeline_eval import load_chunks, load_questions, build_bm25_index, _naive_tokenize, bm25_scores_batch_fast
    from .ensemble_fusion import top_n_indices, aggregate_max, save_ranks
except ImportError:  # cho phép chạy trực tiếp
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.metrics import evaluate_predictions
    from src.evaluation.full_pipeline_eval import (
        load_chunks, load_questions, build_bm25_index, _naive_tokenize, bm25_scores_batch_fast,
    )
    from src.evaluation.ensemble_fusion import top_n_indices, aggregate_max, save_ranks


# ---------------------------------------------------------------------------
# Phần KHÔNG phụ thuộc k1/b — build 1 LẦN
# ---------------------------------------------------------------------------

def build_raw_tf_matrix(bm25_index) -> dict:
    """COPY Y HỆT phần dựng ma trận thưa (rows/cols/data) của build_bm25_fast_matrix() trong
    full_pipeline_eval.py, NHƯNG DỪNG LẠI ở tf_matrix THÔ — chưa nhân k1/(k1+1)/K. Phần phụ thuộc
    k1/b được tách riêng ra apply_k1_b() để không phải lặp lại bước flatten (đắt nhất) mỗi combo."""
    import numpy as np
    import scipy.sparse as sp
    try:
        import pandas as pd
    except ImportError as e:  # noqa: BLE001
        raise ImportError("Cần thêm pandas (map từ vựng -> chỉ số cột nhanh): pip install pandas") from e

    vocab_index = {term: i for i, term in enumerate(bm25_index.idf.keys())}
    idf = np.asarray([bm25_index.idf[t] for t in vocab_index], dtype=np.float32)
    n_docs = len(bm25_index.doc_freqs)
    doc_len = np.asarray(bm25_index.doc_len, dtype=np.float64)
    avgdl = float(bm25_index.avgdl)
    doc_freqs = bm25_index.doc_freqs
    nnz = sum(len(freqs) for freqs in doc_freqs)

    print(f"  Dựng ma trận tần suất thô: {n_docs} document x {len(vocab_index)} từ vựng, {nnz} phần tử khác-0...")
    doc_lens_arr = np.fromiter((len(f) for f in doc_freqs), dtype=np.int64, count=n_docs)
    rows = np.repeat(np.arange(n_docs, dtype=np.int32), doc_lens_arr)
    terms_flat = list(itertools.chain.from_iterable(f.keys() for f in doc_freqs))
    data = np.fromiter(itertools.chain.from_iterable(f.values() for f in doc_freqs), dtype=np.float32, count=nnz)
    cols = pd.Index(list(vocab_index.keys())).get_indexer(terms_flat).astype(np.int32)
    del terms_flat

    if (cols < 0).any():
        raise RuntimeError(
            "Có từ trong doc_freqs KHÔNG map được vào vocab_index - không nên xảy ra (vocab lấy "
            "trực tiếp từ chính doc_freqs). Báo lại lỗi này kèm log đầy đủ nếu gặp phải."
        )

    tf_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_docs, len(vocab_index)), dtype=np.float32)
    return {
        "tf_matrix": tf_matrix, "vocab_index": vocab_index, "idf": idf,
        "doc_len": doc_len, "avgdl": avgdl, "n_docs": n_docs,
    }


# ---------------------------------------------------------------------------
# Phần PHỤ THUỘC k1/b — tính lại RẺ cho mỗi combo
# ---------------------------------------------------------------------------

def apply_k1_b(raw: dict, k1: float, b: float) -> dict:
    """Trả về dict {"matrix", "vocab_index", "idf"} CÙNG FORMAT với build_bm25_fast_matrix() gốc —
    dùng trực tiếp được với bm25_scores_batch_fast() KHÔNG cần sửa gì. Đúng công thức BM25Okapi
    (xem docstring module): K phụ thuộc doc_len/avgdl/k1/b, transform = tf*(k1+1)/(tf+K)."""
    import numpy as np
    import scipy.sparse as sp

    tf_matrix = raw["tf_matrix"]
    doc_len = raw["doc_len"]
    avgdl = raw["avgdl"]
    K = (k1 * (1.0 - b + b * doc_len / avgdl)).astype(np.float32)
    row_idx_per_nnz = np.repeat(np.arange(raw["n_docs"], dtype=np.int32), np.diff(tf_matrix.indptr))
    transformed = tf_matrix.data * (k1 + 1.0) / (tf_matrix.data + K[row_idx_per_nnz])
    bm25_term_matrix = sp.csr_matrix((transformed, tf_matrix.indices, tf_matrix.indptr), shape=tf_matrix.shape)
    return {"matrix": bm25_term_matrix, "vocab_index": raw["vocab_index"], "idf": raw["idf"]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_build_raw(args) -> None:
    chunks = load_chunks(args.chunks)
    print(f"Build BM25Okapi index trên {len(chunks)} chunk (k1/b mặc định của thư viện, KHÔNG quan "
          f"trọng vì raw tf_matrix không phụ thuộc k1/b — chỉ idf/doc_freqs được giữ lại)...")
    bm25_index = build_bm25_index(chunks)
    raw = build_raw_tf_matrix(bm25_index)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(raw, f, protocol=4)
    print(f"Đã lưu raw tf_matrix (không phụ thuộc k1/b): {out}  "
          f"({raw['n_docs']} document, {len(raw['vocab_index'])} từ vựng, avgdl={raw['avgdl']:.2f})")
    print("Bước tiếp: python -m src.evaluation.sweep_bm25_k1b sweep")


def _standalone_recall(raw: dict, k1: float, b: float, questions: List[dict], doc_ids: List[str],
                       ground_truth: dict, top_k: int, question_batch_size: int) -> float:
    import numpy as np

    bm25_fast = apply_k1_b(raw, k1, b)
    predictions: Dict[str, List[str]] = {}
    for start in range(0, len(questions), question_batch_size):
        batch = questions[start : start + question_batch_size]
        q_tokens_batch = [
            q["question_segmented"].split() if q.get("question_segmented") else _naive_tokenize(q["question"])
            for q in batch
        ]
        scores_batch = bm25_scores_batch_fast(bm25_fast, q_tokens_batch)
        for i, q in enumerate(batch):
            idx = top_n_indices(scores_batch[i], top_k * 20)  # đủ rộng để aggregate_max gộp chunk->doc
            rrf_like = {int(j): float(scores_batch[i][j]) for j in idx}
            predictions[str(q["question_id"])] = aggregate_max(rrf_like, doc_ids, top_k)
    r = evaluate_predictions(ground_truth, predictions)
    return r["avg_recall"]


def cmd_sweep(args) -> None:
    with open(args.raw, "rb") as f:
        raw = pickle.load(f)
    chunks = load_chunks(args.chunks)
    doc_ids = [c["doc_id"] for c in chunks]
    questions = load_questions(args.questions)
    if args.limit_questions:
        questions = questions[: args.limit_questions]
    ground_truth = load_ground_truth(args.ground_truth)

    k1_grid = [float(x) for x in args.k1.split(",") if x.strip() != ""]
    b_grid = [float(x) for x in args.b.split(",") if x.strip() != ""]
    combos = list(itertools.product(k1_grid, b_grid))
    print(f"Sweep {len(combos)} tổ hợp (k1 x b) trên {len(questions)} câu hỏi, top-k={args.top_k}\n")
    print(f"{'k1':>6s} {'b':>6s} {'Recall':>9s}")
    print("-" * 26)

    results = []
    for k1, b in combos:
        recall = _standalone_recall(raw, k1, b, questions, doc_ids, ground_truth, args.top_k, args.question_batch_size)
        mark = "  <-- mặc định rank_bm25" if (abs(k1 - 1.5) < 1e-9 and abs(b - 0.75) < 1e-9) else ""
        print(f"{k1:>6.2f} {b:>6.2f} {recall:>9.4f}{mark}")
        results.append({"k1": k1, "b": b, "avg_recall": recall})

    results_sorted = sorted(results, key=lambda r: -r["avg_recall"])
    default_row = next((r for r in results if abs(r["k1"] - 1.5) < 1e-9 and abs(r["b"] - 0.75) < 1e-9), None)
    best = results_sorted[0]
    print("\n" + "=" * 60)
    print(f"Tốt nhất : k1={best['k1']:g} b={best['b']:g}  Recall={best['avg_recall']:.4f} (BM25 ĐỨNG RIÊNG)")
    if default_row is not None:
        delta = best["avg_recall"] - default_row["avg_recall"]
        print(f"Mặc định : k1=1.5 b=0.75            Recall={default_row['avg_recall']:.4f}")
        print(f"Δ (BM25 đứng riêng, KHÔNG phải ensemble) = {delta:+.4f}")
        print("\n⚠️ NHỚ: đây là Recall của BM25 ĐỨNG RIÊNG, không phải ensemble cuối cùng. Δ này sẽ bị")
        print("   PHA LOÃNG khi fuse với bm25-weight=0.1 hiện có. Bước tiếp để đo Δ THẬT trên ensemble:")
        print(f"   python -m src.evaluation.sweep_bm25_k1b export --raw {args.raw} --k1 {best['k1']:g} --b {best['b']:g} \\")
        print("       --out outputs/cache/ranks_bm25_tuned.npz")
        print("   rồi dùng file đó thay ranks_bm25.npz trong scripts/run_ensemble_fusion.ps1 (hoặc gọi")
        print("   thẳng ensemble_fusion.py fuse --rank bm25=outputs/cache/ranks_bm25_tuned.npz ...) để")
        print("   so Recall ENSEMBLE cuối cùng với baseline 0.9319.")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"best": best, "default": default_row, "all": results_sorted}, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu báo cáo: {args.report}")


def cmd_export(args) -> None:
    """Xuất .npz tương thích ensemble_fusion.save_ranks() cho 1 combo (k1,b) cụ thể — để cắm thẳng
    vào `ensemble_fusion.py fuse --rank bm25=...` thay cho ranks_bm25.npz mặc định (k1=1.5,b=0.75),
    đo Δ THẬT trên ensemble đầy đủ ở đúng bm25-weight=0.1 hiện có."""
    with open(args.raw, "rb") as f:
        raw = pickle.load(f)
    chunks = load_chunks(args.chunks)
    questions = load_questions(args.questions)
    if args.limit_questions:
        questions = questions[: args.limit_questions]

    import numpy as np

    bm25_fast = apply_k1_b(raw, args.k1, args.b)
    all_ranks = np.zeros((len(questions), args.top_n), dtype=np.int32)
    all_scores = np.zeros((len(questions), args.top_n), dtype=np.float32)
    qids: List[str] = []
    for start in range(0, len(questions), args.question_batch_size):
        batch = questions[start : start + args.question_batch_size]
        q_tokens_batch = [
            q["question_segmented"].split() if q.get("question_segmented") else _naive_tokenize(q["question"])
            for q in batch
        ]
        scores_batch = bm25_scores_batch_fast(bm25_fast, q_tokens_batch)
        for i, q in enumerate(batch):
            idx = top_n_indices(scores_batch[i], args.top_n)
            all_ranks[start + i] = idx
            all_scores[start + i] = scores_batch[i][idx]
            qids.append(str(q["question_id"]))
        print(f"  đã xử lý {min(start + args.question_batch_size, len(questions))}/{len(questions)} câu hỏi")

    save_ranks(args.out, qids, all_ranks, scores=all_scores)
    print(f"k1={args.k1} b={args.b} -> {args.out}")


# ---------------------------------------------------------------------------
# selftest — KHÔNG cần corpus thật, đối chiếu trực tiếp với rank_bm25.BM25Okapi.get_scores()
# ---------------------------------------------------------------------------

def cmd_selftest(args) -> None:
    import numpy as np
    from rank_bm25 import BM25Okapi

    n_ok = 0

    def check(name: str, cond: bool):
        nonlocal n_ok
        status = "OK" if cond else "HỎNG"
        print(f"  [{status}] {name}")
        if not cond:
            raise SystemExit(f"[lỗi] selftest thất bại tại: {name}")
        n_ok += 1

    corpus_tokens = [
        "luật lao động quy định thời gian làm việc".split(),
        "bảo hiểm y tế chi trả khám chữa bệnh".split(),
        "thời gian nghỉ phép năm theo luật lao động".split(),
        "quy định về bảo hiểm xã hội bắt buộc".split(),
        "hợp đồng lao động thời hạn và không thời hạn".split(),
    ]
    queries = [
        "luật lao động thời gian".split(),
        "bảo hiểm xã hội".split(),
        "hợp đồng không thời hạn".split(),
    ]

    for label, k1, b in [("MẶC ĐỊNH (k1=1.5, b=0.75)", 1.5, 0.75), ("KHÁC mặc định (k1=1.2, b=0.5)", 1.2, 0.5)]:
        print(f"\n=== {label} ===")
        ref_index = BM25Okapi(corpus_tokens, k1=k1, b=b)  # NGUỒN THAM CHIẾU — thư viện gốc, độc lập
        ref_scores = np.array([ref_index.get_scores(q) for q in queries])

        # đường ống của module này: build 1 lần bằng index MẶC ĐỊNH của thư viện (k1/b bất kỳ vì
        # không ảnh hưởng raw tf_matrix/idf), rồi apply_k1_b() với k1/b đang test
        raw_index = BM25Okapi(corpus_tokens)  # k1/b mặc định, KHÔNG liên quan tới k1/b đang test
        raw = build_raw_tf_matrix(raw_index)
        bm25_fast = apply_k1_b(raw, k1, b)
        our_scores = bm25_scores_batch_fast(bm25_fast, queries)

        check(f"idf/doc_len/avgdl không đổi bất kể k1/b lúc build raw (so idf 'luật')",
              abs(raw_index.idf.get("luật", 0) - ref_index.idf.get("luật", 0)) < 1e-9)
        check(f"scores khớp get_scores() gốc (đối chiếu TRỰC TIẾP, không suy luận công thức)",
              np.allclose(our_scores, ref_scores, atol=1e-4))

    print(f"\n=== ĐẠT {n_ok}/{n_ok} kiểm tra — apply_k1_b() ĐÚNG với CẢ mặc định lẫn k1/b tuỳ chỉnh ===")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_raw = sub.add_parser("build-raw", help="Build 1 LẦN: BM25Okapi index + ma trận tần suất thô (không phụ thuộc k1/b)")
    p_raw.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    p_raw.add_argument("--out", default="outputs/cache/bm25_raw_tf.pkl")
    p_raw.set_defaults(func=cmd_build_raw)

    p_sweep = sub.add_parser("sweep", help="Sweep lưới k1 x b, đo Recall BM25 ĐỨNG RIÊNG (tức thì, không GPU)")
    p_sweep.add_argument("--raw", default="outputs/cache/bm25_raw_tf.pkl")
    p_sweep.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    p_sweep.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    p_sweep.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    p_sweep.add_argument("--limit-questions", type=int, default=None)
    p_sweep.add_argument("--question-batch-size", type=int, default=64)
    p_sweep.add_argument("--k1", default="1.0,1.2,1.5,1.8,2.2,2.6")
    p_sweep.add_argument("--b", default="0.3,0.5,0.75,0.9")
    p_sweep.add_argument("--top-k", type=int, default=5)
    p_sweep.add_argument("--report", default=None)
    p_sweep.set_defaults(func=cmd_sweep)

    p_exp = sub.add_parser("export", help="Xuất .npz (compat ensemble_fusion) cho 1 combo (k1,b) cụ thể")
    p_exp.add_argument("--raw", default="outputs/cache/bm25_raw_tf.pkl")
    p_exp.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    p_exp.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    p_exp.add_argument("--limit-questions", type=int, default=None)
    p_exp.add_argument("--question-batch-size", type=int, default=64)
    p_exp.add_argument("--k1", type=float, required=True)
    p_exp.add_argument("--b", type=float, required=True)
    p_exp.add_argument("--top-n", type=int, default=200)
    p_exp.add_argument("--out", required=True)
    p_exp.set_defaults(func=cmd_export)

    p_self = sub.add_parser("selftest", help="Đối chiếu trực tiếp với rank_bm25.BM25Okapi.get_scores(), không cần corpus thật")
    p_self.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
