"""KIỂM CHỨNG GIẢ THUYẾT THIÊN LỆCH ĐỘ DÀI trong gộp chunk->document (2026-08-28).

BỐI CẢNH (từ diagnose_ceiling.py chạy trên dữ liệu thật):
    Trần pool = 99.00%, Recall@5 hiện tại = 0.9319, Recall@100 = 0.9890.
    85.7% câu sai (60/70) là LỖI XẾP HẠNG (tài liệu đúng CÓ trong pool, chỉ xếp hạng 6-100),
    trong đó 32 câu tài liệu đúng nằm ngay HẠNG 6-10 — sát nút.
    => Vấn đề KHÔNG phải thiếu tín hiệu truy hồi. Vấn đề là XẾP HẠNG trong top-10.

GIẢ THUYẾT CẦN KIỂM CHỨNG: cách gộp `max` đang THIÊN VỊ TÀI LIỆU DÀI.
    Điểm document = điểm chunk CAO NHẤT của nó. Corpus có 351.394 chunk / 8.532 document
    (~41 chunk/doc) nhưng phân bố CỰC KỲ lệch. Một văn bản 500 chunk có 500 "vé số" để
    tình cờ có 1 chunk điểm cao; văn bản 3 chunk chỉ có 3 vé. Dưới phép lấy max, kỳ vọng
    của giá trị lớn nhất TĂNG theo số mẫu NGAY CẢ KHI không liên quan gì hơn.

VÌ SAO ĐIỀU NÀY GIẢI THÍCH ĐƯỢC "MỌI HƯỚNG ĐỀU THẤT BẠI": thiên lệch này nằm ở tầng GỘP,
    SAU khi mọi tín hiệu đã fuse xong. Đổi tín hiệu (thêm nhánh model), đổi cách fuse
    (RRF -> CombSUM), đổi trọng số, chỉnh k1/b — TẤT CẢ đều bị kế thừa nguyên vẹn thiên lệch
    này ở bước cuối. Đó là lý do cấu trúc khiến ~15 thử nghiệm khác nhau đều hội tụ về ~0.93.

LƯU Ý sweep_aggregation.py TRƯỚC ĐÂY đã thử max/sum_top2/sum_top3/mean_top2 và max thắng —
    nhưng KHÔNG cái nào trong số đó KHỬ THIÊN LỆCH ĐỘ DÀI. sum_top2/sum_top3 làm thiên lệch
    NẶNG HƠN (càng nhiều chunk càng cộng được nhiều). Đây là phép thử KHÁC HẲN.

Script này làm 3 việc:
  1. ĐO trực tiếp: tài liệu đúng bị xếp sau có ngắn hơn tài liệu đã đè nó không?
  2. THỬ: gộp có khử thiên lệch  score = max_chunk_score - alpha*log(1+n_chunk)  , sweep alpha.
  3. So sánh GHÉP CẶP (McNemar) chứ không chỉ 2 con số Recall — nhạy hơn nhiều ở Δ nhỏ.

CÁCH DÙNG (từ thư mục GỐC repo):
    python -m src.evaluation.test_length_bias
Thuần CPU, dùng lại cache có sẵn, không encode lại gì.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .evaluate import load_ground_truth
    from .full_pipeline_eval import load_chunks
    from .ensemble_fusion import load_ranks, load_dup_groups, fuse_one_question
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.full_pipeline_eval import load_chunks
    from src.evaluation.ensemble_fusion import load_ranks, load_dup_groups, fuse_one_question


def aggregate(rrf_score, doc_ids, n_chunks_of, alpha, top_k, group_of=None, mode="log"):
    """Gộp chunk->document. alpha=0 => y hệt aggregate_max() hiện tại (đối chứng)."""
    best = {}
    for chunk_idx, score in rrf_score.items():
        d = doc_ids[chunk_idx]
        if d not in best or score > best[d]:
            best[d] = score
    if alpha != 0.0:
        adj = {}
        for d, s in best.items():
            n = n_chunks_of[d]
            if mode == "log":
                adj[d] = s - alpha * math.log(1.0 + n)
            else:  # chia
                adj[d] = s / (1.0 + alpha * math.log(1.0 + n))
        best = adj
    ranked = sorted(best.items(), key=lambda x: -x[1])
    if not group_of:
        return [d for d, _ in ranked[:top_k]]
    out, seen = [], set()
    for d, _ in ranked:
        g = group_of.get(d)
        if g is not None:
            if g in seen:
                continue
            seen.add(g)
        out.append(d)
        if len(out) >= top_k:
            break
    return out[:top_k]


def mcnemar_p(b: int, c: int) -> float:
    """p-value 2 phía, kiểm định dấu chính xác (binomial) — dùng đúng khi b+c nhỏ,
    KHÔNG dùng xấp xỉ chi-square vì ở đây b+c thường chỉ vài chục."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    total = sum(math.comb(n, i) for i in range(lo + 1))
    return min(1.0, 2.0 * total / (2.0 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    ap.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    ap.add_argument("--dedup-groups", default=None)
    ap.add_argument("--rank-v1", default="outputs/cache/ranks_v1.npz")
    ap.add_argument("--rank-harrier", default="outputs/cache/ranks_harrier.npz")
    ap.add_argument("--rank-bm25", default="outputs/cache/ranks_bm25.npz")
    ap.add_argument("--alphas", default="0,0.0002,0.0005,0.001,0.002,0.004,0.008")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    doc_ids = [c["doc_id"] for c in chunks]
    n_chunks_of = Counter(doc_ids)
    gt = load_ground_truth(args.ground_truth)
    group_of = load_dup_groups(args.dedup_groups)

    sources, ref_qids = {}, None
    for name, path in [("v1", args.rank_v1), ("harrier", args.rank_harrier), ("bm25", args.rank_bm25)]:
        qids, ranks = load_ranks(path)
        ref_qids = ref_qids or qids
        sources[name] = ranks

    weights = {"v1": 1.0, "harrier": 0.2, "bm25": 0.1}
    use_top = {n: 100 for n in sources}

    # ---- 0. Phân bố số chunk / document ------------------------------------------
    counts = sorted(n_chunks_of.values())
    print("=" * 78)
    print("0. PHÂN BỐ SỐ CHUNK MỖI DOCUMENT (nguồn gốc của thiên lệch)")
    print("=" * 78)
    print(f"   {len(counts):,} document / {len(chunks):,} chunk")
    for q in [50, 75, 90, 95, 99]:
        idx = int(len(counts) * q / 100)
        print(f"   phân vị {q:2d}% : {counts[min(idx, len(counts)-1)]:>7,} chunk")
    print(f"   nhỏ nhất  : {counts[0]:,} chunk")
    print(f"   LỚN NHẤT  : {counts[-1]:,} chunk")
    print(f"   => tài liệu dài nhất có NHIỀU HƠN {counts[-1] // max(counts[len(counts)//2],1)}x "
          f"số 'vé số' so với tài liệu trung vị khi lấy max")

    # ---- 1. Tài liệu đúng có bị tài liệu DÀI HƠN đè không? -------------------------
    print()
    print("=" * 78)
    print("1. TRONG CÁC CÂU SAI: tài liệu đúng có NGẮN HƠN tài liệu đã đè nó không?")
    print("=" * 78)
    gold_lens, blocker_lens, ratios = [], [], []
    n_fail = 0
    baseline_correct = {}
    for qi, qid in enumerate(ref_qids):
        if qid not in gt:
            continue
        gold = set(gt[qid])
        rank_rows = {n: sources[n][qi] for n in sources}
        combined = fuse_one_question(rank_rows, weights, use_top, 60)
        top5 = aggregate(combined, doc_ids, n_chunks_of, 0.0, args.top_k, group_of)
        ok = bool(gold & set(top5))
        baseline_correct[qid] = ok
        if ok:
            continue
        n_fail += 1
        gl = statistics.median([n_chunks_of[d] for d in gold if d in n_chunks_of] or [0])
        bl = statistics.median([n_chunks_of[d] for d in top5])
        if gl > 0:
            gold_lens.append(gl); blocker_lens.append(bl); ratios.append(bl / gl)

    if ratios:
        print(f"   {n_fail} câu sai, so được {len(ratios)} câu")
        print(f"   Trung vị số chunk của TÀI LIỆU ĐÚNG (bị bỏ lỡ) : {statistics.median(gold_lens):8.1f}")
        print(f"   Trung vị số chunk của 5 TÀI LIỆU ĐÃ ĐÈ NÓ       : {statistics.median(blocker_lens):8.1f}")
        print(f"   Trung vị TỈ LỆ (đè / đúng)                      : {statistics.median(ratios):8.2f}x")
        longer = sum(1 for r in ratios if r > 1.0)
        print(f"   Số câu mà tài liệu đè DÀI HƠN tài liệu đúng     : {longer}/{len(ratios)} "
              f"({100*longer/len(ratios):.1f}%)")
        print()
        if statistics.median(ratios) > 1.3 and longer / len(ratios) > 0.55:
            print("   ==> CÓ DẤU HIỆU thiên lệch độ dài. Sang phần 2 để thử khử.")
        else:
            print("   ==> KHÔNG thấy dấu hiệu thiên lệch độ dài rõ rệt — giả thuyết này YẾU,")
            print("       phần 2 nhiều khả năng cũng không cải thiện. Đọc số ở phần 2 rồi kết luận.")

    # ---- 2. Sweep alpha + McNemar ghép cặp ----------------------------------------
    print()
    print("=" * 78)
    print("2. THỬ GỘP CÓ KHỬ THIÊN LỆCH:  score = max_chunk - alpha*log(1+so_chunk)")
    print("=" * 78)
    print("   b = số câu SAI->ĐÚNG (được cứu) · c = số câu ĐÚNG->SAI (bị hỏng)")
    print("   p = kiểm định dấu ghép cặp (McNemar chính xác). Δ Recall một mình KHÔNG đủ kết luận.")
    print()
    print(f"   {'alpha':>8s} {'Recall@5':>10s} {'Δ':>9s} {'b(cứu)':>8s} {'c(hỏng)':>8s} {'p':>9s}")
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    rows = []
    base_recall = None
    for alpha in alphas:
        correct, rec_sum = {}, 0.0
        for qi, qid in enumerate(ref_qids):
            if qid not in gt:
                continue
            gold = set(gt[qid])
            rank_rows = {n: sources[n][qi] for n in sources}
            combined = fuse_one_question(rank_rows, weights, use_top, 60)
            top5 = aggregate(combined, doc_ids, n_chunks_of, alpha, args.top_k, group_of)
            correct[qid] = bool(gold & set(top5))
            rec_sum += len(gold & set(top5)) / len(gold)
        recall = rec_sum / len(baseline_correct)
        if base_recall is None:
            base_recall = recall
        b = sum(1 for q in correct if correct[q] and not baseline_correct[q])
        c = sum(1 for q in correct if not correct[q] and baseline_correct[q])
        p = mcnemar_p(b, c)
        rows.append({"alpha": alpha, "recall": recall, "delta": recall - base_recall,
                     "b_saved": b, "c_broken": c, "p": p})
        print(f"   {alpha:8.4f} {recall:10.4f} {recall-base_recall:+9.4f} {b:8d} {c:8d} {p:9.4f}")

    print()
    print("   CÁCH ĐỌC: alpha=0 là ĐỐI CHỨNG (phải ra đúng 0.9319). Một alpha đáng tin cần")
    print("   ĐỒNG THỜI: Δ dương, b >> c, và p < 0.05. Chỉ Δ dương mà b≈c là NHIỄU.")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu: {args.report}")


if __name__ == "__main__":
    main()
