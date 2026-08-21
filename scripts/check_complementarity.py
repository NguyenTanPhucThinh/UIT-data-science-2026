"""
Kiem tra THAT (khong suy doan) xem gte-Qwen2 co "cuu" duoc nhung tai lieu dung ma AITeamVN bo lo hay
khong - truoc khi bo cong viet code fusion 3 chieu. Dung lai 2 file prediction DA CO SAN (khong can
GPU, khong can encode lai gi), chi doc JSON va so sanh set tai lieu du doan cho tung cau hoi.

Cach chay (tu thu muc goc UIT-data-science-2026, mat vai giay):
    python scripts/check_complementarity.py

Neu ty le "cuu duoc" o nhom n_relevant>=2 THAP (vd < 15-20%) -> gte-Qwen2 KHONG bu dap duoc diem yeu
nay, fusion 3 chieu it kha nang giup - nen bo huong nay, tap trung vao huong khac (tang top-k pool,
tinh chinh BM25, phan tich loi sau hon).
Neu ty le CAO (vd > 30-40%) -> co tin hieu that de thu fusion 3 chieu.
"""
from __future__ import annotations

import argparse
import json


def load_ground_truth(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(qid): set(str(d) for d in (item.get("answer") or [])) for qid, item in data.items()}


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(qid): set(str(d) for d in (item.get("answer") or [])) for qid, item in data.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    ap.add_argument("--predictions-a", default="outputs/predictions/aiteamvn_rerank_test.hybrid_fused.json",
                     help="AITeamVN hybrid_fused, bm25-weight=0.4 (file da co san tu lan test reranker)")
    ap.add_argument("--predictions-b", default="outputs/predictions/gte_qwen2_val_eval.json",
                     help="gte-Qwen2 final predictions, bm25-weight=0.1 (file da co san tu lan chay qua dem)")
    ap.add_argument("--label-a", default="AITeamVN")
    ap.add_argument("--label-b", default="gte-Qwen2")
    args = ap.parse_args()

    gt = load_ground_truth(args.ground_truth)
    pred_a = load_predictions(args.predictions_a)
    pred_b = load_predictions(args.predictions_b)

    print(f"Ground truth: {len(gt)} cau hoi | {args.label_a}: {len(pred_a)} du doan | {args.label_b}: {len(pred_b)} du doan")
    common = set(gt) & set(pred_a) & set(pred_b)
    print(f"So cau hoi co du lieu ca 3 nguon: {len(common)}\n")

    by_n = {}  # n_relevant -> {"missed": 0, "rescued": 0, "questions_with_miss": 0}
    total_missed = 0
    total_rescued = 0

    for qid in common:
        relevant = gt[qid]
        n_rel = len(relevant)
        missed_by_a = relevant - pred_a[qid]
        if not missed_by_a:
            continue
        bucket = by_n.setdefault(n_rel, {"missed": 0, "rescued": 0, "questions_with_miss": 0})
        bucket["questions_with_miss"] += 1
        for doc in missed_by_a:
            bucket["missed"] += 1
            total_missed += 1
            if doc in pred_b[qid]:
                bucket["rescued"] += 1
                total_rescued += 1

    print("=== Ty le tai lieu AITeamVN BO LO nhung gte-Qwen2 CO trong top-5 (theo n_relevant) ===")
    print(f"{'n_relevant':>10} | {'cau hoi bi mieu (co it nhat 1 mieu)':>36} | {'tai lieu bi mieu':>16} | {'duoc cuu boi gte-Qwen2':>22} | {'ty le cuu':>10}")
    for n_rel in sorted(by_n):
        b = by_n[n_rel]
        rate = 100.0 * b["rescued"] / b["missed"] if b["missed"] else 0.0
        print(f"{n_rel:>10} | {b['questions_with_miss']:>36} | {b['missed']:>16} | {b['rescued']:>22} | {rate:>9.1f}%")

    overall_rate = 100.0 * total_rescued / total_missed if total_missed else 0.0
    print(f"\nTONG: {total_missed} tai lieu bi AITeamVN bo lo, gte-Qwen2 cuu duoc {total_rescued} ({overall_rate:.1f}%)")

    print(
        "\nDIEN GIAI:\n"
        "- Ty le cuu THAP (vd <15-20%) o nhom n_relevant>=2 -> gte-Qwen2 gan nhu KHONG tim ra duoc\n"
        "  nhung tai lieu ma AITeamVN bo lo - fusion 3 chieu it kha nang giup nhom nay, nen BO huong\n"
        "  nay, chuyen sang huong khac (tang top-k pool, tinh chinh BM25, phan tich loi sau hon).\n"
        "- Ty le cuu CAO (vd >30-40%) -> co tin hieu that, dang thu code fusion 3 chieu.\n"
        "- Luu y: file predictions-a dung bm25-weight=0.4 (khong phai 0.1 dang dung chinh thuc) va\n"
        "  predictions-b bi anh huong boi truncation (--max-seq-length 512, xem check_truncation_gte_qwen2.py)\n"
        "  - ket qua nay la UOC LUONG so bo, khong phai con so cuoi cung chinh xac 100%."
    )


if __name__ == "__main__":
    main()
