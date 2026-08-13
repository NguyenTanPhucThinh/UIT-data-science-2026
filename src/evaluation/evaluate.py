"""
CLI: đánh giá một file dự đoán (định dạng submission.json) so với ground truth (train.json
hoặc bất kỳ file nào cùng schema {"qid": {"question":..., "answer":[...]}, ...}).

Cách dùng:
    python -m src.evaluation.evaluate \\
        --ground-truth data/raw/train.json \\
        --predictions outputs/my_predictions.json \\
        --report outputs/eval_report.json

File predictions PHẢI đúng định dạng submission.json chính thức:
    {"86666": {"answer": ["280282", "..."]}, ...}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    from .metrics import MAX_PREDICTED_DOCS, evaluate_predictions
except ImportError:  # cho phép chạy trực tiếp: python src/evaluation/evaluate.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evaluation.metrics import MAX_PREDICTED_DOCS, evaluate_predictions


def load_ground_truth(path: str) -> dict:
    """Đọc train.json (hoặc file cùng schema) -> {qid: [document_id đúng, ...]}.

    Tự động bỏ qua các câu hỏi answer=null/rỗng (vd nếu lỡ trỏ nhầm vào public-official.json).
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    gt = {}
    skipped = 0
    for qid, item in raw.items():
        answer = item.get("answer")
        if not answer:
            skipped += 1
            continue
        gt[qid] = [str(d) for d in answer]
    if skipped:
        print(f"[cảnh báo] bỏ qua {skipped} câu hỏi không có ground truth (answer null/rỗng)", file=sys.stderr)
    return gt


def load_predictions(path: str) -> dict:
    """Đọc file dự đoán đúng định dạng submission.json -> {qid: [document_id, ...]}.

    Không raise lỗi cứng khi định dạng sai từng câu hỏi - chỉ cảnh báo và coi như không trả lời,
    để vẫn tính được điểm cho phần còn lại. Muốn kiểm tra NGHIÊM NGẶT định dạng trước khi nộp,
    dùng src/evaluation/validate_submission.py.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    preds = {}
    for qid, item in raw.items():
        if not isinstance(item, dict) or "answer" not in item:
            print(f"[cảnh báo] question_id={qid!r} thiếu trường 'answer', coi như không trả lời", file=sys.stderr)
            preds[qid] = []
            continue
        answer = item["answer"]
        if not isinstance(answer, list):
            print(f"[cảnh báo] question_id={qid!r} có 'answer' không phải mảng, coi như không trả lời", file=sys.stderr)
            preds[qid] = []
            continue
        preds[qid] = [str(d) for d in answer]
    return preds


def compute_breakdown_by_n_relevant(per_question: dict) -> dict:
    """Gom câu hỏi theo số lượng document liên quan (n_relevant) và tính Recall/Precision trung
    bình riêng cho từng nhóm - dùng để KIỂM CHỨNG CÓ HỆ THỐNG giả thuyết quan sát thấy khi nhìn
    danh sách worst-k (vd "toàn bộ 10 câu Recall thấp nhất đều là câu hỏi chỉ có 1 document đúng"),
    thay vì chỉ suy đoán từ một mẫu nhỏ 10 câu.

    Trả về {n_relevant: {"count", "avg_recall", "avg_precision"}}, sort theo n_relevant tăng dần
    khi in ra (key ở đây là int, JSON report sẽ tự chuyển thành string).
    """
    groups: Dict[int, list] = defaultdict(list)
    for stats in per_question.values():
        groups[stats["n_relevant"]].append(stats)

    breakdown = {}
    for n_relevant, group in groups.items():
        breakdown[n_relevant] = {
            "count": len(group),
            "avg_recall": sum(g["recall"] for g in group) / len(group),
            "avg_precision": sum(g["precision"] for g in group) / len(group),
        }
    return breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá Recall/Precision Task 1 - LegalIR")
    parser.add_argument("--ground-truth", required=True, help="Đường dẫn train.json (hoặc file nhãn tương tự)")
    parser.add_argument("--predictions", required=True, help="Đường dẫn file dự đoán, đúng định dạng submission.json")
    parser.add_argument("--report", default=None, help="Nếu chỉ định, lưu báo cáo chi tiết (JSON) ra đường dẫn này")
    parser.add_argument("--worst-k", type=int, default=10, help="In ra K câu hỏi Recall thấp nhất để debug (0 = tắt)")
    parser.add_argument(
        "--breakdown-by-n-relevant", dest="breakdown", action="store_true", default=True,
        help="In bảng Recall/Precision trung bình theo số lượng document liên quan (n_relevant) mỗi câu hỏi (mặc định BẬT) - vd để kiểm chứng câu hỏi 1-document có khó hơn câu hỏi nhiều-document hay không.",
    )
    parser.add_argument("--no-breakdown-by-n-relevant", dest="breakdown", action="store_false")
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions)

    result = evaluate_predictions(ground_truth, predictions)

    print("=" * 60)
    print(f"Số câu hỏi đánh giá   : {result['n_questions']}")
    print(f"Thiếu dự đoán         : {result['n_missing_predictions']}")
    print(f"Vi phạm (>{MAX_PREDICTED_DOCS} doc)       : {result['n_violations_over_5_docs']}")
    print("-" * 60)
    print(f"Recall trung bình     : {result['avg_recall']:.4f}   <-- độ đo chính")
    print(f"Precision trung bình  : {result['avg_precision']:.4f}   <-- độ đo phụ")
    print("=" * 60)

    if args.worst_k > 0:
        worst = sorted(result["per_question"].items(), key=lambda kv: kv[1]["recall"])[: args.worst_k]
        print(f"\n{args.worst_k} câu hỏi có Recall thấp nhất (để debug):")
        for qid, stats in worst:
            print(
                f"  {qid}: recall={stats['recall']:.2f} precision={stats['precision']:.2f} "
                f"(relevant={stats['n_relevant']}, predicted={stats['n_predicted']})"
            )

    breakdown = None
    if args.breakdown:
        breakdown = compute_breakdown_by_n_relevant(result["per_question"])
        print("\nPhân tích theo số lượng document liên quan (n_relevant) mỗi câu hỏi:")
        print(f"  {'n_relevant':>10s} {'so_cau_hoi':>11s} {'Recall_TB':>10s} {'Precision_TB':>13s}")
        for n_relevant in sorted(breakdown):
            stats = breakdown[n_relevant]
            print(
                f"  {n_relevant:>10d} {stats['count']:>11d} {stats['avg_recall']:>10.4f} "
                f"{stats['avg_precision']:>13.4f}"
            )
        if 1 in breakdown and len(breakdown) > 1:
            other_recalls = [s["avg_recall"] * s["count"] for n, s in breakdown.items() if n != 1]
            other_counts = [s["count"] for n, s in breakdown.items() if n != 1]
            if sum(other_counts):
                avg_recall_multi = sum(other_recalls) / sum(other_counts)
                gap = avg_recall_multi - breakdown[1]["avg_recall"]
                if gap > 0.05:
                    print(
                        f"\n  -> Câu hỏi chỉ có 1 document đúng (n_relevant=1) có Recall thấp hơn "
                        f"~{gap:.4f} so với câu hỏi có >1 document đúng - xác nhận có hệ thống giả "
                        "thuyết quan sát ở danh sách worst-k phía trên (không phải trùng hợp ngẫu nhiên)."
                    )

    if args.report:
        report_data = dict(result)
        if breakdown is not None:
            report_data["breakdown_by_n_relevant"] = {str(k): v for k, v in breakdown.items()}
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu báo cáo chi tiết: {args.report}")


if __name__ == "__main__":
    main()
