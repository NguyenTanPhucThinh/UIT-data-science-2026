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


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá Recall/Precision Task 1 - LegalIR")
    parser.add_argument("--ground-truth", required=True, help="Đường dẫn train.json (hoặc file nhãn tương tự)")
    parser.add_argument("--predictions", required=True, help="Đường dẫn file dự đoán, đúng định dạng submission.json")
    parser.add_argument("--report", default=None, help="Nếu chỉ định, lưu báo cáo chi tiết (JSON) ra đường dẫn này")
    parser.add_argument("--worst-k", type=int, default=10, help="In ra K câu hỏi Recall thấp nhất để debug (0 = tắt)")
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

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu báo cáo chi tiết: {args.report}")


if __name__ == "__main__":
    main()
