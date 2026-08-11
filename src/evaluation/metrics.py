"""
Metrics cho Task 1 - LegalIR: Recall@5 và Precision@5 theo đúng công thức BTC.

Tham khảo: docs/DSC2026_Task1_LegalIR_Data_Overview.docx, mục 4 (Đánh giá tác vụ).

Công thức chính thức:
    Recall_i    = |Relevant_i ∩ Predicted_i| / |Relevant_i|
    Precision_i = |Relevant_i ∩ Predicted_i| / |Predicted_i|   (0 nếu Predicted_i rỗng)
Ràng buộc: nếu |Predicted_i| > 5 thì Recall_i = Precision_i = 0 cho câu hỏi đó
(nhưng câu hỏi đó vẫn được tính vào điểm trung bình chung).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

MAX_PREDICTED_DOCS = 5


def compute_recall_precision(relevant: Sequence[str], predicted: Sequence[str]) -> Tuple[float, float]:
    """Tính Recall và Precision cho MỘT câu hỏi, đúng theo công thức và ràng buộc của BTC.

    Args:
        relevant: danh sách document_id đúng (ground truth) của câu hỏi.
        predicted: danh sách document_id hệ thống trả về cho câu hỏi (thứ tự không ảnh hưởng
            tới Recall/Precision - hai độ đo này chỉ quan tâm tập hợp, không quan tâm rank).

    Returns:
        (recall, precision)

    Raises:
        ValueError: nếu `relevant` rỗng (không có ground truth để so sánh).
    """
    relevant_set = {str(d).strip() for d in relevant}
    if not relevant_set:
        raise ValueError("relevant rỗng - không có ground truth cho câu hỏi này")

    # Ràng buộc nộp bài: vượt quá 5 document_id -> cả Recall và Precision = 0 cho câu hỏi này.
    if len(predicted) > MAX_PREDICTED_DOCS:
        return 0.0, 0.0

    predicted_set = {str(d).strip() for d in predicted}
    matched = relevant_set & predicted_set

    recall = len(matched) / len(relevant_set)
    precision = (len(matched) / len(predicted_set)) if predicted_set else 0.0
    return recall, precision


def evaluate_predictions(
    ground_truth: Dict[str, List[str]],
    predictions: Dict[str, List[str]],
    *,
    strict: bool = False,
) -> dict:
    """Đánh giá toàn bộ tập câu hỏi, trả về Recall/Precision trung bình + chi tiết từng câu.

    Args:
        ground_truth: {question_id: [document_id đúng, ...]}. Chỉ nên chứa các câu hỏi
            THỰC SỰ có nhãn (vd đã lọc bỏ answer=None của public-official.json).
        predictions: {question_id: [document_id dự đoán, ...]}, đúng định dạng submission.json
            (mảng answer, KHÔNG bọc thêm key "answer" ở đây - việc bóc tách đã làm ở evaluate.py).
        strict: nếu True, raise lỗi ngay khi thiếu dự đoán cho 1 câu hỏi thay vì coi là rỗng.

    Returns:
        dict gồm avg_recall, avg_precision, n_questions, n_missing_predictions,
        n_violations_over_5_docs, per_question (chi tiết từng câu để debug).
    """
    per_question: Dict[str, dict] = {}
    n_missing = 0
    n_violations = 0
    recalls: List[float] = []
    precisions: List[float] = []

    for qid, relevant in ground_truth.items():
        predicted = predictions.get(qid)
        if predicted is None:
            n_missing += 1
            if strict:
                raise KeyError(f"Thiếu dự đoán cho question_id={qid!r}")
            predicted = []  # không trả lời -> tính như Recall=0, Precision=0

        if len(predicted) > MAX_PREDICTED_DOCS:
            n_violations += 1

        recall, precision = compute_recall_precision(relevant, predicted)
        recalls.append(recall)
        precisions.append(precision)
        per_question[qid] = {
            "recall": recall,
            "precision": precision,
            "n_relevant": len(relevant),
            "n_predicted": len(predicted),
        }

    n = len(ground_truth)
    avg_recall = sum(recalls) / n if n else 0.0
    avg_precision = sum(precisions) / n if n else 0.0

    return {
        "avg_recall": avg_recall,
        "avg_precision": avg_precision,
        "n_questions": n,
        "n_missing_predictions": n_missing,
        "n_violations_over_5_docs": n_violations,
        "per_question": per_question,
    }
