"""
CLI: kiểm tra file submission.json có đúng định dạng và ràng buộc của BTC trước khi đóng gói nộp bài.

Kiểm tra:
  1. Là JSON object hợp lệ.
  2. Mỗi giá trị phải có trường "answer" dạng mảng string.
  3. Không có document_id trùng lặp trong cùng một câu hỏi.
  4. (nếu truyền --corpus) mỗi document_id phải tồn tại trong selected-contexts.zip.
  5. Cảnh báo (không chặn) nếu câu hỏi nào có >5 document_id - BTC sẽ chấm 0 câu đó.
  6. (nếu truyền --questions) không thiếu question_id nào so với file câu hỏi tham chiếu
     (vd public-official.json) - Task 2 bắt buộc đủ mọi câu hỏi, Task 1 nên làm tương tự
     cho chắc dù thể lệ Task 1 không nêu rõ ràng buộc này.

Cách dùng (--corpus chấp nhận CẢ selected-contexts.zip LẪN thư mục đã giải nén sẵn, vd
data/raw/selected-contexts nếu build_pipeline.py đã tự giải nén và xóa file .zip gốc):
    python -m src.evaluation.validate_submission \\
        --predictions outputs/submission.json \\
        --corpus data/raw/selected-contexts.zip \\
        --questions data/raw/public-official.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Set

try:
    from ..training.data_utils import CorpusReader
except ImportError:  # cho phép chạy trực tiếp
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.training.data_utils import CorpusReader


def load_valid_doc_ids(corpus_path: str) -> Set[str]:
    """Dùng CorpusReader (data_utils.py) để hỗ trợ CẢ selected-contexts.zip lẫn thư mục đã giải
    nén - trước đây hàm này tự đọc thẳng bằng zipfile.ZipFile nên sẽ crash nếu build_pipeline.py
    đã tự giải nén (xóa mất file .zip gốc, chỉ còn thư mục)."""
    return set(CorpusReader(corpus_path).ids())


def validate(predictions_path: str, corpus_path: Optional[str], questions_path: Optional[str]) -> List[str]:
    errors: List[str] = []
    warnings: List[str] = []

    try:
        with open(predictions_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"JSON không hợp lệ: {e}"]

    if not isinstance(data, dict):
        return ["File gốc phải là JSON Object (key=question_id)"]

    valid_ids = load_valid_doc_ids(corpus_path) if corpus_path else None

    for qid, item in data.items():
        if not isinstance(item, dict):
            errors.append(f"[{qid}] giá trị phải là Object, không phải {type(item).__name__}")
            continue
        if "answer" not in item:
            errors.append(f"[{qid}] thiếu trường 'answer'")
            continue
        answer = item["answer"]
        if not isinstance(answer, list):
            errors.append(f"[{qid}] 'answer' phải là mảng (list), không phải {type(answer).__name__}")
            continue
        if not all(isinstance(d, str) for d in answer):
            errors.append(f"[{qid}] tất cả document_id trong 'answer' phải là string")
            continue
        if len(answer) != len(set(answer)):
            errors.append(f"[{qid}] có document_id trùng lặp trong answer: {answer}")
        if len(answer) > 5:
            warnings.append(f"[{qid}] có {len(answer)} document_id (>5) -> Recall/Precision câu này sẽ bị chấm 0")
        if valid_ids is not None:
            unknown = [d for d in answer if d not in valid_ids]
            if unknown:
                errors.append(f"[{qid}] document_id không tồn tại trong kho dữ liệu: {unknown}")

    if questions_path:
        with open(questions_path, encoding="utf-8") as f:
            all_questions = json.load(f)
        missing = set(all_questions.keys()) - set(data.keys())
        if missing:
            sample = sorted(missing)[:10]
            errors.append(f"Thiếu {len(missing)} question_id so với {questions_path}, ví dụ: {sample}")

    for w in warnings:
        print(f"[cảnh báo] {w}", file=sys.stderr)

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra định dạng submission.json trước khi nộp")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--corpus", default=None, help="Đường dẫn selected-contexts.zip HOẶC thư mục đã giải nén, để kiểm tra document_id có tồn tại")
    parser.add_argument("--questions", default=None, help="File câu hỏi tham chiếu (vd public-official.json) để kiểm tra không thiếu câu hỏi")
    args = parser.parse_args()

    errors = validate(args.predictions, args.corpus, args.questions)

    if errors:
        print(f"❌ Tìm thấy {len(errors)} lỗi:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✅ File hợp lệ, sẵn sàng đóng gói submission.zip")


if __name__ == "__main__":
    main()
