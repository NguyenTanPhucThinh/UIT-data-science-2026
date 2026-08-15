import json

from src.evaluation.validate_legalqa_submission import validate_legalqa_submission


def test_validate_legalqa_submission_accepts_expected_schema(tmp_path):
    predictions = tmp_path / "submission.json"
    questions = tmp_path / "questions.json"
    predictions.write_text(
        json.dumps({"1": {"answer": "Câu trả lời."}}, ensure_ascii=False),
        encoding="utf-8",
    )
    questions.write_text(
        json.dumps({"1": {"question": "Câu hỏi?"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert validate_legalqa_submission(predictions, questions) == []


def test_validate_legalqa_submission_rejects_reasoning_wrapper(tmp_path):
    predictions = tmp_path / "submission.json"
    predictions.write_text(
        json.dumps({"1": {"answer": "<think>ẩn</think>"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    errors = validate_legalqa_submission(predictions)
    assert errors
    assert "reasoning" in errors[0]
