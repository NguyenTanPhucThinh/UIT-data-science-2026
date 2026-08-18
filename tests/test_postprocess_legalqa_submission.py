import json

from src.evaluation.postprocess_legalqa_submission import postprocess_submission


def test_postprocess_submission_preserves_schema_and_reports_changes(tmp_path):
    source = tmp_path / "submission_raw.json"
    output = tmp_path / "submission.json"
    source.write_text(
        json.dumps(
            {
                "1": {
                    "answer": "Dựa trên ngữ cảnh được cung cấp, người lao động được nghỉ 05 ngày."
                },
                "2": {"answer": "Theo Điều 5, thời hạn là 30 ngày."},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = postprocess_submission(source, output)
    cleaned = json.loads(output.read_text(encoding="utf-8"))

    assert cleaned == {
        "1": {"answer": "Người lao động được nghỉ 05 ngày."},
        "2": {"answer": "Theo Điều 5, thời hạn là 30 ngày."},
    }
    assert report["total_answers"] == 2
    assert report["changed_answers"] == 1
    assert report["unchanged_answers"] == 1
