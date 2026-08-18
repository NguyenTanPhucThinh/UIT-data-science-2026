import json

from src.evaluation.run_legalqa import (
    _append_generation_checkpoint,
    _load_generation_checkpoint,
    _load_retrieval_context,
)


def test_generation_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "answers.jsonl"
    _append_generation_checkpoint(path, "q1", "Câu trả lời 1")
    _append_generation_checkpoint(path, "q2", "Câu trả lời 2")

    assert _load_generation_checkpoint(path) == {
        "q1": {"answer": "Câu trả lời 1"},
        "q2": {"answer": "Câu trả lời 2"},
    }


def test_load_retrieval_context_validates_and_reuses_entries(tmp_path):
    path = tmp_path / "retrieval.json"
    path.write_text(
        json.dumps(
            {
                "q1": {
                    "question": "Câu hỏi?",
                    "document_ids": ["doc1"],
                    "evidence": [{"doc_id": "doc1", "passage": "Bằng chứng"}],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    predictions, evidence = _load_retrieval_context(
        path, [{"question_id": "q1", "question": "Câu hỏi?"}]
    )
    assert predictions == {"q1": ["doc1"]}
    assert evidence["q1"][0]["passage"] == "Bằng chứng"
