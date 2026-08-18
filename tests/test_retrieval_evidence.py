import json

from src.evaluation.full_pipeline_eval import load_questions, rerank_candidate_details


class FakeReranker:
    def predict(self, pairs, show_progress_bar=False):
        assert len(pairs) == 2
        return [0.2, 0.9]


def test_rerank_candidate_details_retains_chunk_provenance():
    chunks = [
        {
            "doc_id": "1",
            "chunk_id": "1_0",
            "chunk_index": 0,
            "passage": "a",
            "legal_section": "Điều 1",
        },
        {
            "doc_id": "2",
            "chunk_id": "2_0",
            "chunk_index": 0,
            "passage": "b",
            "legal_section": "Điều 2",
        },
    ]
    result = rerank_candidate_details(
        FakeReranker(), "question", [("1", 0, 0.1), ("2", 1, 0.2)], chunks, top_k=2
    )
    assert result[0]["doc_id"] == "2"
    assert result[0]["corpus_chunk_index"] == 1
    assert result[0]["chunk_id"] == "2_0"
    assert result[0]["reranker_score"] == 0.9


def test_load_questions_does_not_treat_task2_answer_as_document_ids(tmp_path):
    path = tmp_path / "task2.json"
    path.write_text(
        json.dumps({"q1": {"question": "Hỏi?", "answer": "Câu trả lời."}}, ensure_ascii=False),
        encoding="utf-8",
    )
    loaded = load_questions(str(path))
    assert loaded[0]["answer_doc_ids"] == []
