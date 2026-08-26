import json

import numpy as np

from src.evaluation.full_pipeline_eval import (
    load_questions,
    rerank_candidate_details,
    run_pipeline,
)


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


def test_run_pipeline_populates_optional_task2_evidence():
    class FakeEmbedding:
        def encode(self, texts, **kwargs):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    class FakeBm25:
        def get_scores(self, tokens):
            return np.asarray([0.1, 0.9], dtype=np.float32)

    chunks = [
        {"doc_id": "A", "chunk_id": "A_0", "passage": "A"},
        {"doc_id": "B", "chunk_id": "B_0", "passage": "B"},
    ]
    evidence = {}
    predictions, _ = run_pipeline(
        FakeEmbedding(),
        None,
        chunks,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        FakeBm25(),
        [{"question_id": "q1", "question": "Hỏi?", "question_segmented": "Hỏi ?"}],
        top_k=1,
        dense_top_n=2,
        bm25_top_n=2,
        candidate_docs=2,
        question_batch_size=1,
        evidence_output=evidence,
    )

    assert predictions["q1"] == [evidence["q1"][0]["doc_id"]]
    assert evidence["q1"][0]["passage"] in {"A", "B"}
