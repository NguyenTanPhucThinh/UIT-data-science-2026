import numpy as np
import pytest

from src.evaluation.ensemble_context_for_task2 import build_retrieval_context
from src.evaluation.ensemble_fusion import save_ranks


def _save(path, question, ranks):
    save_ranks(
        str(path),
        [question["question_id"]],
        np.asarray([ranks], dtype=np.int32),
        question_texts=[question["question"]],
    )


def test_build_retrieval_context_keeps_winning_chunk_and_required_contract(tmp_path):
    question = {"question_id": "q1", "question": "Quy định nào được áp dụng?"}
    chunks = [
        {"doc_id": "A", "chunk_id": "A_0", "chunk_index": 0, "passage": "A0"},
        {"doc_id": "A", "chunk_id": "A_1", "chunk_index": 1, "passage": "A1"},
        {"doc_id": "B", "chunk_id": "B_0", "chunk_index": 0, "passage": "B0"},
        {"doc_id": "C", "chunk_id": "C_0", "chunk_index": 0, "passage": "C0"},
    ]
    anchor = tmp_path / "anchor.npz"
    harrier = tmp_path / "harrier.npz"
    bm25 = tmp_path / "bm25.npz"
    _save(anchor, question, [0, 2, 3])
    _save(harrier, question, [2, 1, 3])
    _save(bm25, question, [3, 0, 2])

    context = build_retrieval_context(
        questions=[question],
        chunks=chunks,
        rank_paths={"aiteamvn": str(anchor), "harrier": str(harrier), "bm25": str(bm25)},
        weights={"aiteamvn": 1.0, "harrier": 0.2, "bm25": 0.1},
        use_top_n={"aiteamvn": 3, "harrier": 3, "bm25": 3},
        rrf_k=60,
        top_k=3,
        show_progress=False,
    )

    item = context["q1"]
    assert item["question"] == question["question"]
    assert item["document_ids"] == [evidence["doc_id"] for evidence in item["evidence"]]
    assert item["document_ids"][0] == "B"
    assert item["evidence"][0]["corpus_chunk_index"] == 2
    assert item["evidence"][0]["passage"] == "B0"
    evidence_a = next(evidence for evidence in item["evidence"] if evidence["doc_id"] == "A")
    assert evidence_a["corpus_chunk_index"] == 0
    assert item["evidence"][0]["reranker_score"] is None


def test_build_retrieval_context_rejects_stale_question_text(tmp_path):
    old_question = {"question_id": "q1", "question": "Câu hỏi cũ"}
    rank_path = tmp_path / "rank.npz"
    _save(rank_path, old_question, [0])

    with pytest.raises(ValueError, match="stale"):
        build_retrieval_context(
            questions=[{"question_id": "q1", "question": "Câu hỏi mới"}],
            chunks=[{"doc_id": "A", "passage": "Nội dung"}],
            rank_paths={"aiteamvn": str(rank_path)},
            weights={"aiteamvn": 1.0},
            use_top_n={"aiteamvn": 1},
            show_progress=False,
        )
