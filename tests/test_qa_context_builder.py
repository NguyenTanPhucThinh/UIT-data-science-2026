from src.qa.config import ContextConfig
from src.qa.context_builder import EvidenceContextBuilder
from src.qa.types import EvidenceChunk


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


def test_context_builder_adds_same_document_neighbor_and_respects_budget():
    chunks = [
        {"doc_id": "10", "chunk_id": "10_0", "passage": "Mở đầu", "legal_section": "Mở đầu"},
        {"doc_id": "10", "chunk_id": "10_1", "passage": "Điều liên quan", "legal_section": "Điều 1"},
        {"doc_id": "20", "chunk_id": "20_0", "passage": "Không cùng văn bản", "legal_section": "Điều 2"},
    ]
    evidence = EvidenceChunk(
        doc_id="10",
        passage="Điều liên quan",
        corpus_chunk_index=1,
        chunk_id="10_1",
        retrieval_rank=1,
    )
    builder = EvidenceContextBuilder(
        CharacterTokenizer(),
        ContextConfig(
            max_context_tokens=200,
            max_documents=1,
            max_chunks_per_document=2,
            neighbor_window=1,
        ),
    )

    result = builder.build([evidence], chunks)

    assert "Mở đầu" in result.text
    assert "Điều liên quan" in result.text
    assert "Không cùng văn bản" not in result.text
    assert result.token_count <= 200
    assert any(item.is_neighbor for item in result.evidence_used)


def test_context_builder_truncates_long_passage():
    chunks = [{"doc_id": "10", "chunk_id": "10_0", "passage": "x" * 1000}]
    evidence = EvidenceChunk("10", "x" * 1000, 0, chunk_id="10_0", retrieval_rank=1)
    builder = EvidenceContextBuilder(
        CharacterTokenizer(),
        ContextConfig(max_context_tokens=100, neighbor_window=0),
    )

    result = builder.build([evidence], chunks)

    assert 0 < result.token_count <= 100
    assert len(result.evidence_used) == 1
