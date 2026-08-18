from src.qa.config import ContextConfig, LegalQAGenerationConfig, LLMConfig
from src.qa.pipeline import LegalQAPipeline
from src.qa.types import EvidenceChunk, QAInput


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeGenerator:
    tokenizer = FakeTokenizer()

    def count_message_tokens(self, messages):
        return sum(len(item["content"]) for item in messages)

    def generate_batch(self, messages_batch, generation_config):
        return ["<answer>### Trả lời: Nội dung đúng.</answer>" for _ in messages_batch]


def test_qa_pipeline_generates_submission_without_loading_model():
    pipeline = LegalQAPipeline(
        LLMConfig(),
        ContextConfig(max_context_tokens=500, neighbor_window=0),
        LegalQAGenerationConfig(batch_size=2),
        generator=FakeGenerator(),
    )
    chunks = [{"doc_id": "1", "chunk_id": "1_0", "passage": "Nội dung đúng."}]
    inputs = [
        QAInput(
            question_id="q1",
            question="Hỏi?",
            evidence=(EvidenceChunk("1", "Nội dung đúng.", 0, chunk_id="1_0"),),
        )
    ]
    results = pipeline.answer_many(inputs, chunks)
    assert results[0].answer == "Nội dung đúng."
    assert pipeline.to_submission(results) == {"q1": {"answer": "Nội dung đúng."}}


def test_qa_pipeline_calls_checkpoint_callback_after_each_result():
    pipeline = LegalQAPipeline(
        LLMConfig(),
        ContextConfig(max_context_tokens=500, neighbor_window=0),
        LegalQAGenerationConfig(batch_size=2),
        generator=FakeGenerator(),
    )
    chunks = [{"doc_id": "1", "chunk_id": "1_0", "passage": "Nội dung đúng."}]
    inputs = [
        QAInput(
            question_id="q1",
            question="Hỏi?",
            evidence=(EvidenceChunk("1", "Nội dung đúng.", 0, chunk_id="1_0"),),
        )
    ]
    saved = []
    pipeline.answer_many(inputs, chunks, on_result=saved.append)
    assert [result.question_id for result in saved] == ["q1"]
