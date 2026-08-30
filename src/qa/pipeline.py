from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Sequence

from .config import ContextConfig, LegalQAGenerationConfig, LLMConfig
from .context_builder import EvidenceContextBuilder
from .generator import LocalLLMGenerator
from .postprocess import clean_generated_answer, unsupported_legal_markers
from .prompts import available_prompt_variants, build_messages
from .types import QAInput, QAResult
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LegalQAPipeline:
    """Context construction, local generation and deterministic answer cleanup."""

    def __init__(
        self,
        llm_config: LLMConfig,
        context_config: ContextConfig,
        generation_config: LegalQAGenerationConfig,
        generator: LocalLLMGenerator | None = None,
        prompt_variant: str = "baseline",
    ):
        if prompt_variant not in available_prompt_variants():
            allowed = ", ".join(available_prompt_variants())
            raise ValueError(
                f"Unknown prompt_variant={prompt_variant!r}; choose one of: {allowed}"
            )
        self.generation_config = generation_config
        self.prompt_variant = prompt_variant
        self.generator = generator or LocalLLMGenerator(llm_config)
        self.context_builder = EvidenceContextBuilder(
            self.generator.tokenizer, context_config
        )

    def answer_many(
        self,
        inputs: Sequence[QAInput],
        corpus_chunks: Sequence[Mapping[str, Any]],
        on_result: Callable[[QAResult], None] | None = None,
    ) -> list[QAResult]:
        prepared: list[tuple[QAInput, str, int, list]] = []
        for item in inputs:
            built = self.context_builder.build(item.evidence, corpus_chunks)
            messages = build_messages(
                item.question, built.text, prompt_variant=self.prompt_variant
            )
            prompt_tokens = self.generator.count_message_tokens(messages)
            prepared.append(
                (item, built.text, prompt_tokens, [messages, list(built.evidence_used)])
            )

        results: list[QAResult] = []
        batch_size = self.generation_config.batch_size
        progress = tqdm(
            total=len(prepared),
            desc="Generating LegalQA answers",
            unit="question",
        )
        try:
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                raw_answers = self.generator.generate_batch(
                    [entry[3][0] for entry in batch], self.generation_config
                )
                for (item, context, prompt_tokens, payload), raw_answer in zip(
                    batch, raw_answers
                ):
                    answer = clean_generated_answer(raw_answer)
                    unsupported = unsupported_legal_markers(answer, context)
                    if unsupported:
                        logger.warning(
                            "question_id=%s contains unsupported legal markers: %s",
                            item.question_id,
                            unsupported,
                        )
                    result = QAResult(
                        question_id=item.question_id,
                        answer=answer,
                        prompt_tokens=prompt_tokens,
                        evidence_used=payload[1],
                        raw_answer=raw_answer,
                    )
                    results.append(result)
                    if on_result is not None:
                        on_result(result)
                progress.update(len(batch))
        finally:
            progress.close()
        return results

    @staticmethod
    def to_submission(results: Sequence[QAResult]) -> dict[str, dict[str, str]]:
        return {result.question_id: result.submission_item() for result in results}
