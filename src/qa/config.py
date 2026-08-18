from __future__ import annotations

from dataclasses import dataclass


DEFAULT_LLM_MODEL = "AITeamVN/Vi-Qwen2-1.5B-RAG"


@dataclass(frozen=True)
class LLMConfig:
    model_name_or_path: str = DEFAULT_LLM_MODEL
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False


@dataclass(frozen=True)
class ContextConfig:
    max_context_tokens: int = 6000
    max_documents: int = 5
    max_chunks_per_document: int = 3
    neighbor_window: int = 1
    include_document_name: bool = True
    include_legal_section: bool = True

    def __post_init__(self) -> None:
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be > 0")
        if self.max_documents <= 0:
            raise ValueError("max_documents must be > 0")
        if self.max_chunks_per_document <= 0:
            raise ValueError("max_chunks_per_document must be > 0")
        if self.neighbor_window < 0:
            raise ValueError("neighbor_window must be >= 0")


@dataclass(frozen=True)
class LegalQAGenerationConfig:
    max_new_tokens: int = 768
    batch_size: int = 1
    do_sample: bool = False
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
