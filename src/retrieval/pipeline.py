from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from .bm25 import BM25Retriever
from .dense import DenseRetriever, DEFAULT_MODEL
from .fusion import DEFAULT_RRF_K, aggregate_chunks_to_documents, rrf_fuse
from .reranker import DEFAULT_RERANKER_MODEL, Reranker


DEFAULT_DENSE_TOP_N = 100
DEFAULT_BM25_TOP_N = 100
DEFAULT_CANDIDATE_DOCS = 20
DEFAULT_FINAL_TOP_K = 5


class RetrievalPipeline:
    """Development end-to-end Retrieval pipeline.

    Dense + BM25 -> RRF -> chunk-to-document aggregation -> reranker -> top-5.
    """

    def __init__(
        self,
        index_path: str | Path = "vectorstore/corpus.index",
        metadata_path: str | Path = "embeddings/metadata.jsonl",
        embedding_model: str = DEFAULT_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        device: str = "auto",
        dense_top_n: int = DEFAULT_DENSE_TOP_N,
        bm25_top_n: int = DEFAULT_BM25_TOP_N,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_docs: int = DEFAULT_CANDIDATE_DOCS,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ):
        if final_top_k > 5:
            raise ValueError("Task 1 output must contain at most 5 document IDs")

        self.dense_top_n = dense_top_n
        self.bm25_top_n = bm25_top_n
        self.rrf_k = rrf_k
        self.candidate_docs = candidate_docs
        self.final_top_k = final_top_k

        self.dense = DenseRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            model_name=embedding_model,
            device=device,
        )
        self.bm25 = BM25Retriever(metadata_path=metadata_path)
        self._reranker_model = reranker_model
        self._reranker_device = device
        self._reranker: Reranker | None = None

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(
                metadata=self.dense.metadata,
                model_name=self._reranker_model,
                device=self._reranker_device,
            )
        return self._reranker

    def _release_dense_gpu_memory(self) -> None:
        """Free GPU memory held by the dense encoder before reranking."""
        reranker_device = self._resolve_device(self._reranker_device)
        if self.dense.device != "cuda" or reranker_device != "cuda":
            return

        self.dense.model.to("cpu")
        torch.cuda.empty_cache()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def retrieve(self, question: str) -> list[str]:
        if self.dense.device == "cuda":
            self.dense.model.to(self.dense.device)

        dense_results = self.dense.retrieve(question, top_k=self.dense_top_n)
        bm25_results = self.bm25.retrieve(question, top_k=self.bm25_top_n)

        fused = rrf_fuse(
            dense_results=dense_results,
            bm25_results=bm25_results,
            rrf_k=self.rrf_k,
        )

        candidates = aggregate_chunks_to_documents(
            fused_scores=fused,
            metadata=self.dense.metadata,
            top_k=self.candidate_docs,
        )

        self._release_dense_gpu_memory()

        reranked = self.reranker.rerank(
            question=question,
            candidates=candidates,
            top_k=self.final_top_k,
        )

        return [item.doc_id for item in reranked]

    def predict(self, questions: dict[str, dict]) -> dict[str, dict[str, list[str]]]:
        predictions: dict[str, dict[str, list[str]]] = {}

        for qid, item in questions.items():
            question = item["question"]
            predictions[str(qid)] = {"answer": self.retrieve(question)}

        return predictions


def load_question_dict(path: str | Path) -> dict[str, dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected train/submission question data as a JSON object")

    return data
