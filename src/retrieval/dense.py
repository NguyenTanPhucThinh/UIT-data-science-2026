from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .types import RetrievalResult


DEFAULT_MODEL = "bqbbao6/VN_legal_embedding_512"


class DenseRetriever:
    """Development dense retriever backed by the handed-off local FAISS index."""

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name
        self.device = self._resolve_device(device)

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = self._load_metadata(self.metadata_path)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "Artifact mismatch: FAISS ntotal=%d, metadata rows=%d"
                % (self.index.ntotal, len(self.metadata))
            )

        self.model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            device=self.device,
        )

        model_dim = self.model.get_embedding_dimension()
        if model_dim != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch: model={model_dim}, FAISS={self.index.d}"
            )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _load_metadata(path: Path) -> list[dict]:
        metadata: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
                for field in ("doc_id", "chunk_id", "passage"):
                    if field not in item:
                        raise ValueError(f"Missing '{field}' at {path}:{line_no}")
                metadata.append(item)
        if not metadata:
            raise ValueError(f"No metadata records found in {path}")
        return metadata

    def encode_query(self, question: str) -> np.ndarray:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        # The model follows an E5-style asymmetric setup.
        query_text = "query: " + question.strip()
        embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype=np.float32)

    def retrieve(self, question: str, top_k: int = 100) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        top_k = min(top_k, self.index.ntotal)
        query_vector = self.encode_query(question)
        scores, indices = self.index.search(query_vector, top_k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            if idx < 0:
                continue
            item = self.metadata[idx]
            results.append(
                RetrievalResult(
                    chunk_index=idx,
                    doc_id=str(item["doc_id"]),
                    chunk_id=str(item["chunk_id"]),
                    score=float(score),
                )
            )
        return results
