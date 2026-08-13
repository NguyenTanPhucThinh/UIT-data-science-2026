from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .types import RetrievalResult

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class BM25Retriever:
    """Development BM25 retriever over the handed-off metadata.jsonl artifact.

    The current project artifact already contains word-segmented `passage` text,
    so the dev implementation uses whitespace tokenization instead of running
    VnCoreNLP inside Retrieval.
    """

    def __init__(self, metadata_path: str | Path):
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.metadata: list[dict] = []
        tokenized_corpus: list[list[str]] = []

        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {self.metadata_path}:{line_no}"
                    ) from exc

                self._validate_item(item, line_no)
                self.metadata.append(item)
                tokenized_corpus.append(self._tokenize_passage(item["passage"]))

        if not self.metadata:
            raise ValueError(f"No metadata records found in {self.metadata_path}")

        self._doc_ids = [str(item["doc_id"]) for item in self.metadata]
        self.index = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _validate_item(item: dict, line_no: int) -> None:
        required = {"doc_id", "chunk_id", "passage"}
        missing = required.difference(item)
        if missing:
            raise ValueError(
                f"Metadata line {line_no} missing required fields: {sorted(missing)}"
            )

    @staticmethod
    def _tokenize_passage(text: str) -> list[str]:
        # The handed-off passage is already segmented; whitespace is therefore
        # the intended dev token boundary. Lowercasing is kept symmetric with
        # query tokenization and is sufficient for the initial implementation.
        return [token.lower() for token in text.split() if token]

    @staticmethod
    def _tokenize_query(text: str) -> list[str]:
        return [token.lower() for token in _WORD_RE.findall(text)]

    def retrieve(self, question: str, top_k: int = 100) -> list[RetrievalResult]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        query_tokens = self._tokenize_query(question)
        scores = np.asarray(self.index.get_scores(query_tokens), dtype=np.float32)

        k = min(top_k, scores.size)
        if k == 0:
            return []

        # Avoid a full Python sort over all chunks for the initial dev version.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            RetrievalResult(
                chunk_index=int(idx),
                doc_id=self._doc_ids[int(idx)],
                chunk_id=str(self.metadata[int(idx)]["chunk_id"]),
                score=float(scores[int(idx)]),
            )
            for idx in top_idx
        ]
