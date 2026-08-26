"""Build a LegalQA retrieval context from cached multi-source Task 1 rankings."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from tqdm import tqdm

from src.evaluation.ensemble_fusion import (
    fuse_one_question,
    parse_named,
    question_text_hash,
)
from src.evaluation.full_pipeline_eval import load_chunks, load_questions
from src.evaluation.task_selection import require_question_task

logger = logging.getLogger(__name__)


def _load_rank_bundle(path: str) -> tuple[list[str], np.ndarray, list[str] | None]:
    with np.load(path, allow_pickle=False) as data:
        question_ids = [str(item) for item in data["question_ids"]]
        ranks = np.asarray(data["ranks"], dtype=np.int32)
        hashes = (
            [str(item) for item in data["question_hashes"]]
            if "question_hashes" in data.files
            else None
        )
    if ranks.ndim != 2 or ranks.shape[0] != len(question_ids):
        raise ValueError(f"Invalid rank matrix in {path}: shape={ranks.shape}")
    return question_ids, ranks, hashes


def _rank_documents_with_evidence(
    rrf_score: Dict[int, float], chunks: Sequence[dict], top_k: int
) -> tuple[list[str], list[dict]]:
    """Aggregate chunk scores by max while retaining the winning chunk per document."""
    best: Dict[str, tuple[float, int]] = {}
    for corpus_chunk_index, score in rrf_score.items():
        if corpus_chunk_index < 0 or corpus_chunk_index >= len(chunks):
            raise IndexError(
                f"Rank contains corpus_chunk_index={corpus_chunk_index}, "
                f"but corpus has {len(chunks)} chunks"
            )
        doc_id = str(chunks[corpus_chunk_index]["doc_id"])
        previous = best.get(doc_id)
        if previous is None or score > previous[0]:
            best[doc_id] = (float(score), int(corpus_chunk_index))

    ranked = sorted(best.items(), key=lambda item: -item[1][0])[:top_k]
    document_ids: list[str] = []
    evidence: list[dict] = []
    for retrieval_rank, (doc_id, (score, corpus_chunk_index)) in enumerate(
        ranked, start=1
    ):
        chunk = chunks[corpus_chunk_index]
        document_ids.append(doc_id)
        evidence.append(
            {
                "doc_id": doc_id,
                "corpus_chunk_index": corpus_chunk_index,
                "chunk_id": (
                    str(chunk["chunk_id"]) if chunk.get("chunk_id") is not None else None
                ),
                "document_chunk_index": chunk.get("chunk_index"),
                "legal_section": chunk.get("legal_section"),
                "document_name": chunk.get("doc_name"),
                "passage": str(chunk.get("passage", "")),
                "retrieval_rank": retrieval_rank,
                "rrf_score": score,
                "reranker_score": None,
            }
        )
    return document_ids, evidence


def build_retrieval_context(
    *,
    questions: Sequence[dict],
    chunks: Sequence[dict],
    rank_paths: Dict[str, str],
    weights: Dict[str, float],
    use_top_n: Dict[str, int],
    rrf_k: int = 60,
    top_k: int = 5,
    allow_legacy_ranks: bool = False,
    show_progress: bool = True,
) -> dict[str, dict]:
    if not rank_paths:
        raise ValueError("At least one rank source is required")
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    expected_qids = [str(item["question_id"]) for item in questions]
    expected_hashes = [question_text_hash(item["question"]) for item in questions]
    loaded: Dict[str, np.ndarray] = {}
    for name, path in rank_paths.items():
        qids, ranks, hashes = _load_rank_bundle(path)
        if qids != expected_qids:
            raise ValueError(
                f"Question IDs/order in rank source {name!r} do not match --questions"
            )
        if hashes is None:
            if not allow_legacy_ranks:
                raise ValueError(
                    f"Rank source {name!r} has no question fingerprints. Rebuild it, "
                    "or pass --allow-legacy-ranks only when you have verified the input manually."
                )
            logger.warning("Using legacy rank source without question fingerprints: %s", path)
        elif hashes != expected_hashes:
            raise ValueError(
                f"Question text in rank source {name!r} is stale or does not match --questions"
            )
        requested_top_n = use_top_n.get(name, 100)
        if requested_top_n <= 0 or ranks.shape[1] < requested_top_n:
            raise ValueError(
                f"Rank source {name!r} has top-{ranks.shape[1]}, "
                f"but top-{requested_top_n} was requested"
            )
        loaded[name] = ranks

    output: dict[str, dict] = {}
    iterator = enumerate(expected_qids)
    if show_progress:
        iterator = tqdm(iterator, total=len(expected_qids), desc="Building Task 2 context")
    for question_index, question_id in iterator:
        rank_rows = {name: ranks[question_index] for name, ranks in loaded.items()}
        rrf_score = fuse_one_question(rank_rows, weights, use_top_n, rrf_k)
        document_ids, evidence = _rank_documents_with_evidence(rrf_score, chunks, top_k)
        if not evidence or any(not item["passage"].strip() for item in evidence):
            raise ValueError(f"No usable evidence produced for question_id={question_id}")
        output[question_id] = {
            "question": str(questions[question_index]["question"]),
            "document_ids": document_ids,
            "evidence": evidence,
        }
    return output


def _write_json_atomic(data: object, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--rank", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--weights", action="append", default=[], metavar="NAME=WEIGHT")
    parser.add_argument("--use-top-n", action="append", default=[], metavar="NAME=N")
    parser.add_argument("--default-top-n", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-legacy-ranks", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    require_question_task(args.questions, expected_task="2")
    rank_paths = parse_named(args.rank, "rank")
    explicit_weights = {
        name: float(value) for name, value in parse_named(args.weights, "weights").items()
    }
    unknown_weights = set(explicit_weights) - set(rank_paths)
    if unknown_weights:
        raise ValueError(f"Weights reference unknown rank sources: {sorted(unknown_weights)}")
    weights = {name: explicit_weights.get(name, 1.0) for name in rank_paths}

    explicit_top_n = {
        name: int(value) for name, value in parse_named(args.use_top_n, "use-top-n").items()
    }
    unknown_top_n = set(explicit_top_n) - set(rank_paths)
    if unknown_top_n:
        raise ValueError(f"--use-top-n references unknown sources: {sorted(unknown_top_n)}")
    use_top_n = {name: explicit_top_n.get(name, args.default_top_n) for name in rank_paths}

    questions = load_questions(args.questions)
    chunks = load_chunks(args.chunks)
    context = build_retrieval_context(
        questions=questions,
        chunks=chunks,
        rank_paths=rank_paths,
        weights=weights,
        use_top_n=use_top_n,
        rrf_k=args.rrf_k,
        top_k=args.top_k,
        allow_legacy_ranks=args.allow_legacy_ranks,
        show_progress=not args.no_progress,
    )
    _write_json_atomic(context, args.output)
    logger.info("Saved retrieval context for %d Task 2 questions to %s", len(context), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
