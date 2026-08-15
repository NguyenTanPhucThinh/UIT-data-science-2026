"""Run the complete local Task 2 pipeline: retrieval -> evidence context -> LLM -> JSON."""
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

from src.evaluation.full_pipeline_eval import (
    build_bm25_index,
    encode_chunks,
    load_chunks,
    load_questions,
    resolve_device,
    run_pipeline,
)
from src.qa.config import (
    DEFAULT_LLM_MODEL,
    ContextConfig,
    LegalQAGenerationConfig,
    LLMConfig,
)
from src.qa.pipeline import LegalQAPipeline
from src.qa.types import EvidenceChunk, QAInput
from src.evaluation.task_selection import require_question_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _write_json(data: object, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_retrieval_context(
    path: str | Path, questions: list[dict]
) -> tuple[dict[str, list[str]], dict[str, list[dict]]]:
    context_path = Path(path)
    if not context_path.exists():
        raise FileNotFoundError(
            f"Retrieval context not found for --reuse-retrieval-context: {context_path}"
        )
    with context_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Retrieval context must be a JSON object: {context_path}")

    predictions: dict[str, list[str]] = {}
    evidence_by_qid: dict[str, list[dict]] = {}
    for question in questions:
        qid = str(question["question_id"])
        item = data.get(qid)
        if not isinstance(item, dict):
            raise ValueError(f"Retrieval context is missing question_id={qid}")
        cached_question = str(item.get("question", "")).strip()
        if cached_question and cached_question != str(question["question"]).strip():
            raise ValueError(
                f"Retrieval context question mismatch for question_id={qid}"
            )
        document_ids = item.get("document_ids")
        evidence = item.get("evidence")
        if not isinstance(document_ids, list) or not isinstance(evidence, list):
            raise ValueError(f"Invalid retrieval context entry for question_id={qid}")
        predictions[qid] = [str(doc_id) for doc_id in document_ids]
        evidence_by_qid[qid] = evidence
    return predictions, evidence_by_qid


def _load_generation_checkpoint(path: Path) -> dict[str, dict[str, str]]:
    answers: dict[str, dict[str, str]] = {}
    if not path.exists():
        return answers
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Ignoring incomplete checkpoint line %d in %s", line_no, path
                )
                continue
            qid = str(item.get("question_id", ""))
            answer = item.get("answer")
            if not qid or not isinstance(answer, str):
                raise ValueError(f"Invalid generation checkpoint at {path}:{line_no}")
            answers[qid] = {"answer": answer}
    return answers


def _append_generation_checkpoint(path: Path, question_id: str, answer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"question_id": question_id, "answer": answer},
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()


def _release_retrieval_models(*models: object) -> None:
    for model in models:
        try:
            model.to("cpu")
        except (AttributeError, RuntimeError):
            try:
                model.model.to("cpu")
            except (AttributeError, RuntimeError):
                pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True, help="Final Task 2 submission JSON")
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument(
        "--embedding-model", default="models/vietnamese-legal-embedding-ft-full"
    )
    parser.add_argument("--reranker-model", default="models/reranker-ft-full")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument(
        "--embedding-cache",
        default="outputs/cache/chunks_corpus_embeddings__ft_full.npz",
    )
    parser.add_argument("--retrieval-context-output", default=None)
    parser.add_argument(
        "--reuse-retrieval-context",
        action="store_true",
        help="Load retrieval results from --retrieval-context-output and skip retrieval",
    )
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument(
        "--generation-checkpoint",
        default=None,
        help="JSONL file written after every generated answer",
    )
    parser.add_argument(
        "--resume-generation",
        action="store_true",
        help="Resume answers already present in --generation-checkpoint",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-n", type=int, default=100)
    parser.add_argument("--bm25-top-n", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-docs", type=int, default=20)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--question-batch-size", type=int, default=64)
    parser.add_argument("--answer-batch-size", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-documents", type=int, default=5)
    parser.add_argument("--max-chunks-per-document", type=int, default=3)
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--device", default="auto", help="Retrieval device")
    parser.add_argument("--llm-device", default="auto")
    parser.add_argument(
        "--llm-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit-questions", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_question_task(args.questions, expected_task="2")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.max_documents > args.top_k:
        logger.warning(
            "max_documents=%d but retrieval top_k=%d; only %d documents can be used",
            args.max_documents,
            args.top_k,
            args.top_k,
        )

    chunks = load_chunks(args.chunks)
    questions = load_questions(args.questions)
    if args.limit_questions:
        questions = questions[: args.limit_questions]

    question_by_id = {item["question_id"]: item["question"] for item in questions}
    if args.reuse_retrieval_context:
        if not args.retrieval_context_output:
            raise ValueError(
                "--reuse-retrieval-context requires --retrieval-context-output"
            )
        predictions, evidence_by_qid = _load_retrieval_context(
            args.retrieval_context_output, questions
        )
        logger.info(
            "Reused retrieval context for %d questions from %s",
            len(predictions),
            args.retrieval_context_output,
        )
    else:
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt before running LegalQA") from exc

        device = resolve_device(args.device)
        logger.info("Loading retrieval embedding model: %s", args.embedding_model)
        embedding_model = SentenceTransformer(args.embedding_model, device=device)
        if device == "cuda":
            try:
                embedding_model = embedding_model.half()
            except (RuntimeError, TypeError):
                logger.warning(
                    "Could not cast embedding model to fp16; continuing with original dtype"
                )
        logger.info("Loading retrieval reranker: %s", args.reranker_model)
        reranker_model = CrossEncoder(args.reranker_model, num_labels=1, device=device)

        chunk_embeddings = encode_chunks(
            embedding_model,
            chunks,
            args.embedding_batch_size,
            args.embedding_cache,
            passage_prefix=args.passage_prefix,
        )
        bm25_index = build_bm25_index(chunks)
        evidence_by_qid = {}
        predictions, _ = run_pipeline(
            embedding_model,
            reranker_model,
            chunks,
            chunk_embeddings,
            bm25_index,
            questions,
            top_k=args.top_k,
            dense_top_n=args.dense_top_n,
            bm25_top_n=args.bm25_top_n,
            rrf_k=args.rrf_k,
            candidate_docs=args.candidate_docs,
            question_batch_size=args.question_batch_size,
            query_prefix=args.query_prefix,
            evidence_output=evidence_by_qid,
        )

        if args.retrieval_context_output:
            _write_json(
                {
                    qid: {
                        "question": question_by_id[qid],
                        "document_ids": predictions[qid],
                        "evidence": evidence_by_qid.get(qid, []),
                    }
                    for qid in question_by_id
                },
                args.retrieval_context_output,
            )

        _release_retrieval_models(embedding_model, reranker_model)
        del embedding_model, reranker_model, chunk_embeddings, bm25_index
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    checkpoint_path = Path(
        args.generation_checkpoint or (str(args.output) + ".partial.jsonl")
    )
    checkpoint_answers = (
        _load_generation_checkpoint(checkpoint_path)
        if args.resume_generation
        else {}
    )
    unknown_checkpoint_ids = set(checkpoint_answers) - set(question_by_id)
    if unknown_checkpoint_ids:
        raise ValueError(
            "Generation checkpoint contains question IDs absent from the input: "
            + ", ".join(sorted(unknown_checkpoint_ids)[:5])
        )
    if not args.resume_generation:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("", encoding="utf-8")

    qa_inputs = [
        QAInput(
            question_id=qid,
            question=question_by_id[qid],
            evidence=tuple(
                EvidenceChunk.from_mapping(item)
                for item in evidence_by_qid.get(qid, [])
            ),
        )
        for qid in question_by_id
        if qid not in checkpoint_answers
    ]
    logger.info(
        "Generation resume state: %d completed, %d remaining (checkpoint=%s)",
        len(checkpoint_answers),
        len(qa_inputs),
        checkpoint_path,
    )
    results = []
    if qa_inputs:
        qa_pipeline = LegalQAPipeline(
            llm_config=LLMConfig(
                model_name_or_path=args.llm_model,
                device=args.llm_device,
                dtype=args.llm_dtype,
                local_files_only=args.local_files_only,
            ),
            context_config=ContextConfig(
                max_context_tokens=args.max_context_tokens,
                max_documents=args.max_documents,
                max_chunks_per_document=args.max_chunks_per_document,
                neighbor_window=args.neighbor_window,
            ),
            generation_config=LegalQAGenerationConfig(
                max_new_tokens=args.max_new_tokens,
                batch_size=args.answer_batch_size,
                do_sample=False,
            ),
        )

        def save_result(result) -> None:
            checkpoint_answers[result.question_id] = result.submission_item()
            _append_generation_checkpoint(
                checkpoint_path, result.question_id, result.answer
            )

        results = qa_pipeline.answer_many(
            qa_inputs, chunks, on_result=save_result
        )

    missing_answers = set(question_by_id) - set(checkpoint_answers)
    if missing_answers:
        raise RuntimeError(
            f"Generation ended with {len(missing_answers)} unanswered questions"
        )
    submission = {qid: checkpoint_answers[qid] for qid in question_by_id}
    _write_json(submission, args.output)
    logger.info("Saved %d Task 2 answers to %s", len(submission), args.output)

    if args.diagnostics_output:
        _write_json(
            {
                result.question_id: {
                    "prompt_tokens": result.prompt_tokens,
                    "answer_chars": len(result.answer),
                    "evidence_used": [item.to_dict() for item in result.evidence_used],
                }
                for result in results
            },
            args.diagnostics_output,
        )


if __name__ == "__main__":
    main()
