"""
Sinh ngữ cảnh (context) truy xuất từ pipeline Task 1 đã fine-tune, để BÀN GIAO cho người làm Task 2
(LegalQA) dùng làm đầu vào RAG - người làm Task 2 KHÔNG cần tự cài đặt lại toàn bộ pipeline
retrieval (embedding + BM25 + RRF + reranker), chỉ cần chạy đúng 1 lệnh này rồi dùng file JSON kết
quả.

Tái sử dụng NGUYÊN VẸN logic retrieval đã kiểm chứng của full_pipeline_eval.py (hàm run_pipeline())
- KHÔNG viết lại retrieval logic, tránh lệch kết quả so với pipeline Task 1 đã chốt (dòng #4/#6
trong claude/submission-log-task1.md, Score PUBLIC 0.8387).

Khác với full_pipeline_eval.py (chỉ xuất document_id, đúng định dạng nộp bài Task 1), script này
xuất chunk bằng chứng thực sự được reranker chọn, kèm provenance và score. Không nạp toàn văn
document vì văn bản pháp luật có thể rất dài và làm tràn context window của LLM Task 2.

Mặc định dùng model TỐT NHẤT hiện có (models/vietnamese-legal-embedding-ft-full +
models/reranker-ft-full - ĐÚNG cấu hình đã dùng tạo submission_v2.zip, Score PUBLIC 0.8387) - đổi
bằng --embedding-model/--reranker-model nếu muốn thử cấu hình khác (vd sau khi có cấu hình tốt hơn
từ các thử nghiệm đang làm ở Task 1).

LƯU Ý QUAN TRỌNG cho người dùng ở Task 2: --top-k mặc định ở đây là 5 (giống Task 1), nhưng Task 2
KHÔNG bị giới hạn "tối đa 5 document" như Task 1 (đó là ràng buộc riêng của ĐỊNH DẠNG NỘP BÀI Task 1,
xem claude/quy-dinh-cuoc-thi-task2.md mục 8) - có thể tăng --top-k lên (vd 8-10) nếu muốn nhiều ngữ
cảnh hơn cho LLM sinh câu trả lời, miễn còn vừa cửa sổ ngữ cảnh của model đã chọn.

Cách dùng:
    python -m src.evaluation.retrieve_for_task2 \\
        --questions <file câu hỏi Task 2 do BTC cấp> \\
        --output outputs/predictions/task2_context.json \\
        --top-k 5

Định dạng output (1 dòng/câu hỏi, đã đủ để ghép prompt RAG ngay):
{
  "<question_id>": {
    "question": "<nội dung câu hỏi>",
    "evidence": [
      {"doc_id": "...", "chunk_id": "...", "passage": "<chunk liên quan>", ...},
      ...
    ]
  }
}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    from .full_pipeline_eval import (
        load_chunks, load_questions, resolve_device, encode_chunks, build_bm25_index, run_pipeline,
    )
except ImportError:  # cho phép chạy trực tiếp
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evaluation.full_pipeline_eval import (
        load_chunks, load_questions, resolve_device, encode_chunks, build_bm25_index, run_pipeline,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-model", default="models/vietnamese-legal-embedding-ft-full",
                         help="Mặc định = model TỐT NHẤT hiện có (dùng cho submission_v2.zip)")
    parser.add_argument("--reranker-model", default="models/reranker-ft-full",
                         help="Mặc định = model TỐT NHẤT hiện có (dùng cho submission_v2.zip)")
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--questions", required=True, help="File câu hỏi Task 2 (BTC cấp) - hỗ trợ cùng 2 định dạng như full_pipeline_eval.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5, help="Số document/câu hỏi trả về - Task 2 KHÔNG bị giới hạn 5 như Task 1, có thể tăng nếu muốn nhiều ngữ cảnh hơn")
    parser.add_argument("--dense-top-n", type=int, default=100)
    parser.add_argument("--bm25-top-n", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-docs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--question-batch-size", type=int, default=64)
    parser.add_argument("--embedding-cache", default="outputs/cache/chunks_corpus_embeddings__ft_full.npz",
                         help="Mặc định tái dùng cache đã có sẵn (model -ft-full) - KHÔNG cần encode lại corpus")
    parser.add_argument("--limit-questions", type=int, default=None, help="Chỉ chạy N câu hỏi đầu (debug nhanh)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--fp16-inference", dest="fp16_inference", action="store_true", default=True)
    parser.add_argument("--no-fp16-inference", dest="fp16_inference", action="store_false")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
    except ImportError:
        logger.error("Thiếu sentence-transformers. Chạy: pip install sentence-transformers torch")
        sys.exit(1)
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        logger.error("Thiếu rank_bm25. Chạy: pip install rank_bm25")
        sys.exit(1)

    device = resolve_device(args.device)
    if device == "cuda":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    chunks = load_chunks(args.chunks)
    questions = load_questions(args.questions)
    if args.limit_questions:
        questions = questions[: args.limit_questions]

    logger.info("Load embedding model: %s", args.embedding_model)
    embedding_model = SentenceTransformer(args.embedding_model, device=device)
    if device == "cuda" and args.fp16_inference:
        try:
            embedding_model = embedding_model.half()
            logger.info("Đã cast embedding model sang fp16 (dùng --no-fp16-inference để tắt).")
        except Exception as e:  # noqa: BLE001
            logger.warning("Không cast được model sang fp16 (%s) - encode ở fp32.", e)
    logger.info("Load reranker model: %s", args.reranker_model)
    reranker_model = CrossEncoder(args.reranker_model, num_labels=1, device=device)

    chunk_embeddings = encode_chunks(
        embedding_model, chunks, args.batch_size, args.embedding_cache, passage_prefix=args.passage_prefix
    )
    bm25_index = build_bm25_index(chunks)

    evidence_by_qid: dict[str, list[dict]] = {}
    predictions, _ = run_pipeline(
        embedding_model, reranker_model, chunks, chunk_embeddings, bm25_index, questions,
        top_k=args.top_k, dense_top_n=args.dense_top_n, bm25_top_n=args.bm25_top_n, rrf_k=args.rrf_k,
        candidate_docs=args.candidate_docs, question_batch_size=args.question_batch_size,
        query_prefix=args.query_prefix, collect_stages=False, evidence_output=evidence_by_qid,
    )

    question_text_by_id = {q["question_id"]: q["question"] for q in questions}

    output = {}
    for qid in predictions:
        output[qid] = {
            "question": question_text_by_id.get(qid, ""),
            "evidence": evidence_by_qid.get(qid, []),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu ngữ cảnh truy xuất cho %d câu hỏi tại: %s", len(output), args.output)
    logger.info(
        "Bàn giao cho Task 2: mỗi câu hỏi có 'question' + 'evidence' đã xếp theo reranker. "
        "Dùng src.qa.context_builder để mở rộng chunk lân cận và cắt theo token budget của LLM."
    )


if __name__ == "__main__":
    main()
