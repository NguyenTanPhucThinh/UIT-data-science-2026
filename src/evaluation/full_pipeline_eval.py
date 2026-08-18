"""
Pipeline hybrid ĐẦY ĐỦ (dense + BM25 + RRF fusion + rerank) cho Task 1 - LegalIR, chạy hoàn toàn
LOCAL - không cần Qdrant Cloud (Storage & Representation) hay code retrieval thật của Core RAG.

Đây là bản dựng ĐÚNG theo 4 giai đoạn đã thống nhất trong docs/de-xuat-pipeline-task1.md:
    Giai đoạn 1: Retrieval 2 nhánh (dense cosine similarity + BM25 sparse) trên CÙNG một tập
                 chunk, hợp nhất bằng Reciprocal Rank Fusion (RRF) -> fused chunk ranking.
    Giai đoạn 2: Gom chunk -> document (lấy RRF score cao nhất trong các chunk cùng document)
                 -> danh sách candidate document.
    Giai đoạn 3: Rerank - ghép (câu hỏi, chunk đại diện tốt nhất của mỗi candidate document)
                 qua cross-encoder reranker.
    Giai đoạn 4: Chọn top-K document theo điểm rerank.

QUAN TRỌNG - dùng ĐÚNG data Kiên (Data & Utilities) đã xử lý, KHÔNG tự chunk lại bằng
chunk_passage() như baseline_retrieval.py:
    - Corpus: data/processed/chunks_corpus.jsonl (field 'passage' cho dense, 'passage_segmented'
      - đã qua VnCoreNLP - cho BM25). File này CHƯA có sẵn nếu bạn chưa chạy pipeline của Kiên.
      Sinh ra bằng: python "src/ingestion + EDA/build_pipeline.py" (cần Java 8+ cho VnCoreNLP,
      tự tải model VnCoreNLP lần chạy đầu).
    - Câu hỏi: khuyến khích dùng file đã tiền xử lý của Kiên (vd data/processed/val_preprocessed.jsonl,
      có sẵn field 'question_segmented' để khớp cách tách từ với corpus). Script này CŨNG chấp
      nhận file dạng train.json gốc ({qid: {"question":..., "answer":[...]}}) - khi đó BM25 sẽ
      tokenize câu hỏi bằng cách đơn giản (KHÔNG qua VnCoreNLP) và in cảnh báo, vì thiếu
      question_segmented sẽ làm BM25 kém chính xác hơn một chút so với corpus đã segment.

Cách dùng (sau khi đã có data/processed/chunks_corpus.jsonl từ Kiên):
    python -m src.evaluation.full_pipeline_eval \\
        --embedding-model models/vietnamese-legal-embedding-ft \\
        --reranker-model models/reranker-ft \\
        --chunks data/processed/chunks_corpus.jsonl \\
        --questions data/processed/val_preprocessed.jsonl \\
        --output outputs/full_pipeline_predictions.json \\
        --embedding-cache outputs/chunks_corpus_embeddings__ft.npz

Rồi chấm điểm: LƯU Ý evaluate.py hiện chỉ đọc ground-truth dạng train.json (1 JSON dict), CHƯA
đọc được *_preprocessed.jsonl của Kiên - nên dùng val_split_20.json (sinh bởi
src/training/split_train_json.py, xem docs/... - CÙNG bộ qid với val_preprocessed.jsonl vì cả
2 đều bắt nguồn từ train_val_split.json) làm --ground-truth, KHÔNG dùng val_preprocessed.jsonl:
    python -m src.evaluation.evaluate \\
        --ground-truth data/processed/val_split_20.json \\
        --predictions outputs/full_pipeline_predictions.json

QUAN TRỌNG - prefix "query: "/"passage: " (nhánh DENSE): bqbbao6/VN_legal_embedding_512
fine-tune từ intfloat/multilingual-e5-base (đã xác nhận qua model card HuggingFace), là
bi-encoder bất đối xứng kiểu E5 - BẮT BUỘC "query: " trước câu hỏi, "passage: " trước chunk
corpus khi encode DENSE, nếu không similarity gần như ngẫu nhiên. Script này tự thêm 2 prefix
đó mặc định (đổi bằng --query-prefix/--passage-prefix, truyền "" để tắt nếu dùng model khác).
Nhánh BM25 (sparse) và bước RERANK (cross-encoder, ghép (question, passage) làm 1 chuỗi rồi
chấm điểm) KHÔNG dùng prefix này - đó là quy ước riêng của bi-encoder E5, không áp dụng cho
BM25/cross-encoder. Nếu bạn có --embedding-cache sinh ra TRƯỚC khi có prefix, XÓA và encode lại.

DEBUG THEO GIAI ĐOẠN (--stage-eval, thêm 2026-08-13): mặc định script chỉ cho ra 1 con số cuối
cùng (sau rerank+top-k), khó biết giai đoạn nào của pipeline đang yếu. Thêm --stage-eval để ghi
thêm dự đoán RIÊNG cho từng giai đoạn (dense-only, BM25-only, sau hybrid merge trước rerank) bên
cạnh kết quả cuối, rồi dùng src/evaluation/stage_report.py để chấm điểm cả 4 giai đoạn cùng lúc
và xem chênh lệch Recall giữa các giai đoạn liên tiếp - vd nếu dense-only và hybrid_fused gần
bằng nhau nhưng final thấp hơn hẳn, nghĩa là RERANKER đang làm giảm chất lượng chứ không phải
retrieval yếu.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from ..training.data_utils import resolve_device
except ImportError:  # cho phép chạy trực tiếp
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.training.data_utils import resolve_device

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _naive_tokenize(text: str) -> List[str]:
    """Fallback khi không có sẵn text đã segment bằng VnCoreNLP (question_segmented/passage_segmented
    rỗng). Kém chính xác hơn segment thật (không gộp được từ ghép), nhưng vẫn hoạt động được."""
    return [w.lower() for w in _WORD_RE.findall(text)]


# ---------------------------------------------------------------------------
# Load dữ liệu
# ---------------------------------------------------------------------------

def load_chunks(chunks_path: str) -> List[dict]:
    """Đọc data/processed/chunks_corpus.jsonl (output Phase 4 của Kiên).

    Mỗi chunk cần tối thiểu: doc_id, passage. passage_segmented là tùy chọn (dùng cho BM25 nếu
    có, fallback naive tokenize nếu không).
    """
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "doc_id" not in obj or "passage" not in obj:
                raise ValueError(
                    f"Chunk thiếu field doc_id/passage: {list(obj.keys())} - "
                    "kiểm tra lại có đúng là data/processed/chunks_corpus.jsonl không."
                )
            chunks.append(obj)
    if not chunks:
        raise ValueError(f"Không đọc được chunk nào từ {chunks_path}")
    logger.info("Đã load %d chunk từ %s", len(chunks), chunks_path)
    return chunks


def load_questions(questions_path: str) -> List[dict]:
    """Chấp nhận 2 dạng:
      1. train.json gốc / train_split_80.json / val_split_20.json: {qid: {"question", "answer"}}
      2. File JSONL đã tiền xử lý của Kiên (train_preprocessed.jsonl/val_preprocessed.jsonl/...):
         mỗi dòng {"question_id", "question", "question_segmented", "answer_doc_ids"}
    Trả về danh sách dict đã chuẩn hóa: question_id, question, question_segmented (có thể None),
    answer_doc_ids.
    """
    # LƯU Ý: KHÔNG lọc theo answer/answer_doc_ids ở đây - khác với data_utils.load_labeled_questions()
    # (dùng khi TRAIN, phải bỏ câu không có nhãn). Hàm này dùng để TẠO DỰ ĐOÁN (giống
    # baseline_retrieval.py), nên phải giữ TOÀN BỘ câu hỏi trong file - kể cả public-official.json
    # (answer luôn = null, vì đó chính là tập cần dự đoán để nộp bài). Lọc mất sẽ khiến script
    # không tạo được dự đoán nào cho public test.
    with open(questions_path, encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        questions = []
        for qid, item in data.items():
            # Task 1 uses answer=[doc_id, ...], while Task 2 reference files use
            # answer="generated legal answer". Retrieval must not interpret characters in a
            # Task 2 answer string as document IDs.
            raw_answer = item.get("answer")
            answer = raw_answer if isinstance(raw_answer, list) else []
            questions.append(
                {
                    "question_id": str(qid),
                    "question": item["question"],
                    "question_segmented": None,
                    "answer_doc_ids": [str(d) for d in answer],
                }
            )
        logger.warning(
            "%s là file dạng train.json/public-official.json gốc, KHÔNG có question_segmented - "
            "BM25 sẽ tokenize câu hỏi bằng cách đơn giản (không qua VnCoreNLP), kém khớp hơn với "
            "corpus đã segment. Nên dùng file *_preprocessed.jsonl của Kiên nếu cần độ chính xác "
            "BM25 tốt nhất.",
            questions_path,
        )
        logger.info("Đã load %d câu hỏi từ %s", len(questions), questions_path)
        return questions

    # JSONL
    questions = []
    with open(questions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            answer_doc_ids = obj.get("answer_doc_ids") or []
            questions.append(
                {
                    "question_id": str(obj["question_id"]),
                    "question": obj["question"],
                    "question_segmented": obj.get("question_segmented"),
                    "answer_doc_ids": [str(d) for d in answer_doc_ids],
                }
            )
    if not questions:
        raise ValueError(f"Không đọc được câu hỏi nào từ {questions_path}")
    logger.info("Đã load %d câu hỏi từ %s", len(questions), questions_path)
    return questions


# ---------------------------------------------------------------------------
# Giai đoạn 0: encode + index
# ---------------------------------------------------------------------------

def encode_chunks(
    model, chunks: List[dict], batch_size: int, cache_path: Optional[str] = None, passage_prefix: str = ""
):
    """LƯU Ý: nếu đổi `passage_prefix` giữa các lần chạy, XÓA cache_path cũ trước - cache không
    lưu lại prefix đã dùng, dùng nhầm cache cũ (không prefix) sẽ cho điểm SAI."""
    import numpy as np

    if cache_path:
        cache_file = Path(cache_path)
        if cache_file.exists():
            logger.info("Tìm thấy cache embedding tại %s, đang load...", cache_path)
            data = np.load(cache_file, allow_pickle=True)
            if len(data["doc_ids"]) == len(chunks):
                logger.info("Đã load %d chunk từ cache (bỏ qua bước encode)", len(chunks))
                return data["embeddings"]
            logger.warning(
                "Cache có %d chunk nhưng corpus hiện tại có %d chunk - encode lại từ đầu.",
                len(data["doc_ids"]), len(chunks),
            )

    texts = [passage_prefix + c["passage"] for c in chunks]
    logger.info("Encoding %d chunk (dense, passage_prefix=%r)...", len(texts), passage_prefix)
    emb = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        doc_ids = np.array([c["doc_id"] for c in chunks], dtype=object)
        np.savez_compressed(cache_file, embeddings=emb, doc_ids=doc_ids)
        logger.info("Đã lưu cache embedding tại %s", cache_path)
    return emb


def build_bm25_index(chunks: List[dict]):
    from rank_bm25 import BM25Okapi

    tokenized = []
    n_fallback = 0
    for c in chunks:
        segmented = c.get("passage_segmented")
        if segmented:
            tokenized.append(segmented.split())
        else:
            n_fallback += 1
            tokenized.append(_naive_tokenize(c["passage"]))
    if n_fallback:
        logger.warning(
            "%d/%d chunk KHÔNG có passage_segmented - dùng tokenize đơn giản (fallback) cho BM25.",
            n_fallback, len(chunks),
        )
    logger.info("Building BM25 index trên %d chunk...", len(tokenized))
    return BM25Okapi(tokenized)


# ---------------------------------------------------------------------------
# Giai đoạn 1 + 2: hybrid retrieval (dense + BM25) -> RRF fusion -> candidate documents
# ---------------------------------------------------------------------------

def rrf_fuse(
    dense_scores,
    bm25_scores,
    doc_ids: Sequence[str],
    dense_top_n: int = 100,
    bm25_top_n: int = 100,
    rrf_k: int = 60,
) -> Dict[int, float]:
    """RRF fusion ở CẤP CHUNK cho MỘT câu hỏi.

    Args:
        dense_scores: mảng similarity, 1 giá trị / chunk (thứ tự khớp với `doc_ids`).
        bm25_scores: mảng điểm BM25, 1 giá trị / chunk (cùng thứ tự).
        doc_ids: doc_id tương ứng từng chunk (cùng thứ tự, chỉ dùng để biết độ dài - hàm này
            KHÔNG gom theo doc, xem `aggregate_to_documents` cho bước đó).

    Returns:
        {chunk_index: rrf_score} - CHỈ chứa các chunk lọt top-N của ít nhất 1 trong 2 nhánh
        (không phải toàn bộ corpus - đúng tinh thần RRF: chỉ hợp nhất ứng viên top của mỗi nhánh).
    """
    import numpy as np

    n = len(doc_ids)
    dense_top_n = min(dense_top_n, n)
    bm25_top_n = min(bm25_top_n, n)

    dense_top_idx = np.argpartition(-dense_scores, dense_top_n - 1)[:dense_top_n]
    dense_top_idx = dense_top_idx[np.argsort(-dense_scores[dense_top_idx])]

    bm25_top_idx = np.argpartition(-bm25_scores, bm25_top_n - 1)[:bm25_top_n]
    bm25_top_idx = bm25_top_idx[np.argsort(-bm25_scores[bm25_top_idx])]

    rrf_score: Dict[int, float] = defaultdict(float)
    for rank, idx in enumerate(dense_top_idx, start=1):
        rrf_score[int(idx)] += 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(bm25_top_idx, start=1):
        rrf_score[int(idx)] += 1.0 / (rrf_k + rank)
    return rrf_score


def aggregate_to_documents(
    rrf_score: Dict[int, float], doc_ids: Sequence[str], candidate_top_n: int = 20
) -> List[Tuple[str, int, float]]:
    """Gom các chunk đã fuse theo document_id, lấy chunk có RRF score CAO NHẤT làm đại diện.

    Returns:
        Danh sách (doc_id, chunk_index_đại_diện, doc_score) đã sort giảm dần, cắt còn
        `candidate_top_n` document - đây là các ứng viên đưa vào bước rerank (giai đoạn 3).
    """
    best_by_doc: Dict[str, Tuple[int, float]] = {}
    for chunk_idx, score in rrf_score.items():
        doc_id = doc_ids[chunk_idx]
        current = best_by_doc.get(doc_id)
        if current is None or score > current[1]:
            best_by_doc[doc_id] = (chunk_idx, score)

    ranked = sorted(
        [(doc_id, chunk_idx, score) for doc_id, (chunk_idx, score) in best_by_doc.items()],
        key=lambda x: -x[2],
    )
    return ranked[:candidate_top_n]


# ---------------------------------------------------------------------------
# Stage-eval (2026-08-13): đo Recall/Precision RIÊNG từng giai đoạn pipeline (dense-only,
# BM25-only, sau hybrid merge - trước rerank, sau rerank+top-k) để xem giai đoạn nào đang yếu,
# thay vì chỉ có 1 con số cuối cùng. Dùng --stage-eval ở main(), rồi xem báo cáo tổng hợp bằng
# src/evaluation/stage_report.py.
# ---------------------------------------------------------------------------

def build_doc_chunk_index(doc_ids: Sequence[str]) -> Dict[str, "object"]:
    """Precompute (1 LẦN DUY NHẤT, ngoài vòng lặp câu hỏi) danh sách chunk-index thuộc từng
    document. Dùng cho aggregate_chunks_to_docs_by_score() ở stage-eval - nếu build lại cho mỗi
    câu hỏi sẽ rất chậm (quét lại toàn bộ chunk N lần, N = số câu hỏi)."""
    import numpy as np

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, doc_id in enumerate(doc_ids):
        groups[doc_id].append(idx)
    return {doc_id: np.array(idxs, dtype=np.int64) for doc_id, idxs in groups.items()}


def aggregate_chunks_to_docs_by_score(
    scores, doc_to_indices: Dict[str, "object"], top_k: int = 5
) -> List[str]:
    """Gom TOÀN BỘ chunk theo document (dùng chỉ mục đã precompute bằng build_doc_chunk_index -
    KHÔNG giới hạn top-N chunk như RRF), lấy điểm cao nhất trong các chunk cùng document làm điểm
    đại diện, trả về top_k document_id theo điểm giảm dần.

    Dùng để đo Recall/Precision THUẦN của MỘT nhánh retrieval riêng lẻ (chỉ dense HOẶC chỉ BM25,
    KHÔNG qua RRF/candidate-restriction) ở chế độ --stage-eval - KHÔNG dùng cho luồng dự đoán
    chính thức (luồng chính thức dùng rrf_fuse + aggregate_to_documents + rerank_candidates).
    """
    doc_scores = [(doc_id, float(scores[idxs].max())) for doc_id, idxs in doc_to_indices.items()]
    doc_scores.sort(key=lambda x: -x[1])
    return [doc_id for doc_id, _ in doc_scores[:top_k]]


# ---------------------------------------------------------------------------
# Giai đoạn 3 + 4: rerank -> top-K
# ---------------------------------------------------------------------------

def rerank_candidate_details(
    reranker_model,
    question: str,
    candidates: List[Tuple[str, int, float]],
    chunks: List[dict],
    top_k: int = 5,
) -> List[dict]:
    """Rerank candidates and retain the representative evidence chunk for Task 2.

    `chunk_index` in this return value is deliberately named `corpus_chunk_index` to avoid
    confusion with the document-local `chunk_index` field stored inside each chunk record.
    """
    if not candidates:
        return []

    pairs = [(question, chunks[chunk_idx]["passage"]) for _, chunk_idx, _ in candidates]
    # show_progress_bar=False: hàm này gọi 1 LẦN CHO MỖI CÂU HỎI (vòng lặp bên ngoài) - nếu không
    # tắt tường minh, sentence-transformers tự in progress bar riêng mỗi lần gọi (mặc định bật khi
    # log level <= INFO), ra hàng nghìn dòng "Batches: 100%|..." rác màn hình, không có ích (mỗi
    # câu hỏi chỉ có vài chục candidate, chấm điểm gần như tức thì).
    scores = reranker_model.predict(pairs, show_progress_bar=False)

    reranked = sorted(
        zip(candidates, scores), key=lambda item: -float(item[1])
    )
    details: List[dict] = []
    for rank, ((doc_id, corpus_chunk_index, rrf_score), reranker_score) in enumerate(
        reranked[:top_k], start=1
    ):
        chunk = chunks[corpus_chunk_index]
        details.append(
            {
                "doc_id": str(doc_id),
                "corpus_chunk_index": int(corpus_chunk_index),
                "chunk_id": (
                    str(chunk["chunk_id"]) if chunk.get("chunk_id") is not None else None
                ),
                "document_chunk_index": chunk.get("chunk_index"),
                "legal_section": chunk.get("legal_section"),
                "document_name": chunk.get("doc_name"),
                "passage": chunk["passage"],
                "retrieval_rank": rank,
                "rrf_score": float(rrf_score),
                "reranker_score": float(reranker_score),
            }
        )
    return details


def rerank_candidates(
    reranker_model, question: str, candidates: List[Tuple[str, int, float]], chunks: List[dict], top_k: int = 5
) -> List[str]:
    """Backward-compatible Task 1 wrapper returning only ranked document IDs."""
    return [
        item["doc_id"]
        for item in rerank_candidate_details(
            reranker_model, question, candidates, chunks, top_k
        )
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_pipeline(
    embedding_model,
    reranker_model,
    chunks: List[dict],
    chunk_embeddings,
    bm25_index,
    questions: List[dict],
    top_k: int = 5,
    dense_top_n: int = 100,
    bm25_top_n: int = 100,
    rrf_k: int = 60,
    candidate_docs: int = 20,
    question_batch_size: int = 64,
    query_prefix: str = "",
    collect_stages: bool = False,
    evidence_output: Optional[Dict[str, List[dict]]] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """Trả về (predictions, stage_predictions).

    `predictions` = kết quả CUỐI CÙNG (sau rerank + top-k) - luôn có, không đổi hành vi so với
    trước đây.

    `stage_predictions` chỉ được điền nếu `collect_stages=True` (bật bằng --stage-eval ở CLI),
    gồm 3 giai đoạn TRUNG GIAN (đều cắt còn top_k document để so Recall@top_k công bằng với kết
    quả cuối) - dùng để debug xem giai đoạn nào của pipeline đang yếu:
        "dense_only"   : chỉ nhánh dense (embedding), KHÔNG qua BM25/RRF/rerank.
        "bm25_only"    : chỉ nhánh BM25 (sparse), KHÔNG qua dense/RRF/rerank.
        "hybrid_fused" : sau khi hợp nhất RRF (dense+BM25), TRƯỚC khi rerank.
    Nếu `collect_stages=False`, trả về stage_predictions = {} (rỗng).
    """
    import numpy as np

    doc_ids = [c["doc_id"] for c in chunks]
    predictions: Dict[str, List[str]] = {}
    stage_predictions: Dict[str, Dict[str, List[str]]] = (
        {"dense_only": {}, "bm25_only": {}, "hybrid_fused": {}} if collect_stages else {}
    )
    doc_to_indices = build_doc_chunk_index(doc_ids) if collect_stages else None

    for start in range(0, len(questions), question_batch_size):
        batch = questions[start : start + question_batch_size]
        # query_prefix chỉ dùng cho nhánh DENSE (embedding_model.encode ngay dưới đây) - BM25
        # (q_tokens ở dưới) và rerank (dùng q["question"] gốc, không prefix) KHÔNG áp dụng prefix
        # này, vì đó là quy ước riêng của bi-encoder E5-style.
        batch_texts = [query_prefix + q["question"] for q in batch]
        # show_progress_bar=False: cùng lý do như rerank_candidates() bên dưới - gọi lặp lại nhiều
        # lần (1 lần/batch câu hỏi), để mặc định sẽ in progress bar rác mỗi lần gọi.
        batch_dense_emb = embedding_model.encode(
            batch_texts,
            batch_size=question_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # (batch, n_chunks) - toàn bộ corpus embeddings đã nằm sẵn trong RAM (giống baseline_retrieval.py)
        dense_sims_batch = batch_dense_emb @ chunk_embeddings.T

        for i, q in enumerate(batch):
            dense_scores = dense_sims_batch[i]

            q_tokens = q["question_segmented"].split() if q.get("question_segmented") else _naive_tokenize(q["question"])
            bm25_scores = np.asarray(bm25_index.get_scores(q_tokens))

            rrf_score = rrf_fuse(dense_scores, bm25_scores, doc_ids, dense_top_n, bm25_top_n, rrf_k)
            candidates = aggregate_to_documents(rrf_score, doc_ids, candidate_docs)
            reranked_details = rerank_candidate_details(
                reranker_model, q["question"], candidates, chunks, top_k
            )
            top_docs = [item["doc_id"] for item in reranked_details]

            predictions[q["question_id"]] = top_docs
            if evidence_output is not None:
                evidence_output[q["question_id"]] = reranked_details

            if collect_stages:
                qid = q["question_id"]
                stage_predictions["dense_only"][qid] = aggregate_chunks_to_docs_by_score(
                    dense_scores, doc_to_indices, top_k
                )
                stage_predictions["bm25_only"][qid] = aggregate_chunks_to_docs_by_score(
                    bm25_scores, doc_to_indices, top_k
                )
                # candidates đã sort giảm dần theo RRF score (aggregate_to_documents) - lấy top_k
                # đầu tiên để so Recall@top_k công bằng với kết quả cuối (TRƯỚC khi qua rerank).
                stage_predictions["hybrid_fused"][qid] = [doc_id for doc_id, _, _ in candidates[:top_k]]

        logger.info("Đã xử lý %d/%d câu hỏi", min(start + question_batch_size, len(questions)), len(questions))

    return predictions, stage_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-model", required=True, help="vd: models/vietnamese-legal-embedding-ft hoặc model gốc chưa fine-tune")
    parser.add_argument("--reranker-model", required=True, help="vd: models/reranker-ft hoặc model gốc chưa fine-tune")
    parser.add_argument("--chunks", required=True, help="data/processed/chunks_corpus.jsonl (output Phase 4 của Kiên)")
    parser.add_argument("--questions", required=True, help="File câu hỏi: train.json-shape hoặc *_preprocessed.jsonl của Kiên")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-n", type=int, default=100, help="Số chunk top dense đưa vào RRF")
    parser.add_argument("--bm25-top-n", type=int, default=100, help="Số chunk top BM25 đưa vào RRF")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-docs", type=int, default=20, help="Số document ứng viên đưa vào rerank (giai đoạn 3)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size encode corpus (dense)")
    parser.add_argument("--question-batch-size", type=int, default=128)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--limit-questions", type=int, default=None, help="Chỉ chạy N câu hỏi đầu (debug nhanh)")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--query-prefix", default="query: ",
        help='Prefix thêm trước câu hỏi khi encode DENSE (E5-style, mặc định "query: "). KHÔNG áp dụng cho BM25/rerank. Truyền "" để tắt.',
    )
    parser.add_argument(
        "--passage-prefix", default="passage: ",
        help='Prefix thêm trước mỗi chunk khi encode DENSE (E5-style, mặc định "passage: "). KHÔNG áp dụng cho BM25/rerank. Truyền "" để tắt.',
    )
    parser.add_argument(
        "--fp16-inference", dest="fp16_inference", action="store_true", default=True,
        help="Cast embedding model sang fp16 trên GPU để encode nhanh hơn (mặc định BẬT, chỉ ảnh hưởng inference, không train nên không cần lo mất ổn định).",
    )
    parser.add_argument(
        "--no-fp16-inference", dest="fp16_inference", action="store_false",
        help="Tắt fp16, encode ở fp32 (chậm hơn nhưng an toàn nếu nghi ngờ fp16 làm sai kết quả).",
    )
    parser.add_argument(
        "--stage-eval", action="store_true", default=False,
        help=(
            "Bật chế độ debug theo giai đoạn (2026-08-13): ngoài file --output (kết quả CUỐI, sau "
            "rerank+top-k, dùng để nộp bài như bình thường), ghi thêm 3 file <output>.dense_only.json, "
            "<output>.bm25_only.json, <output>.hybrid_fused.json - dự đoán RIÊNG từng giai đoạn "
            "(chỉ dense / chỉ BM25 / sau hybrid merge trước rerank). Chấm điểm từng file bằng "
            "src/evaluation/stage_report.py để xem giai đoạn nào đang yếu. Chậm hơn bình thường "
            "một chút (thêm 2 bước gom chunk->document cho dense-only/BM25-only mỗi câu hỏi)."
        ),
    )
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
        # TF32 - tăng tốc matmul trên GPU Ampere+ (RTX 3060 trở lên) gần như miễn phí, không
        # đánh đổi gì đáng kể (giống đã áp dụng bên finetune_reranker.py).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    chunks = load_chunks(args.chunks)
    questions = load_questions(args.questions)
    if args.limit_questions:
        questions = questions[: args.limit_questions]

    logger.info("Load embedding model: %s", args.embedding_model)
    embedding_model = SentenceTransformer(args.embedding_model, device=device)
    if device == "cuda" and args.fp16_inference:
        # Encode ở fp16 - vì đây là INFERENCE (không backward/train) nên không cần lo loss
        # scaling/mất ổn định như lúc train - trên GPU có tensor core (Ampere+) thường nhanh hơn
        # đáng kể so với fp32. Nếu tốc độ vẫn không đổi sau khi bật, đa phần là do model/độ dài
        # chunk đã bão hoà compute chứ không phải do batch size hay dtype - không phải mọi máy
        # đều được lợi bằng nhau.
        try:
            embedding_model = embedding_model.half()
            logger.info("Đã cast embedding model sang fp16 để encode nhanh hơn (dùng --no-fp16-inference để tắt).")
        except Exception as e:  # noqa: BLE001
            logger.warning("Không cast được model sang fp16 (%s) - encode ở fp32 như bình thường.", e)
    logger.info("Load reranker model: %s", args.reranker_model)
    reranker_model = CrossEncoder(args.reranker_model, num_labels=1, device=device)

    logger.info(
        'Dùng prefix E5-style cho nhánh dense: query_prefix=%r, passage_prefix=%r (đổi bằng --query-prefix/--passage-prefix nếu model khác không cần)',
        args.query_prefix, args.passage_prefix,
    )
    chunk_embeddings = encode_chunks(
        embedding_model, chunks, args.batch_size, args.embedding_cache, passage_prefix=args.passage_prefix
    )
    bm25_index = build_bm25_index(chunks)

    if args.stage_eval:
        logger.info("--stage-eval BẬT: sẽ ghi thêm dự đoán riêng từng giai đoạn (dense_only/bm25_only/hybrid_fused).")

    predictions, stage_predictions = run_pipeline(
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
        collect_stages=args.stage_eval,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({qid: {"answer": docs} for qid, docs in predictions.items()}, f, ensure_ascii=False, indent=2)

    logger.info("Đã lưu %d dự đoán tại: %s", len(predictions), args.output)

    if args.stage_eval:
        for stage_key, stage_preds in stage_predictions.items():
            stage_path = output_path.with_suffix(f".{stage_key}{output_path.suffix}")
            with open(stage_path, "w", encoding="utf-8") as f:
                json.dump({qid: {"answer": docs} for qid, docs in stage_preds.items()}, f, ensure_ascii=False, indent=2)
            logger.info("Đã lưu dự đoán giai đoạn '%s' tại: %s", stage_key, stage_path)
        logger.info(
            "Bước tiếp theo (báo cáo tổng hợp từng giai đoạn): python -m src.evaluation.stage_report "
            "--ground-truth <val_split_20.json hoặc tương tự> --base %s",
            args.output,
        )
    else:
        # LƯU Ý (bug đã sửa 2026-08-14): KHÔNG dùng args.questions ở đây - đó là file câu hỏi
        # (--questions), KHÁC với file đáp án (--ground-truth, vd val_split_20.json) mà
        # evaluate.py cần. Dùng nhầm gây "json.decoder.JSONDecodeError: Extra data" vì file câu
        # hỏi là .jsonl (nhiều dòng JSON) trong khi evaluate.py đọc như 1 file JSON duy nhất.
        logger.info(
            "Bước tiếp theo: python -m src.evaluation.evaluate --ground-truth <vd: "
            "data/processed/val_split_20.json - file DAP AN, khac voi --questions ben tren> "
            "--predictions %s",
            args.output,
        )


if __name__ == "__main__":
    main()
