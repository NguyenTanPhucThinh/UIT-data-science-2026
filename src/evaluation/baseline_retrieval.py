"""
Retrieval baseline "brute-force": CHỈ dùng embedding model + cosine similarity trên toàn bộ
corpus, KHÔNG có BM25/hybrid fusion/reranker.

Mục đích: đo Recall/Precision MỐC của một embedding model (trước/sau fine-tune) một cách ĐỘC
LẬP với tiến độ pipeline retrieval production (src/retrieval, phụ trách: Nguyễn Trường Giang).
Đây KHÔNG phải pipeline cuối cùng của nhóm — pipeline thật sẽ có thêm BM25 hybrid + reranker
theo đúng kiến trúc trong docs/rag_pipeline_with_anti_hallucination.svg. Khi src/retrieval hoàn
thiện, nên chạy evaluate.py trên output của pipeline THẬT thay vì script này.

Cách dùng:
    python -m src.evaluation.baseline_retrieval \\
        --model bqbbao6/VN_legal_embedding_512 \\
        --corpus data/raw/selected-contexts.zip \\
        --questions data/raw/train.json \\
        --output outputs/baseline_predictions.json \\
        --top-k 5 \\
        --embedding-cache outputs/corpus_embeddings__bqbbao6.npz

Sau đó chấm điểm:
    python -m src.evaluation.evaluate \\
        --ground-truth data/raw/train.json \\
        --predictions outputs/baseline_predictions.json

`--embedding-cache`: vì corpus (8.532 văn bản, hàng chục nghìn chunk) không đổi giữa các lần
chạy, embedding của corpus được lưu cache ra đĩa (.npz) - lần chạy sau (vd sau khi fine-tune
xong, muốn so sánh model mới) chỉ cần đổi `--model` VÀ đổi luôn `--embedding-cache` sang path
khác (mỗi model một cache riêng, vì embedding của model khác nhau thì khác nhau hoàn toàn).

QUAN TRỌNG - prefix "query: "/"passage: ": bqbbao6/VN_legal_embedding_512 fine-tune từ
intfloat/multilingual-e5-base (đã xác nhận qua model card HuggingFace) - là bi-encoder BẤT ĐỐI
XỨNG kiểu E5, BẮT BUỘC thêm "query: " trước câu hỏi và "passage: " trước văn bản corpus trước
khi encode, nếu không sẽ ra similarity gần như ngẫu nhiên. Mặc định script này đã tự thêm 2
prefix trên (đổi qua --query-prefix/--passage-prefix nếu dùng model khác không cần prefix -
truyền chuỗi rỗng "" để tắt). Nếu bạn có --embedding-cache sinh ra TRƯỚC khi có prefix (tên
file cũ), XÓA cache đó và chạy lại - cache cũ không có prefix sẽ cho điểm Recall SAI/thấp hơn
thực tế.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from ..training.data_utils import CorpusReader, chunk_passage, resolve_device
except ImportError:  # cho phép chạy trực tiếp: python src/evaluation/baseline_retrieval.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.training.data_utils import CorpusReader, chunk_passage, resolve_device


def build_corpus_chunks(
    corpus: CorpusReader, max_chars: int = 2000, passage_prefix: str = ""
) -> Tuple[List[str], List[str]]:
    """Chunk toàn bộ corpus, trả về (danh sách text từng chunk, danh sách doc_id tương ứng).

    `passage_prefix` (vd "passage: " cho model E5-style) được thêm vào TRƯỚC khi encode, KHÔNG
    ảnh hưởng tới doc_id/việc chunk văn bản.
    """
    texts: List[str] = []
    doc_ids: List[str] = []
    for doc in corpus.iter_documents():
        for chunk in chunk_passage(doc.passage, max_chars=max_chars):
            texts.append(passage_prefix + chunk)
            doc_ids.append(doc.id)
    return texts, doc_ids


def encode_corpus(model, texts: List[str], batch_size: int = 64):
    import numpy as np

    logger.info("Encoding %d chunk từ corpus...", len(texts))
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def load_or_build_corpus_embeddings(
    model, corpus: CorpusReader, cache_path: str, max_chars: int, batch_size: int, passage_prefix: str = ""
):
    """Load embedding corpus từ cache nếu có (và khớp số lượng văn bản), nếu không thì encode
    lại từ đầu rồi lưu cache. Cache gồm: embeddings (float32 [n_chunks, dim]) + doc_ids song song.

    LƯU Ý: cache KHÔNG lưu lại prefix đã dùng - nếu đổi --passage-prefix (vd bật/tắt "passage: ")
    giữa các lần chạy mà dùng lại cùng 1 file cache, kết quả sẽ SAI vì cache vẫn là embedding cũ.
    Xóa cache thủ công khi đổi prefix.
    """
    import numpy as np

    cache_file = Path(cache_path)
    if cache_file.exists():
        logger.info("Tìm thấy cache embedding tại %s, đang load...", cache_path)
        data = np.load(cache_file, allow_pickle=True)
        doc_ids = list(data["doc_ids"])
        embeddings = data["embeddings"]
        logger.info("Đã load %d chunk từ cache (bỏ qua bước encode)", len(doc_ids))
        return embeddings, doc_ids

    texts, doc_ids = build_corpus_chunks(corpus, max_chars=max_chars, passage_prefix=passage_prefix)
    embeddings = encode_corpus(model, texts, batch_size=batch_size)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, embeddings=embeddings, doc_ids=np.array(doc_ids, dtype=object))
    logger.info("Đã lưu cache embedding corpus tại %s", cache_path)
    return embeddings, doc_ids


def retrieve_top_k(
    question_embeddings,
    corpus_embeddings,
    corpus_doc_ids: List[str],
    top_k: int = 5,
    chunk_batch_size: int = 4000,
):
    """Với mỗi câu hỏi, tính similarity với TỪNG CHUNK rồi lấy điểm MAX theo từng document
    (một document có thể có nhiều chunk - lấy chunk nào giống câu hỏi nhất làm đại diện),
    cuối cùng trả về top-k document_id có điểm cao nhất.

    Cài đặt theo batch trên trục chunk để không phải tạo ma trận (n_question x n_chunk) khổng
    lồ cùng lúc trong RAM (corpus có thể lên tới hàng trăm nghìn chunk).
    """
    import numpy as np

    unique_doc_ids = sorted(set(corpus_doc_ids))
    doc_id_to_col = {d: i for i, d in enumerate(unique_doc_ids)}
    col_indices = np.array([doc_id_to_col[d] for d in corpus_doc_ids])

    n_questions = question_embeddings.shape[0]
    n_docs = len(unique_doc_ids)
    # doc_scores[q, d] = similarity cao nhất giữa câu hỏi q và bất kỳ chunk nào thuộc document d
    doc_scores = np.full((n_questions, n_docs), -np.inf, dtype=np.float32)

    n_chunks = corpus_embeddings.shape[0]
    for start in range(0, n_chunks, chunk_batch_size):
        end = min(start + chunk_batch_size, n_chunks)
        batch_emb = corpus_embeddings[start:end]  # (batch, dim)
        batch_cols = col_indices[start:end]  # (batch,)

        sims = question_embeddings @ batch_emb.T  # (n_questions, batch)

        # scatter-max: doc_scores[:, batch_cols[j]] = max(doc_scores[:, batch_cols[j]], sims[:, j])
        # dùng np.maximum.at trên view transpose để cập nhật đúng cả khi 1 document có nhiều
        # chunk trong CÙNG một batch (np.maximum.at xử lý index trùng lặp chính xác, không như
        # gán trực tiếp doc_scores[:, batch_cols] = ... sẽ bị ghi đè sai khi trùng index).
        doc_scores_T = doc_scores.T  # view, không copy
        np.maximum.at(doc_scores_T, batch_cols, sims.T)

    results = []
    for qi in range(n_questions):
        row = doc_scores[qi]
        top_idx = np.argsort(-row)[:top_k]
        results.append([unique_doc_ids[i] for i in top_idx if row[i] > -np.inf])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline retrieval brute-force (chỉ embedding) cho Task 1 - LegalIR")
    parser.add_argument("--model", required=True, help="vd: bqbbao6/VN_legal_embedding_512 hoặc đường dẫn model đã fine-tune local")
    parser.add_argument("--corpus", required=True, help="Đường dẫn selected-contexts.zip")
    parser.add_argument("--questions", required=True, help="train.json hoặc public-official.json")
    parser.add_argument("--output", required=True, help="Đường dẫn lưu file dự đoán (định dạng submission.json)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-cache", default=None, help="Nếu chỉ định, cache embedding corpus ra file .npz để tái sử dụng giữa các lần chạy")
    parser.add_argument("--limit-questions", type=int, default=None, help="Chỉ chạy N câu hỏi đầu (debug nhanh)")
    parser.add_argument("--device", default="auto", help="'auto' (mặc định, ưu tiên GPU nếu có), 'cuda', hoặc 'cpu'")
    parser.add_argument(
        "--query-prefix", default="query: ",
        help='Prefix thêm trước câu hỏi trước khi encode (E5-style, mặc định "query: "). Truyền "" để tắt nếu dùng model không cần prefix.',
    )
    parser.add_argument(
        "--passage-prefix", default="passage: ",
        help='Prefix thêm trước mỗi chunk corpus trước khi encode (E5-style, mặc định "passage: "). Truyền "" để tắt.',
    )
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("Thiếu sentence-transformers. Chạy: pip install sentence-transformers torch")
        sys.exit(1)

    device = resolve_device(args.device)
    logger.info("Load model: %s", args.model)
    model = SentenceTransformer(args.model, device=device)

    corpus = CorpusReader(args.corpus)
    logger.info("Corpus: %d văn bản", len(corpus))

    if args.query_prefix or args.passage_prefix:
        logger.info(
            'Dùng prefix E5-style: query_prefix=%r, passage_prefix=%r (đổi bằng --query-prefix/--passage-prefix nếu model khác không cần)',
            args.query_prefix, args.passage_prefix,
        )

    if args.embedding_cache:
        corpus_embeddings, corpus_doc_ids = load_or_build_corpus_embeddings(
            model, corpus, args.embedding_cache, args.max_chars, args.batch_size, passage_prefix=args.passage_prefix
        )
    else:
        texts, corpus_doc_ids = build_corpus_chunks(corpus, max_chars=args.max_chars, passage_prefix=args.passage_prefix)
        corpus_embeddings = encode_corpus(model, texts, batch_size=args.batch_size)

    with open(args.questions, encoding="utf-8") as f:
        raw_questions = json.load(f)
    qids = list(raw_questions.keys())
    if args.limit_questions:
        qids = qids[: args.limit_questions]
    question_texts = [args.query_prefix + raw_questions[qid]["question"] for qid in qids]

    logger.info("Encoding %d câu hỏi...", len(question_texts))
    question_embeddings = model.encode(
        question_texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )

    logger.info("Truy xuất top-%d cho từng câu hỏi...", args.top_k)
    top_k_results = retrieve_top_k(question_embeddings, corpus_embeddings, corpus_doc_ids, top_k=args.top_k)

    predictions: Dict[str, dict] = {
        qid: {"answer": docs} for qid, docs in zip(qids, top_k_results)
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    logger.info("Đã lưu %d dự đoán tại: %s", len(predictions), args.output)
    logger.info(
        "Bước tiếp theo: python -m src.evaluation.evaluate --ground-truth %s --predictions %s",
        args.questions, args.output,
    )


if __name__ == "__main__":
    main()
