"""
Fine-tune embedding model cho Task 1 - LegalIR bằng contrastive learning
(MultipleNegativesRankingLoss), có tuỳ chọn hard negative mining (TripletLoss).

TUÂN THỦ QUY ĐỊNH CUỘC THI:
  - Chỉ dùng dữ liệu chính thức BTC cung cấp (train.json + selected-contexts.zip), KHÔNG thêm
    dữ liệu ngoài, KHÔNG gán nhãn thủ công thêm.
  - Model base phải mã nguồn mở, tự tải checkpoint về máy rồi chạy local (hợp lệ) - không phải
    "API" bị cấm. Sau khi fine-tune, số tham số KHÔNG đổi so với base.
  - Nhớ verify lại tổng tham số toàn pipeline (embedding + reranker + ...) vẫn < 4 tỷ.

Cách dùng cơ bản (baseline, in-batch negatives):
    python -m src.training.finetune_embedding \\
        --base-model bqbbao6/VN_legal_embedding_512 \\
        --train-json data/raw/train.json \\
        --corpus data/raw/selected-contexts.zip \\
        --output-dir models/vietnamese-legal-embedding-ft \\
        --epochs 3 --batch-size 16

Bật hard negative mining (khuyến nghị khi đã có baseline chạy ổn, để cải thiện thêm):
    python -m src.training.finetune_embedding ... --hard-negatives

Yêu cầu: pip install sentence-transformers torch (xem requirements.txt).
Chạy tốt trên GPU RTX 3060 (6-12GB) - embedding model cỡ vài trăm triệu tham số không cần VRAM lớn.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from .data_utils import CorpusReader, chunk_passage, load_labeled_questions, resolve_device, select_best_chunk
except ImportError:  # cho phép chạy trực tiếp: python src/training/finetune_embedding.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.training.data_utils import CorpusReader, chunk_passage, load_labeled_questions, resolve_device, select_best_chunk


def build_positive_pairs(train_json_path: str, corpus_zip_path: str, max_chars: int = 2000):
    """Với mỗi câu hỏi, ghép với đoạn văn bản (chunk) khả năng cao nhất chứa câu trả lời,
    trong (các) document mà BTC gán là đúng.

    BTC chỉ gán nhãn ở cấp document, không chỉ rõ đoạn/Điều nào - nên dùng
    `select_best_chunk` (word-overlap heuristic) để chọn chunk hợp lý nhất thay vì lấy đại
    chunk đầu tiên (thường chỉ là quốc hiệu/tiêu đề văn bản, không chứa nội dung trả lời).
    """
    questions = load_labeled_questions(train_json_path)
    corpus = CorpusReader(corpus_zip_path)

    pairs = []  # (question_text, positive_passage, question_id, doc_id)
    missing_docs = 0
    for qid, item in questions.items():
        question = item["question"]
        for doc_id in item["answer"]:
            try:
                doc = corpus.get(doc_id)
            except KeyError:
                missing_docs += 1
                continue
            chunks = chunk_passage(doc.passage, max_chars=max_chars)
            if not chunks:
                continue
            positive = select_best_chunk(question, chunks)
            pairs.append((question, positive, qid, doc_id))

    if missing_docs:
        logger.warning("Bỏ qua %d cặp vì document_id không tồn tại trong corpus", missing_docs)
    logger.info("Tạo được %d cặp (question, positive_passage) từ %d câu hỏi", len(pairs), len(questions))
    return pairs


def mine_hard_negatives(model, pairs, corpus: "CorpusReader", top_k: int = 5, pool_size: int = 2000, max_chars: int = 2000):
    """Dùng chính model (trước fine-tune thêm) để tìm hard negative: văn bản có similarity cao
    với câu hỏi nhưng KHÔNG phải document đúng - giúp model học phân biệt tinh hơn so với chỉ
    dùng in-batch negatives ngẫu nhiên.

    Vì corpus có 8.532 văn bản (nhiều văn bản rất dài), để tiết kiệm thời gian ta lấy ngẫu nhiên
    một `pool_size` văn bản làm "ứng viên nhiễu" thay vì encode toàn bộ corpus mỗi lần gọi. Muốn
    hard negative chất lượng cao hơn (mine trên toàn bộ corpus + dùng index đã build sẵn), nên
    thay phần này bằng cách gọi thẳng vectorstore của nhóm (src/vectorstore) khi module đó xong.
    """
    import numpy as np

    all_ids = corpus.ids()
    pool_ids = random.sample(all_ids, min(pool_size, len(all_ids)))
    pool_texts, pool_doc_ids = [], []
    for doc_id in pool_ids:
        doc = corpus.get(doc_id)
        chunks = chunk_passage(doc.passage, max_chars=max_chars)
        if chunks:
            pool_texts.append(chunks[0])
            pool_doc_ids.append(doc_id)

    logger.info("Encoding %d văn bản trong pool để mine hard negatives...", len(pool_texts))
    pool_emb = model.encode(pool_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    questions = [p[0] for p in pairs]
    q_emb = model.encode(questions, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    sims = q_emb @ pool_emb.T  # (n_questions, pool_size)

    triplets = []
    for i, (question, positive, qid, pos_doc_id) in enumerate(pairs):
        top_idx = np.argsort(-sims[i])[: top_k + 1]
        negs = [pool_doc_ids[j] for j in top_idx if pool_doc_ids[j] != pos_doc_id][:top_k]
        for neg_doc_id in negs:
            neg_text = pool_texts[pool_doc_ids.index(neg_doc_id)]
            triplets.append((question, positive, neg_text))

    logger.info("Tạo được %d triplet (question, positive, hard_negative)", len(triplets))
    return triplets


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune embedding model cho Task 1 - LegalIR")
    parser.add_argument("--base-model", required=True, help="vd: bqbbao6/VN_legal_embedding_512")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--corpus", required=True, help="Đường dẫn selected-contexts.zip")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=2000, help="Độ dài tối đa (ký tự) mỗi chunk dùng để train")
    parser.add_argument("--hard-negatives", action="store_true", help="Bật hard negative mining (TripletLoss thay vì MultipleNegativesRankingLoss)")
    parser.add_argument("--hard-negative-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="'auto' (mặc định, ưu tiên GPU nếu có), 'cuda', hoặc 'cpu'")
    args = parser.parse_args()

    random.seed(args.seed)

    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error(
            "Thiếu sentence-transformers/torch. Chạy: pip install sentence-transformers torch\n"
            "(Khuyến nghị chạy script này trên máy có GPU, vd RTX 3060, để train nhanh hơn CPU nhiều.)"
        )
        sys.exit(1)

    device = resolve_device(args.device)
    logger.info("Load base model: %s", args.base_model)
    model = SentenceTransformer(args.base_model, device=device)

    pairs = build_positive_pairs(args.train_json, args.corpus, max_chars=args.max_chars)
    if not pairs:
        logger.error("Không tạo được cặp training nào, kiểm tra lại train.json/corpus")
        sys.exit(1)

    if args.hard_negatives:
        corpus = CorpusReader(args.corpus)
        triplets = mine_hard_negatives(model, pairs, corpus, top_k=args.hard_negative_top_k, max_chars=args.max_chars)
        train_examples = [InputExample(texts=[q, pos, neg]) for q, pos, neg in triplets]
        loss = losses.TripletLoss(model=model)
        logger.info("Dùng TripletLoss với %d triplet (có hard negative)", len(train_examples))
    else:
        train_examples = [InputExample(texts=[q, pos]) for q, pos, _, _ in pairs]
        loss = losses.MultipleNegativesRankingLoss(model=model)
        logger.info("Dùng MultipleNegativesRankingLoss với %d cặp (in-batch negatives)", len(train_examples))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)
    logger.info("Bắt đầu fine-tune: %d epoch, batch_size=%d, warmup_steps=%d", args.epochs, args.batch_size, warmup_steps)

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        show_progress_bar=True,
    )

    logger.info("Đã lưu model fine-tune tại: %s", args.output_dir)
    logger.info(
        "Bước tiếp theo: chạy lại retrieval pipeline với model này, rồi dùng "
        "src/evaluation/evaluate.py để so sánh Recall/Precision với model gốc."
    )


if __name__ == "__main__":
    main()
