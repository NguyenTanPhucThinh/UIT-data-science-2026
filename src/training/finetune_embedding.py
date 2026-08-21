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

QUAN TRỌNG - prefix "query: "/"passage: ": bqbbao6/VN_legal_embedding_512 fine-tune từ
intfloat/multilingual-e5-base (đã xác nhận qua model card HuggingFace) - là bi-encoder BẤT ĐỐI
XỨNG kiểu E5, BẮT BUỘC thêm "query: " trước câu hỏi và "passage: " trước văn bản trước khi
encode/train (cả lúc TRAIN lẫn lúc INFERENCE phải nhất quán prefix, nếu không model học/predict
sai). Mặc định script này tự thêm 2 prefix trên cho mọi cặp training (đổi bằng
--query-prefix/--passage-prefix, truyền "" để tắt nếu đổi sang model không cần prefix).
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from .data_utils import CorpusReader, chunk_passage, load_labeled_questions, resolve_device, select_best_chunk
except ImportError:  # cho phép chạy trực tiếp: python src/training/finetune_embedding.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.training.data_utils import CorpusReader, chunk_passage, load_labeled_questions, resolve_device, select_best_chunk


def build_positive_pairs(
    train_json_path: str,
    corpus_zip_path: str,
    max_chars: int = 2000,
    query_prefix: str = "",
    passage_prefix: str = "",
):
    """Với mỗi câu hỏi, ghép với đoạn văn bản (chunk) khả năng cao nhất chứa câu trả lời,
    trong (các) document mà BTC gán là đúng.

    BTC chỉ gán nhãn ở cấp document, không chỉ rõ đoạn/Điều nào - nên dùng
    `select_best_chunk` (word-overlap heuristic) để chọn chunk hợp lý nhất thay vì lấy đại
    chunk đầu tiên (thường chỉ là quốc hiệu/tiêu đề văn bản, không chứa nội dung trả lời).

    `query_prefix`/`passage_prefix` (vd "query: "/"passage: " cho model E5-style) chỉ được thêm
    vào SAU khi đã chọn xong chunk bằng `select_best_chunk` (dùng text gốc để so khớp từ cho
    chính xác) - text trả về trong `pairs` đã có prefix, sẵn sàng đưa thẳng vào model.fit().
    """
    questions = load_labeled_questions(train_json_path)
    corpus = CorpusReader(corpus_zip_path)

    pairs = []  # (question_text, positive_passage, question_id, doc_id) - đã có prefix
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
            pairs.append((query_prefix + question, passage_prefix + positive, qid, doc_id))

    if missing_docs:
        logger.warning("Bỏ qua %d cặp vì document_id không tồn tại trong corpus", missing_docs)
    logger.info("Tạo được %d cặp (question, positive_passage) từ %d câu hỏi", len(pairs), len(questions))
    return pairs


def mine_hard_negatives(
    model,
    pairs,
    corpus: "CorpusReader",
    top_k: int = 5,
    pool_size: int = 2000,
    max_chars: int = 2000,
    passage_prefix: str = "",
):
    """Dùng chính model (trước fine-tune thêm) để tìm hard negative: văn bản có similarity cao
    với câu hỏi nhưng KHÔNG phải document đúng - giúp model học phân biệt tinh hơn so với chỉ
    dùng in-batch negatives ngẫu nhiên.

    Vì corpus có 8.532 văn bản (nhiều văn bản rất dài), để tiết kiệm thời gian ta lấy ngẫu nhiên
    một `pool_size` văn bản làm "ứng viên nhiễu" thay vì encode toàn bộ corpus mỗi lần gọi. Muốn
    hard negative chất lượng cao hơn (mine trên toàn bộ corpus + dùng index đã build sẵn), nên
    thay phần này bằng cách gọi thẳng vectorstore của nhóm (src/vectorstore) khi module đó xong.

    `pairs` ở đây đã được `build_positive_pairs` thêm prefix sẵn (question có query_prefix) -
    nên chỉ cần thêm `passage_prefix` cho `pool_texts` (ứng viên negative) để nhất quán.
    """
    import numpy as np

    all_ids = corpus.ids()
    pool_ids = random.sample(all_ids, min(pool_size, len(all_ids)))
    pool_texts, pool_doc_ids = [], []
    for doc_id in pool_ids:
        doc = corpus.get(doc_id)
        chunks = chunk_passage(doc.passage, max_chars=max_chars)
        if chunks:
            pool_texts.append(passage_prefix + chunks[0])
            pool_doc_ids.append(doc_id)

    logger.info("Encoding %d văn bản trong pool để mine hard negatives...", len(pool_texts))
    pool_emb = model.encode(pool_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    questions = [p[0] for p in pairs]  # đã có query_prefix từ build_positive_pairs
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
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Giới hạn số step mỗi epoch; hữu ích cho smoke test GPU, bỏ trống để train đầy đủ",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Dùng automatic mixed precision trên CUDA để giảm VRAM và tăng tốc",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Thư mục lưu checkpoint trung gian để có thể resume khi bị gián đoạn",
    )
    parser.add_argument("--checkpoint-save-steps", type=int, default=100)
    parser.add_argument("--checkpoint-save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument("--max-chars", type=int, default=2000, help="Độ dài tối đa (ký tự) mỗi chunk dùng để train")
    parser.add_argument("--hard-negatives", action="store_true", help="Bật hard negative mining (TripletLoss thay vì MultipleNegativesRankingLoss)")
    parser.add_argument("--hard-negative-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="'auto' (mặc định, ưu tiên GPU nếu có), 'cuda', hoặc 'cpu'")
    parser.add_argument(
        "--query-prefix", default="query: ",
        help='Prefix thêm trước câu hỏi khi train (E5-style, mặc định "query: "). Truyền "" để tắt nếu dùng model không cần prefix.',
    )
    parser.add_argument(
        "--passage-prefix", default="passage: ",
        help='Prefix thêm trước văn bản/passage khi train (E5-style, mặc định "passage: "). Truyền "" để tắt.',
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=None,
        help="Ghi đè model.max_seq_length SAU khi load base model, TRƯỚC khi train (thêm 2026-08-14, "
             "cho model không phải E5-style, vd AITeamVN/Vietnamese_Embedding khuyến nghị 2048 theo "
             "model card). QUAN TRỌNG: phải dùng CÙNG giá trị này khi encode/inference sau này "
             "(--max-seq-length của full_pipeline_eval.py) - lệch giá trị giữa train và inference có "
             "thể làm giảm chất lượng model.",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Bật mixed-precision training (use_amp=True trong model.fit(), thêm 2026-08-15). "
             "sentence-transformers MẶC ĐỊNH train fp32 (use_amp=False) - với model lớn hơn "
             "(vd AITeamVN/Vietnamese_Embedding ~568M, kiến trúc XLM-R-large 24 lớp, so với "
             "bqbbao6 ~278M base 12 lớp) cộng --max-seq-length dài (2048 so với ~512 trước đây, "
             "chi phí attention tăng theo BÌNH PHƯƠNG độ dài chuỗi), train fp32 có thể chậm hơn "
             "đáng kể. Bật cờ này để dùng fp16 autocast, thường nhanh hơn ~1.5-2x trên GPU NVIDIA "
             "đời mới (Turing/Ampere trở lên, có Tensor Core) mà chất lượng gần như không đổi. "
             "Không bật mặc định để không đổi hành vi cũ (bqbbao6 đã train ổn ở fp32).",
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, default=200,
        help="Lưu checkpoint (model tạm) mỗi N step vào <output-dir>-checkpoints (thêm 2026-08-15). "
             "sentence-transformers MẶC ĐỊNH KHÔNG lưu checkpoint giữa chừng (chỉ lưu model cuối cùng "
             "SAU KHI chạy xong toàn bộ epochs) - với những lần chạy dài (vài tiếng trở lên, đặc biệt "
             "model lớn + --max-seq-length dài dễ gây tràn VRAM giữa chừng nếu 1 batch ngẫu nhiên chứa "
             "chunk gần độ dài tối đa), nếu crash (OOM, mất điện, lỗi driver...) ở step 1000/1146 sẽ MẤT "
             "TRẮNG toàn bộ tiến độ, phải chạy lại từ đầu. Truyền 0 để tắt hẳn checkpoint (giống hành vi "
             "cũ trước 2026-08-15).",
    )
    parser.add_argument(
        "--checkpoint-keep", type=int, default=3,
        help="Giữ lại tối đa N checkpoint gần nhất (xoá bớt checkpoint cũ để đỡ tốn ổ đĩa), mặc định 3.",
    )
    parser.add_argument(
        "--resume-from-checkpoint", action="store_true",
        help="Tiếp tục train từ checkpoint gần nhất trong <output-dir>-checkpoints nếu có (thêm "
             "2026-08-15) - dùng khi lần chạy trước bị crash/dừng giữa chừng, tránh phải train lại "
             "từ đầu.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    try:
        from sentence_transformers import InputExample, SentenceTransformer
        from sentence_transformers.sentence_transformer import losses
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error(
            "Thiếu sentence-transformers/torch. Chạy: pip install sentence-transformers torch\n"
            "(Khuyến nghị chạy script này trên máy có GPU, vd RTX 3060, để train nhanh hơn CPU nhiều.)"
        )
        sys.exit(1)

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested but CUDA is unavailable. Install a CUDA-enabled "
                "PyTorch wheel and verify torch.cuda.is_available() first."
            )
    device = resolve_device(args.device)
    logger.info("Load base model: %s", args.base_model)
    model = SentenceTransformer(args.base_model, device=device)
    if args.max_seq_length is not None:
        model.max_seq_length = args.max_seq_length
        logger.info("Đã ghi đè model.max_seq_length = %d (áp dụng cho cả train lẫn hard-negative mining)", args.max_seq_length)
    logger.info("Training device resolved to: %s", model.device)

    logger.info(
        'Dùng prefix E5-style: query_prefix=%r, passage_prefix=%r (đổi bằng --query-prefix/--passage-prefix nếu model khác không cần)',
        args.query_prefix, args.passage_prefix,
    )
    pairs = build_positive_pairs(
        args.train_json, args.corpus, max_chars=args.max_chars,
        query_prefix=args.query_prefix, passage_prefix=args.passage_prefix,
    )
    if not pairs:
        logger.error("Không tạo được cặp training nào, kiểm tra lại train.json/corpus")
        sys.exit(1)

    if args.hard_negatives:
        corpus = CorpusReader(args.corpus)
        triplets = mine_hard_negatives(
            model, pairs, corpus, top_k=args.hard_negative_top_k, max_chars=args.max_chars,
            passage_prefix=args.passage_prefix,
        )
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
        steps_per_epoch=args.steps_per_epoch,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        use_amp=args.use_amp,
        checkpoint_path=args.checkpoint_path,
        checkpoint_save_steps=args.checkpoint_save_steps,
        checkpoint_save_total_limit=args.checkpoint_save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        show_progress_bar=True,
    )

    logger.info("Đã lưu model fine-tune tại: %s", args.output_dir)
    logger.info(
        "Bước tiếp theo: chạy lại retrieval pipeline với model này, rồi dùng "
        "src/evaluation/evaluate.py để so sánh Recall/Precision với model gốc."
    )


if __name__ == "__main__":
    main()
