"""
Fine-tune cross-encoder reranker cho Task 1 - LegalIR.

Reranker nhận đầu vào (câu hỏi, đoạn văn bản) và trả về điểm liên quan (0-1), dùng để sắp xếp
lại top-N ứng viên do bước retrieval (BM25 + dense embedding) trả về - đây là bước cuối trong
kiến trúc hybrid retrieval đã đề xuất (docs/de-xuat-pipeline-task1.md / docs/tong-quan-du-lieu-task1.md).

TUÂN THỦ QUY ĐỊNH CUỘC THI (giống finetune_embedding.py):
  - Chỉ dùng dữ liệu chính thức BTC cung cấp (train.json + selected-contexts.zip), KHÔNG thêm
    dữ liệu ngoài, KHÔNG gán nhãn thủ công thêm.
  - Model base phải mã nguồn mở, tải checkpoint về máy rồi chạy local (hợp lệ, không phải "API"
    bị cấm). Fine-tune KHÔNG làm thay đổi số tham số model.
  - Nhớ verify lại tổng tham số toàn pipeline (embedding + reranker + LLM nếu có) vẫn < 4 tỷ
    SAU KHI đã chọn model reranker cuối cùng.

Cách dùng cơ bản (negative ngẫu nhiên - nhanh, làm baseline trước):
    python -m src.training.finetune_reranker \\
        --base-model BAAI/bge-reranker-v2-m3 \\
        --train-json data/raw/train.json \\
        --corpus data/raw/selected-contexts.zip \\
        --output-dir models/reranker-ft \\
        --epochs 2 --batch-size 16

Bật hard negative mining (khuyến nghị - reranker học phân biệt "gần đúng nhưng sai" mới có ích,
negative ngẫu nhiên thường quá dễ, model không học được gì nhiều):
    python -m src.training.finetune_reranker ... \\
        --hard-negatives --hard-negative-embedding-model bqbbao6/VN_legal_embedding_512

`--hard-negative-embedding-model`: model EMBEDDING (không phải reranker) dùng riêng để mine hard
negative (tìm văn bản "giống câu hỏi" theo embedding nhưng KHÔNG phải đáp án đúng). Có thể trỏ
tới model embedding đã fine-tune (models/vietnamese-legal-embedding-ft) để có hard negative sát
với thực tế pipeline retrieval của nhóm hơn. Việc mine dùng embedding model (rẻ, encode 1 lần)
thay vì chính reranker (đắt, phải chạy pairwise) để tiết kiệm thời gian.

Yêu cầu: pip install sentence-transformers torch (xem requirements.txt).
Chạy tốt trên GPU RTX 3060 (6-12GB) - reranker cỡ vài trăm triệu tham số không cần VRAM lớn,
nhưng train reranker thường CHẬM hơn embedding vì mỗi cặp (câu hỏi, đoạn văn) phải qua model
CÙNG NHAU (cross-attention) thay vì encode riêng - batch_size nhỏ hơn nếu gặp out-of-memory.
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
    from .finetune_embedding import build_positive_pairs, mine_hard_negatives
except ImportError:  # cho phép chạy trực tiếp: python src/training/finetune_reranker.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.training.data_utils import CorpusReader, chunk_passage, load_labeled_questions, resolve_device, select_best_chunk
    from src.training.finetune_embedding import build_positive_pairs, mine_hard_negatives


def build_random_negative_pairs(pairs, corpus: "CorpusReader", max_chars: int = 2000, n_per_positive: int = 1, seed: int = 42):
    """Với mỗi cặp (question, positive), ghép thêm n_per_positive negative NGẪU NHIÊN (văn bản
    bất kỳ trong corpus không nằm trong answer thật của câu hỏi đó).

    Đơn giản, nhanh, không cần model nào cả - dùng làm baseline trước khi thử hard negative.
    Nhược điểm: negative ngẫu nhiên thường quá khác biệt câu hỏi, dễ đoán, reranker học được ít.
    """
    rng = random.Random(seed)
    all_ids = corpus.ids()

    # answer thật theo từng câu hỏi, để không lỡ chọn trúng document đúng làm negative
    answers_by_qid: dict = {}
    for question, positive, qid, pos_doc_id in pairs:
        answers_by_qid.setdefault(qid, set()).add(str(pos_doc_id))

    negative_pairs = []  # (question, negative_text)
    for question, positive, qid, pos_doc_id in pairs:
        answer_ids = answers_by_qid[qid]
        for _ in range(n_per_positive):
            neg_doc_id = rng.choice(all_ids)
            tries = 0
            while neg_doc_id in answer_ids and tries < 10:
                neg_doc_id = rng.choice(all_ids)
                tries += 1
            if neg_doc_id in answer_ids:
                continue
            doc = corpus.get(neg_doc_id)
            chunks = chunk_passage(doc.passage, max_chars=max_chars)
            if not chunks:
                continue
            negative_pairs.append((question, select_best_chunk(question, chunks)))

    logger.info("Tạo được %d cặp negative ngẫu nhiên (%d negative/positive)", len(negative_pairs), n_per_positive)
    return negative_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder reranker cho Task 1 - LegalIR")
    parser.add_argument("--base-model", required=True, help="vd: BAAI/bge-reranker-v2-m3")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--corpus", required=True, help="Đường dẫn selected-contexts.zip")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=2000, help="Độ dài tối đa (ký tự) mỗi chunk dùng để train")
    parser.add_argument("--hard-negatives", action="store_true", help="Bật hard negative mining thay vì negative ngẫu nhiên")
    parser.add_argument(
        "--hard-negative-embedding-model",
        default=None,
        help="Model embedding dùng để mine hard negative (bắt buộc nếu --hard-negatives). "
        "Có thể trỏ tới model embedding đã fine-tune của nhóm.",
    )
    parser.add_argument("--hard-negative-top-k", type=int, default=3, help="Số hard negative lấy cho mỗi câu hỏi")
    parser.add_argument("--n-random-negatives", type=int, default=1, help="Số negative ngẫu nhiên cho mỗi positive (khi KHÔNG dùng --hard-negatives)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="'auto' (mặc định, ưu tiên GPU nếu có), 'cuda', hoặc 'cpu'")
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="Bật mixed-precision training (mặc định BẬT - nhanh hơn 2-4 lần trên GPU hỗ trợ, "
        "vd RTX 3060). CrossEncoder nặng hơn embedding model nhiều (ghép câu hỏi+passage chung "
        "1 chuỗi, chạy qua toàn bộ cross-attention), fp32 mặc định của sentence-transformers "
        "có thể khiến 1 epoch mất hàng chục giờ.",
    )
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Tắt fp16 nếu gặp lỗi/NaN loss")
    parser.add_argument(
        "--bf16", action="store_true", default=True,
        help="Load model ở bfloat16 thay vì fp32 (mặc định BẬT - giảm ~1/2 compute+VRAM, ổn định "
        "hơn fp16). Tự fallback về fp32 nếu bản thư viện không hỗ trợ.",
    )
    parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="Tắt bf16, load model ở fp32")
    args = parser.parse_args()

    if args.hard_negatives and not args.hard_negative_embedding_model:
        logger.error("--hard-negatives cần chỉ định --hard-negative-embedding-model")
        sys.exit(1)

    random.seed(args.seed)

    try:
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error(
            "Thiếu sentence-transformers/torch. Chạy: pip install sentence-transformers torch\n"
            "(Khuyến nghị chạy script này trên máy có GPU, vd RTX 3060, để train nhanh hơn CPU nhiều.)"
        )
        sys.exit(1)

    device = resolve_device(args.device)

    if device == "cuda":
        # TF32: bật tensor core cho phép tính fp32 trên GPU Ampere+ (RTX 3060 trở lên) - nhanh
        # hơn đáng kể (thường 1.5-3 lần) so với fp32 thuần, KHÔNG phụ thuộc API fp16 của
        # sentence-transformers (vốn có thể không được hỗ trợ tùy version, như đã gặp ở trên).
        # Đây là cờ backend của chính torch, luôn có sẵn từ torch>=1.7, không cần cài thêm gì.
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Đã bật TF32 (torch.backends.cuda/cudnn.allow_tf32) để tăng tốc training trên GPU Ampere+.")

    pairs = build_positive_pairs(args.train_json, args.corpus, max_chars=args.max_chars)
    if not pairs:
        logger.error("Không tạo được cặp training nào, kiểm tra lại train.json/corpus")
        sys.exit(1)

    corpus = CorpusReader(args.corpus)

    if args.hard_negatives:
        logger.info("Load embedding model để mine hard negative: %s", args.hard_negative_embedding_model)
        from sentence_transformers import SentenceTransformer

        embed_model = SentenceTransformer(args.hard_negative_embedding_model, device=device)
        triplets = mine_hard_negatives(
            embed_model, pairs, corpus, top_k=args.hard_negative_top_k, max_chars=args.max_chars
        )
        train_examples = []
        for question, positive, negative in triplets:
            train_examples.append(InputExample(texts=[question, positive], label=1.0))
            train_examples.append(InputExample(texts=[question, negative], label=0.0))
        logger.info("Dùng %d ví dụ (hard negative) - %d positive + %d negative", len(train_examples), len(triplets), len(triplets))
    else:
        negative_pairs = build_random_negative_pairs(
            pairs, corpus, max_chars=args.max_chars, n_per_positive=args.n_random_negatives, seed=args.seed
        )
        train_examples = [InputExample(texts=[q, pos], label=1.0) for q, pos, _, _ in pairs]
        train_examples += [InputExample(texts=[q, neg], label=0.0) for q, neg in negative_pairs]
        logger.info(
            "Dùng %d ví dụ (negative ngẫu nhiên) - %d positive + %d negative",
            len(train_examples), len(pairs), len(negative_pairs),
        )

    random.shuffle(train_examples)

    logger.info("Load base reranker: %s", args.base_model)
    model = None
    bf16_applied = False
    if device == "cuda" and args.bf16:
        # Load thẳng weight ở bf16 (thay vì fp32) - giảm ~1/2 compute + VRAM, KHÔNG cần loss
        # scaling như fp16 (bf16 giữ nguyên dải exponent của fp32 nên ổn định hơn khi train).
        # Tên tham số truyền model dtype khác nhau tùy version sentence-transformers/transformers
        # nên thử lần lượt, cái nào không nhận (TypeError) thì thử cái tiếp theo.
        import torch

        for kwarg_name in ("model_kwargs", "automodel_args"):
            try:
                model = CrossEncoder(
                    args.base_model, num_labels=1, device=device,
                    **{kwarg_name: {"torch_dtype": torch.bfloat16}},
                )
                logger.info("Load model ở bf16 thành công (tham số %s=...).", kwarg_name)
                bf16_applied = True
                break
            except TypeError:
                continue
        if model is None:
            logger.warning(
                "Bản sentence-transformers/transformers hiện tại không hỗ trợ load model ở bf16 "
                "qua model_kwargs/automodel_args - load bình thường ở fp32 (chậm hơn, dùng "
                "--no-bf16 để tắt cảnh báo này)."
            )
    if model is None:
        model = CrossEncoder(args.base_model, num_labels=1, device=device)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)
    # Không bật fp16 (AMP) CHỒNG lên model đã load bf16 - dễ lỗi/không nhất quán dtype. bf16 đã
    # tự giảm compute rồi nên không cần fp16 thêm nữa.
    try_fp16 = args.fp16 and not bf16_applied
    logger.info(
        "Bắt đầu fine-tune: %d epoch, batch_size=%d, warmup_steps=%d, bf16=%s, fp16=%s",
        args.epochs, args.batch_size, warmup_steps, bf16_applied, try_fp16,
    )

    fit_kwargs = dict(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        show_progress_bar=True,
    )
    if try_fp16:
        try:
            model.fit(**fit_kwargs, fp16=True)
        except TypeError:
            # Bản sentence-transformers đang cài không nhận tham số fp16 qua fit() (API khác
            # version) - train ở fp32 thay vì crash, chỉ chậm hơn chứ không sai kết quả.
            logger.warning(
                "Bản sentence-transformers hiện tại không nhận tham số fp16 qua CrossEncoder.fit() "
                "- train ở fp32/bf16 hiện có (chậm hơn nhưng vẫn đúng)."
            )
            model.fit(**fit_kwargs)
    else:
        model.fit(**fit_kwargs)

    # QUAN TRỌNG: gọi lưu tường minh, KHÔNG chỉ tin fit(output_path=...) tự lưu cuối cùng - bản
    # sentence-transformers mới (Trainer-based bên trong) có thể không auto-save ở output_path
    # như API cũ nữa (đã gặp trường hợp train xong 1h45p nhưng models/reranker-ft trống rỗng,
    # KHÔNG báo lỗi gì - fit() chạy xong bình thường, chỉ là không có file nào được ghi ra đĩa).
    model.save(args.output_dir)
    if not (Path(args.output_dir) / "config.json").exists():
        logger.error(
            "CẢNH BÁO: đã gọi model.save(%s) nhưng vẫn KHÔNG thấy config.json - kiểm tra lại "
            "quyền ghi thư mục / dung lượng đĩa trống trước khi train lại.", args.output_dir,
        )
        sys.exit(1)
    logger.info("Đã lưu reranker fine-tune tại: %s", args.output_dir)
    logger.info(
        "Bước tiếp theo: chạy lại pipeline retrieval + rerank với model này, rồi dùng "
        "src/evaluation/evaluate.py để so sánh Recall/Precision với reranker gốc (pretrained)."
    )


if __name__ == "__main__":
    main()
