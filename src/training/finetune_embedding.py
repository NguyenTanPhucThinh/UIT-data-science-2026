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
import json  # dùng bởi mine_hard_negatives_from_ranks() (thêm 2026-08-22)
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


def mine_hard_negatives_from_ranks(
    pairs,
    gold_by_qid,
    ranks_path: str,
    chunks_path: str,
    top_k: int = 5,
    max_chars: int = 2000,
    passage_prefix: str = "",
):
    """Hard negative lấy TỪ CHÍNH THỨ HẠNG CỦA RETRIEVER THẬT (thêm 2026-08-22) — chất lượng cao hơn
    hẳn `mine_hard_negatives()` ở trên, và KHÔNG tốn GPU vì tái dùng file thứ hạng đã sinh sẵn.

    Vì sao cần: bản `mine_hard_negatives()` cũ chỉ lấy mẫu NGẪU NHIÊN `pool_size=2000` trên 8.532 văn
    bản (23%) và chỉ dùng CHUNK ĐẦU TIÊN mỗi văn bản — negative vừa ít vừa dễ, model học được rất ít
    (chính docstring hàm đó đã ghi nhận là bản tạm). Đây có thể là nửa còn lại của lý do dòng #12
    fine-tune thất bại (nửa kia là learning rate 2e-5 x 3 epoch, xem `--learning-rate`).

    Cách làm: `ranks_path` (.npz sinh bởi `src/evaluation/ensemble_fusion.py rank`) chứa top-N chunk mà
    retriever ĐANG DÙNG THẬT xếp cho ĐÚNG các câu hỏi train này. Lấy các chunk xếp hạng cao NHƯNG
    thuộc document KHÔNG nằm trong đáp án -> đó chính là những lỗi thật retriever đang mắc, tức
    negative khó nhất và sát thực tế nhất có thể có.
    """
    import numpy as np

    d = np.load(ranks_path, allow_pickle=False)
    qids = [str(q) for q in d["question_ids"]]
    ranks = d["ranks"]
    row_of = {q: i for i, q in enumerate(qids)}

    logger.info("Đọc chunk từ %s để lấy text negative...", chunks_path)
    chunk_doc, chunk_text = [], []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunk_doc.append(str(obj["doc_id"]))
            chunk_text.append(obj["passage"])
    logger.info("Đã đọc %d chunk", len(chunk_doc))

    triplets = []
    missing = 0
    for question, positive, qid, _pos_doc_id in pairs:
        r = row_of.get(str(qid))
        if r is None:
            missing += 1
            continue
        gold = gold_by_qid.get(str(qid), set())
        n = 0
        for ci in ranks[r]:
            ci = int(ci)
            if chunk_doc[ci] in gold:
                continue  # BỎ QUA mọi chunk thuộc document ĐÚNG (kể cả chunk khác của cùng doc)
            triplets.append((question, positive, passage_prefix + chunk_text[ci][:max_chars]))
            n += 1
            if n >= top_k:
                break

    if missing:
        logger.warning(
            "%d cặp không tìm thấy question_id trong %s — bỏ qua. Kiểm tra file thứ hạng có được "
            "sinh ĐÚNG trên tập câu hỏi đang train hay không.", missing, ranks_path,
        )
    logger.info("Tạo được %d triplet từ thứ hạng retriever thật (top_k=%d/câu hỏi)", len(triplets), top_k)
    return triplets


def _model_is_normalized(model) -> bool:
    """Model có module Normalize không (embedding nằm trên mặt cầu đơn vị)? Dùng để chặn cấu hình
    TripletLoss margin > 2 vốn KHÔNG BAO GIỜ thoả mãn được — xem cảnh báo trong main()."""
    try:
        return any(type(m).__name__ == "Normalize" for m in model._modules.values())
    except Exception:  # noqa: BLE001
        return False


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
    # ⚠️ ĐÃ XOÁ 1 bản `--resume-from-checkpoint` TRÙNG ở đây (2026-08-22). Bản còn lại nằm ở cuối
    # hàm (có help đầy đủ). Trùng option string làm argparse ném ArgumentError NGAY khi dựng parser
    # -> script CRASH ở MỌI lệnh gọi, không chạy được lần nào kể từ lần merge 2026-08-18. Đây là bug
    # THỨ 4 cùng gốc merge conflict đó (3 bug trước ở full_pipeline_eval.py).
    # BÀI HỌC: `python -m py_compile` KHÔNG bắt được lỗi này (chỉ kiểm tra cú pháp, không dựng
    # parser). Muốn kiểm tra thật phải chạy `python -m <module> --help`.
    parser.add_argument("--max-chars", type=int, default=2000, help="Độ dài tối đa (ký tự) mỗi chunk dùng để train")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5,
        help="Learning rate cho AdamW (thêm 2026-08-22). TRƯỚC ĐÂY KHÔNG có tham số này: model.fit() "
             "dùng mặc định 2e-5 của sentence-transformers. Với model đã train RẤT KỸ từ đầu (vd "
             "AITeamVN/Vietnamese_Embedding: 568M tham số, 1,1 TRIỆU triplet) mà fine-tune tiếp trên "
             "tập nhỏ (~7-8 nghìn cặp), 2e-5 x 3 epoch là công thức gây CATASTROPHIC FORGETTING - đúng "
             "điều đã xảy ra ở dòng #12 (Score PUBLIC 0.8922, THẤP HƠN bản pretrained 0.9058). "
             "Khuyến nghị fine-tune NHẸ TAY: 5e-6 hoặc thấp hơn, kèm --epochs 1.",
    )
    parser.add_argument("--hard-negatives", action="store_true", help="Bật hard negative mining (TripletLoss thay vì MultipleNegativesRankingLoss)")
    parser.add_argument("--hard-negative-top-k", type=int, default=5)
    parser.add_argument(
        "--loss", choices=["triplet", "mnrl"], default="triplet",
        help="Hàm mất mát khi bật --hard-negatives (thêm 2026-08-23). 'mnrl' = "
             "MultipleNegativesRankingLoss — CHUẨN cho fine-tune retriever: softmax trên cosine, "
             "dùng hard negative tường minh CỘNG in-batch negatives, KHÔNG có margin nên không thể "
             "thoái hoá. 'triplet' = mặc định cũ, GIỮ để tái lập lần chạy trước — nhưng đã gây SỤP "
             "ĐỔ BIỂU DIỄN thật ngày 2026-08-22 (Recall 0.92 -> 0.0046). KHUYẾN NGHỊ: mnrl.",
    )
    parser.add_argument(
        "--triplet-margin", type=float, default=5.0,
        help="Margin cho TripletLoss (mặc định 5.0 = mặc định của sentence-transformers). ⚠️ Với "
             "model CÓ chuẩn hoá L2 thì khoảng cách Euclid tối đa chỉ = 2, nên margin > 2 KHÔNG BAO "
             "GIỜ thoả mãn được và gây sụp đổ biểu diễn — script sẽ CHẶN cấu hình đó.",
    )
    parser.add_argument(
        "--hard-negative-ranks", default=None,
        help="File .npz thứ hạng (sinh bởi `src/evaluation/ensemble_fusion.py rank`) trên ĐÚNG tập câu "
             "hỏi đang train (thêm 2026-08-22). Nếu truyền, hard negative lấy từ chính top-N của "
             "RETRIEVER THẬT thay vì lấy mẫu ngẫu nhiên 2000/8532 văn bản như bản cũ — negative KHÓ "
             "hơn hẳn, sát lỗi thật của pipeline, và KHÔNG tốn GPU (tái dùng file đã có). "
             "Vd: outputs/cache/ranks_hold_v1.npz (top-200 cho 5602 câu train_split_80).",
    )
    parser.add_argument(
        "--hard-negative-chunks", default="data/processed/chunks_corpus.jsonl",
        help="File chunk mà --hard-negative-ranks tham chiếu tới (chỉ số dòng = chỉ số chunk).",
    )
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
        if args.hard_negative_ranks:
            # Nguồn negative TỐT HƠN: lấy từ thứ hạng của retriever THẬT (xem docstring hàm).
            gold_by_qid = {
                str(qid): set(str(d) for d in item["answer"])
                for qid, item in load_labeled_questions(args.train_json).items()
            }
            triplets = mine_hard_negatives_from_ranks(
                pairs, gold_by_qid, args.hard_negative_ranks, args.hard_negative_chunks,
                top_k=args.hard_negative_top_k, max_chars=args.max_chars,
                passage_prefix=args.passage_prefix,
            )
        else:
            corpus = CorpusReader(args.corpus)
            triplets = mine_hard_negatives(
                model, pairs, corpus, top_k=args.hard_negative_top_k, max_chars=args.max_chars,
                passage_prefix=args.passage_prefix,
            )
        if not triplets:
            logger.error("Không tạo được triplet nào — dừng lại (kiểm tra --hard-negative-ranks).")
            sys.exit(1)
        train_examples = [InputExample(texts=[q, pos, neg]) for q, pos, neg in triplets]
        if args.loss == "mnrl":
            # MultipleNegativesRankingLoss với 3 cột: cột 3 = hard negative TƯỜNG MINH, CỘNG THÊM
            # in-batch negatives (mọi positive của câu hỏi khác trong batch). Đây là loss CHUẨN cho
            # fine-tune retriever: dùng softmax trên cosine similarity, KHÔNG có margin nên KHÔNG
            # thể thoái hoá kiểu TripletLoss bên dưới.
            loss = losses.MultipleNegativesRankingLoss(model=model)
            logger.info(
                "Dùng MultipleNegativesRankingLoss với %d bộ (anchor, positive, hard_negative) "
                "+ in-batch negatives. Batch càng lớn càng nhiều in-batch negative -> tín hiệu càng tốt.",
                len(train_examples),
            )
        else:
            loss = losses.TripletLoss(model=model, triplet_margin=args.triplet_margin)
            logger.info("Dùng TripletLoss (margin=%g) với %d triplet", args.triplet_margin, len(train_examples))
            # ⚠️ CẢNH BÁO SINH RA TỪ THẤT BẠI THẬT 2026-08-22 (9 giờ GPU đổ đi):
            # TripletLoss mặc định dùng khoảng cách EUCLIDEAN + margin=5. Model AITeamVN có module
            # Normalize -> mọi embedding nằm trên mặt cầu đơn vị -> khoảng cách Euclid TỐI ĐA = 2 < 5.
            # Margin KHÔNG BAO GIỜ thoả mãn -> gradient không bao giờ tắt -> model bị đẩy vô hạn về
            # lời giải THOÁI HOÁ: đẩy mọi negative tới ĐỐI CỰC của query, nén cả không gian 1024
            # chiều xuống 1 TRỤC (mọi vector chỉ còn +v hoặc -v). Loss vẫn giảm đều trông rất đẹp,
            # nhưng Recall sụp từ 0.92 xuống 0.0046 (≈ đoán ngẫu nhiên).
            # Triệu chứng nhận biết: cos(a,b) ra ĐÚNG ±1.0000, cos trung bình với văn bản ngẫu nhiên ÂM MẠNH.
            if _model_is_normalized(model) and args.triplet_margin > 2.0:
                logger.error(
                    "⚠️⚠️ DỪNG LẠI: model CÓ chuẩn hoá L2 (khoảng cách Euclid tối đa = 2) nhưng "
                    "triplet_margin=%g > 2 -> margin KHÔNG BAO GIỜ thoả mãn -> gần như CHẮC CHẮN "
                    "sụp đổ biểu diễn (đã xảy ra thật 2026-08-22, mất 9 giờ GPU). "
                    "Dùng --loss mnrl (khuyến nghị) hoặc --triplet-margin 0.5.",
                    args.triplet_margin,
                )
                sys.exit(1)
    else:
        train_examples = [InputExample(texts=[q, pos]) for q, pos, _, _ in pairs]
        loss = losses.MultipleNegativesRankingLoss(model=model)
        logger.info("Dùng MultipleNegativesRankingLoss với %d cặp (in-batch negatives)", len(train_examples))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)
    # `--fp16` TỪNG LÀ CỜ CHẾT (2026-08-22): nó được khai báo + ghi tài liệu là "bật use_amp trong
    # model.fit()" nhưng KHÔNG dòng code nào đọc `args.fp16` - fit() chỉ đọc `args.use_amp`. Nghĩa là
    # lần chạy dòng #12 ghi trong log là "có --fp16" THỰC TẾ đã chạy fp32. Nay nối 2 cờ lại để cả hai
    # đều thật sự bật AMP.
    use_amp = bool(args.use_amp or getattr(args, "fp16", False))
    logger.info(
        "Bắt đầu fine-tune: %d epoch, batch_size=%d, warmup_steps=%d, learning_rate=%g, use_amp=%s",
        args.epochs, args.batch_size, warmup_steps, args.learning_rate, use_amp,
    )
    if args.learning_rate >= 2e-5 and args.epochs >= 2:
        logger.warning(
            "⚠️ learning_rate=%g VỚI %d epoch - đây ĐÚNG cấu hình đã gây catastrophic forgetting ở "
            "dòng #12 (public 0.8922 < pretrained 0.9058). Cân nhắc --learning-rate 5e-6 --epochs 1.",
            args.learning_rate, args.epochs,
        )

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=args.output_dir,
        use_amp=use_amp,
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
