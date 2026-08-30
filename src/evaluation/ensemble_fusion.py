"""
ENSEMBLE nhiều model embedding + BM25 bằng RRF nhiều nhánh (2026-08-21).

ĐỘNG LỰC (đo THẬT, không suy đoán — xem submission-log-task1.md mục "ensemble"):
`AITeamVN/Vietnamese_Embedding` (v1, model chính thức) đang MISS 138 tài liệu GT trên val_split_20.
Phân tích complementarity từ các file prediction CÓ SẴN cho thấy các model YẾU HƠN khi đứng riêng lại
cứu được rất nhiều trong số đó:

    vietlegal-harrier  (Recall 0.9040)  cứu 47/138 = 34.1%
    bge-m3 gốc         (Recall 0.8847)  cứu 37/138 = 26.8%
    gte-Qwen2          (Recall 0.8864)  cứu 35/138 = 25.4%
    AITeamVN v2        (Recall 0.9131)  cứu 19/138 = 13.8%
    UNION (trần)                        cứu 67/138 = 48.6%

Ngưỡng do chính `scripts/check_complementarity.py` của dự án đặt ra từ trước: ">30-40% => có tín hiệu
thật để thử fusion". harrier một mình đã vượt ngưỡng.

CƠ CHẾ (2 tác dụng TÁCH BIỆT, đừng lẫn lộn):
  1. MỞ RỘNG POOL: hiện chỉ lấy dense-top-n=100 chunk từ MỘT model. Tài liệu v1 xếp hạng #5000 thì
     KHÔNG BAO GIỜ vào pool -> mất vĩnh viễn. Ensemble lấy HỢP top-N của nhiều model nên tài liệu đó
     vẫn vào pool nếu model khác xếp nó cao. Đây là phần chắc chắn có lợi.
  2. XẾP LẠI HẠNG bằng RRF nhiều nhánh — mở rộng ĐÚNG công thức 2 nhánh đang dùng:
         score(d) = 1.0/(k+rank_v1) + w_h/(k+rank_harrier) + ... + w_bm25/(k+rank_bm25)
     RRF cộng theo THỨ HẠNG nên KHÔNG cần chuẩn hoá điểm giữa các model khác thang đo (đúng lý do RRF
     được chọn từ đầu cho dense+BM25). Phần này CÓ RỦI RO: w quá nhỏ -> không đủ kéo tài liệu được cứu
     lên top-5 (ensemble = baseline); w quá lớn -> rác của model phụ đá văng tài liệu đúng của v1.
     Chỉ sweep trên val mới biết -> đó là toàn bộ mục đích của script này.

⚠️ PHÁT HIỆN QUAN TRỌNG khi unit-test cơ chế fusion (2026-08-21) — CỬA SỔ TRỌNG SỐ RẤT HẸP:
Với `rrf_k=60`, đường cong RRF cực PHẲNG (hạng 1 = 1/61 = 0,01639; hạng 100 = 1/160 = 0,00625, chỉ
chênh 2,6 lần). Mô phỏng tài liệu mà nguồn phụ xếp HẠNG 1 còn v1 KHÔNG xếp trong top-100:

    w_aux = 0.3  -> top-5 KHÔNG đổi (0,00492 < 0,01538 của tài liệu hạng 5 của v1) - vô tác dụng
    w_aux = 0.8  -> top-5 KHÔNG đổi (0,01311 < 0,01538)                             - vô tác dụng
    w_aux = 1.0  -> cứu được, chen vào hạng 2                                        - VÙNG HỮU ÍCH
    w_aux = 1.5  -> TOÀN BỘ top-5 bị nguồn phụ chiếm, đá văng hết tài liệu của v1    - THẢM HOẠ

=> KHÁC HẲN `bm25-weight=0.1` đang dùng! BM25 trùng lặp nhiều với dense nên chỉ cần trọng số nhỏ để
XẾP LẠI HẠNG. Ở đây ta cần MỞ RỘNG POOL (kéo tài liệu v1 hoàn toàn bỏ sót vào), việc đó đòi hỏi trọng
số gần 1.0. Vì vậy lưới sweep phải quét quanh 0.6-1.3, KHÔNG phải 0.1-0.5.

=> 2 CÁCH LÀM CỬA SỔ RỘNG/AN TOÀN HƠN, cả 2 đều đã tham số hoá sẵn, NÊN sweep cùng lúc:
   (a) `--rrf-k` NHỎ hơn (30, 10): đường cong DỐC hơn -> chỉ ứng viên nguồn phụ xếp RẤT CAO mới đủ
       điểm chen vào, phần đuôi rác bị dìm -> chọn lọc hơn hẳn. rrf-k giờ NẰM TRONG lưới sweep.
   (b) `--use-top-n` nhỏ cho nguồn phụ (vd 20-50 thay vì 100): chặn thẳng phần đuôi của nguồn phụ,
       không cho vào pool. An toàn nhất khi muốn tăng trọng số mà sợ rác.

THIẾT KẾ 2 PHA (để sweep cho RẺ và KHÔNG nổ RAM):
  Pha 1 `rank` / `rank-bm25` (chạy 1 LẦN cho mỗi nguồn, tuần tự, KHÔNG chồng RAM):
      load cache embedding của DUY NHẤT model đó -> encode 1398 câu hỏi -> lấy top-N chunk/câu ->
      ghi ra file .npz NHỎ (1398 x 200 int32, ~1MB) -> giải phóng RAM -> sang model kế tiếp.
      KHÔNG encode lại corpus (dùng cache có sẵn) — bắt buộc phải có cache, xem --allow-encode.
  Pha 2 `fuse` (chạy vô số lần, tức thì, KHÔNG đụng GPU):
      chỉ đọc mấy file thứ hạng nhỏ đó rồi fuse + sweep trọng số. Mỗi tổ hợp vài giây.

BẢO HIỂM QUAN TRỌNG: đặt weight của MỌI nguồn phụ = 0 sẽ cho ra ĐÚNG cấu hình chính thức hiện tại
(v1 weight 1.0 + bm25 weight 0.1, rrf-k=60, top-n=100, gộp max) = Recall 0.9295. Baseline LUÔN nằm
trong lưới sweep -> nếu mọi tổ hợp đều thua baseline thì kết luận "đóng hướng" là chắc chắn, không mơ hồ.

Cách dùng (chạy từ thư mục GỐC repo):

  # Pha 1 - mỗi lệnh chạy riêng, tuần tự (KHÔNG chạy song song, sẽ nổ RAM)
  python -m src.evaluation.ensemble_fusion rank --name v1 \
      --embedding-model AITeamVN/Vietnamese_Embedding \
      --embedding-cache outputs/cache/embcompare_aiteamvn.npz \
      --out outputs/cache/ranks_v1.npz

  python -m src.evaluation.ensemble_fusion rank --name harrier \
      --embedding-model mainguyen9/vietlegal-harrier-0.6b --trust-remote-code \
      --embedding-cache outputs/cache/embcompare_vietlegal_harrier.npz \
      --query-instruct "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question" \
      --max-seq-length 1024 --out outputs/cache/ranks_harrier.npz

  python -m src.evaluation.ensemble_fusion rank-bm25 --name bm25 --out outputs/cache/ranks_bm25.npz

  # Pha 2 - sweep, chạy lại bao nhiêu lần cũng được, KHÔNG cần GPU
  # (lưới quanh 0.6-1.3 theo phát hiện "cửa sổ trọng số hẹp" ở trên, + sweep luôn rrf-k)
  python -m src.evaluation.ensemble_fusion fuse \
      --rank v1=outputs/cache/ranks_v1.npz \
      --rank harrier=outputs/cache/ranks_harrier.npz \
      --rank bm25=outputs/cache/ranks_bm25.npz \
      --weights harrier=0,0.6,0.8,1.0,1.2 \
      --weights bm25=0.1 \
      --rrf-k 60,30,10 \
      --report outputs/predictions/ensemble_sweep_report.json

⚠️ TUÂN THỦ: mỗi model thêm vào pipeline đều tính vào hạn mức 4 TỶ THAM SỐ (quy định mục 5) và phải
có trong `scripts/allowed-models.md` — model MỚI ngoài danh sách phải gửi đề xuất cho BTC TRƯỚC 10
NGÀY so với hạn chót private test. v1 (568M) + harrier (596M) + bge-m3 (568M) = ~1,73 tỷ, vẫn dưới
hạn mức, nhưng PHẢI cập nhật allowed-models.md + gửi đề xuất nếu chốt dùng thật.

===============================================================================
THÊM 2026-08-28: FUSE THEO ĐIỂM (CombSUM), KHÔNG CHỈ THEO HẠNG (RRF) — `fuse --mode score`
===============================================================================
ĐỘNG LỰC (phát hiện từ audit code, không phải suy đoán — xem submission-log-task1.md mục "fusion"):
RRF ở trên CHỈ dùng `rank` (vị trí 1, 2, 3, ...), KHÔNG BAO GIỜ dùng điểm similarity/BM25 thật. Hai
tài liệu hạng 1 với similarity 0.95 và 0.40 đóng góp Y HỆT NHAU vào RRF — "hạng 1 áp đảo mọi nguồn
khác nhưng chỉ hơn hạng 2 một chút" và "hạng 1 vượt trội tuyệt đối" bị coi như nhau. Đây là thông tin
BỊ VỨT BỎ hoàn toàn trong toàn bộ pipeline từ trước tới nay (`save_ranks` cũ chỉ lưu `ranks`, không
lưu điểm) — không phải một biến thể của hướng đã thử, mà là CÁCH GỘP MỚI THẬT SỰ, đúng 1 trong 2
phạm trù duy nhất từng đem lại cải thiện thật (đổi TÍN HIỆU hoặc đổi CÁCH GỘP).

CƠ CHẾ: mỗi nguồn (v1, harrier, bm25, m3sparse, reranker...) giờ LƯU THÊM điểm thô của đúng top-N ứng
viên đã lưu hạng (không lưu điểm cho TOÀN BỘ corpus — quá tốn, 1398 câu x 351394 chunk x 4 byte =
~2GB/nguồn). Khi fuse theo điểm: chuẩn hoá per-query (min-max hoặc z-score, MỖI CÂU HỎI RIÊNG — bắt
buộc vì thang điểm khác hẳn nhau giữa cosine similarity [-1,1], BM25 [0,+∞) không giới hạn, và logit
reranker) rồi CỘNG CÓ TRỌNG SỐ (CombSUM): score(d) = Σ w_i * norm_i(score_i(d)), nguồn nào không xếp
hạng tài liệu d thì KHÔNG đóng góp (giống hệt quy ước "weight=0 thì bỏ qua" của RRF).

TƯƠNG THÍCH NGƯỢC: `save_ranks()` thêm tham số `scores` TUỲ CHỌN — mọi lệnh `rank`/`rank-bm25` cũ vẫn
chạy được, chỉ là giờ LUÔN lưu thêm điểm (rẻ, cùng shape với ranks, ~1MB/nguồn). File .npz CŨ (sinh
trước 2026-08-28, không có key "scores") vẫn đọc được bằng `load_ranks()` như trước — `fuse --mode
rrf` (mặc định, KHÔNG đổi) hoạt động y hệt cũ trên cả file cũ lẫn mới. Chỉ `fuse --mode score` mới
cần "scores" trong file — nếu thiếu sẽ báo lỗi rõ ràng, không phải KeyError khó hiểu.

BẢO HIỂM: giống RRF, dùng `--mode rrf` (mặc định) tái lập ĐÚNG baseline 0.9319 vì code nhánh RRF
KHÔNG ĐỔI MỘT DÒNG nào so với trước 2026-08-28.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .evaluate import load_ground_truth
    from .metrics import evaluate_predictions
    from .full_pipeline_eval import (
        load_chunks, load_questions, resolve_device, encode_chunks, build_bm25_index,
        _naive_tokenize, build_bm25_fast_matrix, bm25_scores_batch_fast,
    )
except ImportError:  # cho phép chạy trực tiếp
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.metrics import evaluate_predictions
    from src.evaluation.full_pipeline_eval import (
        load_chunks, load_questions, resolve_device, encode_chunks, build_bm25_index,
        _naive_tokenize, build_bm25_fast_matrix, bm25_scores_batch_fast,
    )


# ---------------------------------------------------------------------------
# tiện ích chung
# ---------------------------------------------------------------------------

def top_n_indices(scores, top_n: int):
    """Lấy chỉ số top-N theo điểm giảm dần — COPY ĐÚNG hành vi của rrf_fuse() trong
    full_pipeline_eval.py (argpartition rồi sort lại phần đã cắt) để tái lập baseline chính xác."""
    import numpy as np

    n = len(scores)
    top_n = min(top_n, n)
    idx = np.argpartition(-scores, top_n - 1)[:top_n]
    return idx[np.argsort(-scores[idx])]


def save_ranks(path: str, question_ids: Sequence[str], ranks, scores=None) -> None:
    """`scores` (thêm 2026-08-28, TUỲ CHỌN): điểm thô (similarity/BM25/reranker logit), CÙNG SHAPE và
    CÙNG THỨ TỰ với `ranks` — scores[q, j] là điểm của tài liệu ranks[q, j]. Bỏ trống (None) vẫn ra
    file y hệt hành vi cũ (tương thích ngược 100% với mọi consumer đang gọi load_ranks())."""
    import numpy as np

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        question_ids=np.array([str(q) for q in question_ids]),
        ranks=ranks.astype(np.int32),
    )
    extra = " (KHÔNG có scores — 'fuse --mode score' sẽ báo lỗi nếu dùng file này)"
    if scores is not None:
        kwargs["scores"] = scores.astype(np.float32)
        extra = f", scores shape={scores.shape}"
    np.savez_compressed(path, **kwargs)
    print(f"Đã lưu thứ hạng: {path}  (shape={ranks.shape}{extra})")


def load_ranks(path: str):
    import numpy as np

    d = np.load(path, allow_pickle=False)
    return [str(q) for q in d["question_ids"]], d["ranks"]


def load_ranks_scores(path: str):
    """Giống load_ranks() nhưng LOAD LUÔN scores — dùng riêng cho `fuse --mode score` (thêm 2026-08-28).
    Trả về (question_ids, ranks, scores) với scores=None nếu file CŨ không có key này, để nơi gọi tự
    báo lỗi rõ ràng thay vì KeyError mơ hồ ở giữa vòng lặp."""
    import numpy as np

    d = np.load(path, allow_pickle=False)
    qids = [str(q) for q in d["question_ids"]]
    scores = d["scores"] if "scores" in d.files else None
    return qids, d["ranks"], scores


# ---------------------------------------------------------------------------
# Pha 1: sinh thứ hạng
# ---------------------------------------------------------------------------

def cmd_rank(args) -> None:
    import numpy as np

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[lỗi] Thiếu sentence-transformers.", file=sys.stderr)
        sys.exit(1)

    cache = Path(args.embedding_cache)
    if not cache.exists() and not args.allow_encode:
        print(
            f"[lỗi] KHÔNG tìm thấy cache embedding: {cache}\n"
            f"      Script này CỐ Ý từ chối encode lại toàn bộ corpus (mất nhiều GIỜ GPU).\n"
            f"      Kiểm tra lại đường dẫn, hoặc thêm --allow-encode nếu THỰC SỰ muốn encode lại.",
            file=sys.stderr,
        )
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

    if args.use_segmented_text:
        # Thêm 2026-08-28 cho nhánh DEk21 (PhoBERT/RoBERTa family) — các model này được train trên
        # tiếng Việt ĐÃ TÁCH TỪ (word segmentation, nối bằng dấu "_"), KHÁC v1/harrier vốn dùng thẳng
        # câu chữ thô. Data đã có sẵn 2 field song song (passage_segmented/question_segmented) từ
        # trước — KHÔNG cần tách từ lại. Cách làm: gán đè "passage"/"question" bằng bản đã tách từ rồi
        # để encode_chunks()/luồng encode câu hỏi bên dưới chạy Y NGUYÊN như cũ, không phải sửa 2 nơi.
        n_miss_p = sum(1 for c in chunks if not c.get("passage_segmented"))
        if n_miss_p:
            print(f"[cảnh báo] {n_miss_p}/{len(chunks)} chunk THIẾU passage_segmented -> dùng lại "
                  f"passage thô cho các chunk đó (có thể làm model PhoBERT-family xếp hạng kém hơn).",
                  file=sys.stderr)
        chunks = [dict(c, passage=(c.get("passage_segmented") or c["passage"])) for c in chunks]
        n_miss_q = sum(1 for q in questions if not q.get("question_segmented"))
        if n_miss_q:
            print(f"[cảnh báo] {n_miss_q}/{len(questions)} câu hỏi THIẾU question_segmented -> dùng "
                  f"lại question thô cho các câu đó.", file=sys.stderr)
        print("--use-segmented-text: đang dùng passage_segmented/question_segmented (không phải văn "
              "bản thô) cho toàn bộ encode.")

    print(f"Load embedding model: {args.embedding_model} (trust_remote_code={args.trust_remote_code})")
    model = SentenceTransformer(args.embedding_model, device=device, trust_remote_code=args.trust_remote_code)
    if args.max_seq_length is not None:
        model.max_seq_length = args.max_seq_length
        print(f"Đã ghi đè max_seq_length = {args.max_seq_length}")
    if device == "cuda" and args.fp16_inference:
        try:
            model = model.half()
        except Exception as e:  # noqa: BLE001
            print(f"[cảnh báo] Không cast fp16 ({e}).", file=sys.stderr)

    query_prefix = args.query_prefix
    if args.query_instruct:
        # GIỐNG HỆT full_pipeline_eval.py dòng ~972 - giữ nguyên quy ước để kết quả so sánh được.
        query_prefix = f"Instruct: {args.query_instruct}\nQuery: "
        print(f"query_prefix (từ --query-instruct) = {query_prefix!r}")

    chunk_embeddings = encode_chunks(
        model, chunks, args.batch_size, str(cache),
        passage_prefix=args.passage_prefix, passage_prompt_name=args.passage_prompt_name,
    )
    if chunk_embeddings.shape[0] != len(chunks):
        print(
            f"[lỗi] Cache có {chunk_embeddings.shape[0]} dòng nhưng chunks có {len(chunks)} dòng — "
            f"KHÔNG khớp, dừng lại (dùng nhầm cache sẽ cho điểm SAI hoàn toàn).",
            file=sys.stderr,
        )
        sys.exit(1)

    all_ranks = np.zeros((len(questions), args.top_n), dtype=np.int32)
    all_scores = np.zeros((len(questions), args.top_n), dtype=np.float32)
    qids: List[str] = []
    for start in range(0, len(questions), args.question_batch_size):
        batch = questions[start : start + args.question_batch_size]
        encode_kwargs = dict(
            batch_size=args.question_batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        q_field = "question_segmented" if args.use_segmented_text else "question"
        if args.query_prompt_name:
            batch_texts = [(q.get(q_field) or q["question"]) for q in batch]
            encode_kwargs["prompt_name"] = args.query_prompt_name
        else:
            batch_texts = [query_prefix + (q.get(q_field) or q["question"]) for q in batch]
        emb = model.encode(batch_texts, **encode_kwargs).astype(np.float32)
        sims = emb @ chunk_embeddings.T
        for i, q in enumerate(batch):
            idx = top_n_indices(sims[i], args.top_n)
            all_ranks[start + i] = idx
            all_scores[start + i] = sims[i][idx]  # cosine similarity thô (embedding đã normalize)
            qids.append(str(q["question_id"]))
        print(f"  đã xử lý {min(start + args.question_batch_size, len(questions))}/{len(questions)} câu hỏi")

    save_ranks(args.out, qids, all_ranks, scores=all_scores)


def cmd_rank_reranker(args) -> None:
    """Sinh thứ hạng từ CROSS-ENCODER RERANKER, coi nó như MỘT NGUỒN NỮA trong RRF (2026-08-22).

    ĐỘNG LỰC — lý do reranker từng thất bại có thể KHÔNG phải vì reranker kém:
    `rerank_candidate_details()` trong full_pipeline_eval.py xếp lại candidate THUẦN theo điểm
    reranker (`sorted(zip(candidates, scores), key=-score)`), VỨT BỎ HOÀN TOÀN điểm RRF. Tức là
    reranker đang chạy ở "TRỌNG SỐ VÔ HẠN" - đúng kiểu thất bại đã chẩn đoán được ở harrier
    (unit-test cho thấy w=1.5 là nguồn phụ chiếm sạch top-5, đá văng hết tài liệu đúng của v1).
    Cách làm ở đây: reranker chỉ ĐÓNG GÓP thứ hạng vào RRF với 1 trọng số dò được, KHÔNG thay thế.
    Đây đúng là biến thể mà log dự án đã tự ghi là ngoại lệ duy nhất đáng thử lại ("kết hợp điểm
    thay vì thay thế"), và tới nay CHƯA ai làm.

    Reranker chỉ chấm top-N ứng viên của 1 nguồn có sẵn (thường là v1) - chunk ngoài top-N đó không
    nhận đóng góp nào từ reranker, điều này ĐÚNG với tinh thần RRF (nguồn nào không xếp thì không cộng).
    """
    import numpy as np

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("[lỗi] Thiếu sentence-transformers.", file=sys.stderr)
        sys.exit(1)

    device = resolve_device(args.device)
    chunks = load_chunks(args.chunks)
    questions = load_questions(args.questions)
    if args.limit_questions:
        questions = questions[: args.limit_questions]

    cand_qids, cand_ranks = load_ranks(args.candidates)
    if [str(q["question_id"]) for q in questions] != cand_qids:
        print(
            f"[lỗi] Thứ tự question_id trong {args.candidates!r} KHÁC {args.questions!r} — "
            f"sẽ chấm nhầm câu hỏi. Sinh lại Pha 1 với cùng --questions.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_cand = min(args.top_n, cand_ranks.shape[1])
    total_pairs = len(questions) * n_cand
    print(f"Load reranker: {args.reranker_model}")
    print(f"  {len(questions)} câu hỏi x {n_cand} ứng viên = {total_pairs:,} cặp qua cross-encoder.")
    print(f"  LƯU Ý: cross-encoder KHÔNG cache được (điểm của đoạn văn phụ thuộc câu hỏi, phải chạy")
    print(f"  cả cặp qua model cùng lúc) - đây là bước ĐẮT NHẤT pipeline, khác hẳn dense/BM25.")
    model = CrossEncoder(args.reranker_model, num_labels=1, device=device)

    if args.max_length:
        model.max_length = args.max_length
        print(f"  Đã đặt max_length = {args.max_length} (chi phí attention tăng theo BÌNH PHƯƠNG độ dài)")

    # Cặp thử để smoke-test ngay sau khi cast fp16 (xem bên dưới vì sao PHẢI test).
    _probe = [
        (questions[0]["question"], chunks[int(cand_ranks[0][0])]["passage"]),
        (questions[0]["question"], chunks[int(cand_ranks[0][1])]["passage"]),
    ]

    if device == "cuda" and args.fp16_inference:
        try:
            # ⚠️ PHẢI gọi .half() TẠI CHỖ, TUYỆT ĐỐI KHÔNG gán `model.model = model.model.half()`.
            # sentence-transformers v5 dựng CrossEncoder theo kiểu modular (base/model.py gọi
            # `module(input, **module_kwargs)`): `model.model` là model HF NẰM TRONG một module
            # wrapper biết cách giải nén dict tokenizer thành input_ids + attention_mask. Gán lại
            # sẽ THAY module wrapper bằng model HF thô -> predict() truyền nguyên BatchEncoding vào
            # vị trí input_ids và thiếu attention_mask -> TypeError: list indices must be integers
            # or slices, not tuple (đã gặp thật 2026-08-22). .half() vốn sửa trọng số tại chỗ và
            # trả về chính nó nên phép gán là thừa.
            model.model.half()
            model.predict(_probe, batch_size=2, show_progress_bar=False)  # smoke-test
            print("  Đã cast reranker sang fp16 (nhanh ~2x trên RTX 3060; tắt bằng --no-fp16-inference)")
        except Exception as e:  # noqa: BLE001
            print(f"  [cảnh báo] fp16 KHÔNG dùng được ({type(e).__name__}: {e})", file=sys.stderr)
            print("  -> tải lại model ở fp32 và chạy tiếp (chậm hơn ~2x nhưng vẫn đúng).", file=sys.stderr)
            model = CrossEncoder(args.reranker_model, num_labels=1, device=device)
            if args.max_length:
                model.max_length = args.max_length

    all_ranks = np.zeros((len(questions), n_cand), dtype=np.int32)
    all_scores = np.zeros((len(questions), n_cand), dtype=np.float32)
    qb = max(1, args.question_batch_size)
    done = 0
    for start in range(0, len(questions), qb):
        batch = questions[start : start + qb]
        # Gộp cặp của NHIỀU câu hỏi vào 1 lần predict() - giảm overhead gọi hàm/tokenizer lặp lại
        # (bản đầu gọi predict() riêng cho TỪNG câu hỏi, tốn overhead không cần thiết).
        pairs, cands = [], []
        for i, q in enumerate(batch):
            cand = cand_ranks[start + i][:n_cand]
            cands.append(cand)
            pairs.extend((q["question"], chunks[int(idx)]["passage"]) for idx in cand)
        scores = np.asarray(
            model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False), dtype=np.float32
        ).reshape(len(batch), n_cand)
        for i, cand in enumerate(cands):
            order = np.argsort(-scores[i])
            all_ranks[start + i] = cand[order]
            all_scores[start + i] = scores[i][order]  # logit thô của cross-encoder (chưa qua sigmoid)
        done += len(batch)
        print(f"  đã chấm {done}/{len(questions)} câu hỏi")

    save_ranks(args.out, [str(q["question_id"]) for q in questions], all_ranks, scores=all_scores)


def cmd_rank_bm25(args) -> None:
    import numpy as np

    chunks = load_chunks(args.chunks)
    questions = load_questions(args.questions)
    if args.limit_questions:
        questions = questions[: args.limit_questions]

    bm25_index = build_bm25_index(chunks)
    print("Xây ma trận BM25 vector hoá (giống --bm25-fast)...")
    bm25_fast = build_bm25_fast_matrix(bm25_index)

    all_ranks = np.zeros((len(questions), args.top_n), dtype=np.int32)
    all_scores = np.zeros((len(questions), args.top_n), dtype=np.float32)
    qids: List[str] = []
    for start in range(0, len(questions), args.question_batch_size):
        batch = questions[start : start + args.question_batch_size]
        q_tokens_batch = [
            q["question_segmented"].split() if q.get("question_segmented") else _naive_tokenize(q["question"])
            for q in batch
        ]
        scores_batch = bm25_scores_batch_fast(bm25_fast, q_tokens_batch)
        for i, q in enumerate(batch):
            idx = top_n_indices(scores_batch[i], args.top_n)
            all_ranks[start + i] = idx
            all_scores[start + i] = scores_batch[i][idx]  # điểm BM25 thô, KHÔNG giới hạn trên
            qids.append(str(q["question_id"]))
        print(f"  đã xử lý {min(start + args.question_batch_size, len(questions))}/{len(questions)} câu hỏi")

    save_ranks(args.out, qids, all_ranks, scores=all_scores)


# ---------------------------------------------------------------------------
# Pha 2: fuse + sweep
# ---------------------------------------------------------------------------

def parse_named(items: Sequence[str], what: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            print(f"[lỗi] --{what} phải có dạng ten=giatri, nhận được: {item!r}", file=sys.stderr)
            sys.exit(1)
        name, value = item.split("=", 1)
        out[name.strip()] = value.strip()
    return out


def fuse_one_question(rank_rows: Dict[str, "object"], weights: Dict[str, float],
                      use_top_n: Dict[str, int], rrf_k: int) -> Dict[int, float]:
    """RRF nhiều nhánh cho MỘT câu hỏi — mở rộng trực tiếp rrf_fuse() 2 nhánh của full_pipeline_eval.py.
    Nguồn có weight = 0 bị BỎ QUA HOÀN TOÀN (không đóng góp, cũng không mở rộng pool) — nhờ vậy đặt
    tất cả weight phụ = 0 sẽ tái lập ĐÚNG baseline hiện tại."""
    rrf: Dict[int, float] = defaultdict(float)
    for name, row in rank_rows.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        n = use_top_n.get(name, len(row))
        for rank, idx in enumerate(row[:n], start=1):
            rrf[int(idx)] += w / (rrf_k + rank)
    return rrf


def normalize_scores(scores, method: str):
    """Chuẩn hoá điểm của MỘT câu hỏi, MỘT nguồn (thêm 2026-08-28). Bắt buộc làm per-query, per-source
    vì thang điểm khác hẳn nhau: cosine similarity [-1,1] (thực tế embedding đã chuẩn hoá nên gần
    [0,1]), BM25 [0,+∞) không có trần, logit reranker chưa qua sigmoid có thể âm/dương tuỳ ý.

    'minmax' (mặc định): co giãn về [0,1] theo min/max của ĐÚNG top-N ứng viên đã lưu cho câu hỏi đó
    (không phải toàn corpus — không có điểm của phần còn lại). Đây là XẤP XỈ: min/max thật của toàn
    corpus có thể rộng hơn, nhưng vì CombSUM chỉ cần so sánh TƯƠNG ĐỐI giữa các ứng viên đã vào pool,
    xấp xỉ này chấp nhận được (giống cách RRF cũng chỉ nhìn thấy rank trong top-N, không phải toàn bộ).
    'zscore': (x - trung bình) / độ lệch chuẩn — ít nhạy với 1-2 điểm ngoại lai hơn minmax.

    Nếu mọi điểm bằng nhau (hi-lo hoặc std ≈ 0, ví dụ top-N toàn 0 do BM25 không khớp từ nào) trả về
    toàn 0 — tránh chia 0, và hợp lý vì khi đó nguồn này không cung cấp thông tin phân biệt gì."""
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    if method == "minmax":
        lo, hi = float(scores.min()), float(scores.max())
        if hi - lo < 1e-12:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)
    elif method == "zscore":
        mu, sd = float(scores.mean()), float(scores.std())
        if sd < 1e-12:
            return np.zeros_like(scores)
        return (scores - mu) / sd
    else:
        raise ValueError(f"Không nhận method={method!r}, chỉ chấp nhận 'minmax' hoặc 'zscore'")


def fuse_one_question_by_score(rank_rows: Dict[str, "object"], score_rows: Dict[str, "object"],
                               weights: Dict[str, float], use_top_n: Dict[str, int],
                               norm_method: str) -> Dict[int, float]:
    """CombSUM cho MỘT câu hỏi — thay vì cộng theo 1/(k+rank) như RRF, cộng theo điểm ĐÃ CHUẨN HOÁ
    per-query per-source: score(d) = Σ w_i * norm_i(score_i(d)). Nguồn có weight=0 hoặc không xếp
    hạng d thì KHÔNG đóng góp (giống hệt quy ước của fuse_one_question ở trên — weight phụ toàn 0 vẫn
    tái lập được cấu hình chỉ-1-nguồn)."""
    comb: Dict[int, float] = defaultdict(float)
    for name, row in rank_rows.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        n = use_top_n.get(name, len(row))
        raw = score_rows[name][:n]
        normed = normalize_scores(raw, norm_method)
        for idx, s in zip(row[:n], normed):
            comb[int(idx)] += w * float(s)
    return comb


def aggregate_max(rrf_score: Dict[int, float], doc_ids: Sequence[str], top_k: int,
                  group_of: Optional[Dict[str, int]] = None) -> List[str]:
    """Gộp chunk -> document lấy chunk điểm CAO NHẤT (giống aggregate_to_documents()).
    Đã xác nhận 2026-08-21 qua sweep_aggregation: 'max' TỐT NHẤT (sum_top2 -0.0069, sum_top3 -0.0107,
    mean_top2 -0.0209).

    `group_of` (thêm 2026-08-23): map doc_id -> id nhóm TRÙNG LẶP NỘI DUNG (27 nhóm/54 doc_id, xác minh
    bằng difflib độ giống 98.78-99.87%, xem data/processed/duplicate_doc_groups.json). Khi truyền vào,
    nếu 2 doc_id trong danh sách xếp hạng là 2 lần crawl của CÙNG 1 văn bản thì chỉ giữ cái xếp cao
    hơn, rồi LẤP ĐẦY slot còn lại bằng ứng viên hạng thấp hơn — thay vì lãng phí 1 trong 5 slot cho
    nội dung ĐÃ có. Vẫn trả về doc_id GỐC nên tương thích 100% format nộp bài.
    Case thật đã ghi nhận: câu hỏi 13366, doc_id 222242 + 276631 cùng là Luật Khám bệnh chữa bệnh 2023
    chiếm 2/5 slot, đẩy 1 tài liệu ground-truth đúng xuống hạng 6 và rớt khỏi top-5."""
    best: Dict[str, float] = {}
    for chunk_idx, score in rrf_score.items():
        doc_id = doc_ids[chunk_idx]
        if doc_id not in best or score > best[doc_id]:
            best[doc_id] = score
    ranked = sorted(best.items(), key=lambda x: -x[1])
    if not group_of:
        return [d for d, _ in ranked[:top_k]]

    out: List[str] = []
    seen_groups = set()
    for doc_id, _ in ranked:
        g = group_of.get(doc_id)
        if g is not None:
            if g in seen_groups:
                continue  # đã có 1 bản của CÙNG văn bản này -> bỏ qua, không phí slot
            seen_groups.add(g)
        out.append(doc_id)
        if len(out) >= top_k:
            break
    return out


def load_dup_groups(path: Optional[str]) -> Optional[Dict[str, int]]:
    """Đọc data/processed/duplicate_doc_groups.json -> {doc_id: group_index}."""
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    groups = data["groups"] if isinstance(data, dict) else data
    out: Dict[str, int] = {}
    for gi, members in enumerate(groups):
        for doc_id in members:
            out[str(doc_id)] = gi
    print(f"  dedup: {len(groups)} nhóm trùng lặp / {len(out)} doc_id (từ {path})")
    return out


def cmd_fuse(args) -> None:
    rank_paths = parse_named(args.rank, "rank")
    if not rank_paths:
        print("[lỗi] Cần ít nhất 1 --rank ten=duong_dan", file=sys.stderr)
        sys.exit(1)

    weight_grids_raw = parse_named(args.weights, "weights")
    top_n_raw = parse_named(args.use_top_n, "use-top-n")

    # Nguồn KHÔNG khai báo --weights coi như CỐ ĐỊNH 1.0 (thường là model neo, vd v1).
    weight_grids: Dict[str, List[float]] = {}
    for name in rank_paths:
        if name in weight_grids_raw:
            weight_grids[name] = [float(x) for x in weight_grids_raw[name].split(",") if x.strip() != ""]
        else:
            weight_grids[name] = [1.0]
    use_top_n = {name: int(top_n_raw.get(name, args.default_top_n)) for name in rank_paths}

    chunks = load_chunks(args.chunks)
    doc_ids = [c["doc_id"] for c in chunks]
    # --no-eval: dùng cho PUBLIC TEST (không có đáp án). Khi đó chỉ sinh prediction, KHÔNG chấm điểm.
    ground_truth = None if args.no_eval else load_ground_truth(args.ground_truth)
    dup_group_of = load_dup_groups(args.dedup_groups)

    loaded = {}
    loaded_scores = {}
    ref_qids = None
    for name, path in rank_paths.items():
        if args.mode == "score":
            qids, ranks, scores = load_ranks_scores(path)
            if scores is None:
                print(
                    f"[lỗi] {name}: file {path!r} KHÔNG có điểm đã lưu (sinh trước 2026-08-28, hoặc\n"
                    f"      lệnh rank cũ chưa có --scores). 'fuse --mode score' BẮT BUỘC cần scores.\n"
                    f"      Chạy lại lệnh rank/rank-bm25/rank-reranker tương ứng để sinh lại file mới.",
                    file=sys.stderr,
                )
                sys.exit(1)
            loaded_scores[name] = scores
        else:
            qids, ranks = load_ranks(path)
        if ref_qids is None:
            ref_qids = qids
        elif qids != ref_qids:
            print(
                f"[lỗi] File thứ hạng {path!r} có danh sách/thứ tự question_id KHÁC các file trước — "
                f"chắc chắn sẽ fuse SAI câu hỏi. Sinh lại Pha 1 với CÙNG --questions cho mọi nguồn.",
                file=sys.stderr,
            )
            sys.exit(1)
        if ranks.shape[1] < use_top_n[name]:
            print(
                f"[lỗi] {name}: file chỉ có top-{ranks.shape[1]} nhưng yêu cầu dùng top-{use_top_n[name]}. "
                f"Sinh lại Pha 1 với --top-n lớn hơn.",
                file=sys.stderr,
            )
            sys.exit(1)
        loaded[name] = ranks
        print(f"  {name:12s} top-{ranks.shape[1]} (dùng {use_top_n[name]}), weight grid = {weight_grids[name]}")

    if args.mode == "score":
        rrf_k_grid = [None]  # không dùng trong chế độ score, giữ 1 phần tử để combos không nhân đôi vô ích
        print(f"Chế độ: score (CombSUM, chuẩn hoá per-query bằng '{args.score_norm}')")
    else:
        rrf_k_grid = [int(x) for x in str(args.rrf_k).split(",") if x.strip() != ""]
    combos = [
        (dict(zip(weight_grids.keys(), vals)), k)
        for vals in itertools.product(*weight_grids.values())
        for k in rrf_k_grid
    ]
    grid_desc = "mode=score" if args.mode == "score" else f"rrf-k{rrf_k_grid}"
    print(f"\nTổng {len(combos)} tổ hợp (trọng số x {grid_desc}) x {len(ref_qids)} câu hỏi, top-k={args.top_k}\n")

    if args.no_eval and len(combos) != 1:
        print(
            f"[lỗi] --no-eval (public test, KHÔNG có đáp án) yêu cầu ĐÚNG 1 tổ hợp, đang có {len(combos)}.\n"
            f"      Truyền mỗi tham số 1 giá trị duy nhất (vd --weights harrier=0.8 --rrf-k 8) — sweep\n"
            f"      chỉ có ý nghĩa trên tập CÓ đáp án (val_split_20).",
            file=sys.stderr,
        )
        sys.exit(1)

    results = []
    best_predictions = None
    best_recall = -1.0
    for ci, (weights, rrf_k) in enumerate(combos, start=1):
        predictions: Dict[str, List[str]] = {}
        for qi, qid in enumerate(ref_qids):
            rank_rows = {name: loaded[name][qi] for name in loaded}
            if args.mode == "score":
                score_rows = {name: loaded_scores[name][qi] for name in loaded_scores}
                combined = fuse_one_question_by_score(rank_rows, score_rows, weights, use_top_n, args.score_norm)
            else:
                combined = fuse_one_question(rank_rows, weights, use_top_n, rrf_k)
            predictions[qid] = aggregate_max(combined, doc_ids, args.top_k, dup_group_of)
        label = " ".join(f"{k}={v:g}" for k, v in weights.items())
        label += f" rrf_k={rrf_k}" if args.mode == "rrf" else f" mode=score norm={args.score_norm}"
        if args.no_eval:
            best_predictions = predictions
            n_bad = sum(1 for docs in predictions.values() if len(docs) > args.top_k or len(set(docs)) != len(docs))
            print(f"  {label:56s} -> {len(predictions)} câu hỏi, vi phạm định dạng: {n_bad}")
            if n_bad:
                print("[lỗi] Có câu hỏi >5 doc hoặc trùng doc_id — SAI quy định, dừng lại.", file=sys.stderr)
                sys.exit(1)
            continue
        r = evaluate_predictions(ground_truth, predictions)
        results.append({"weights": dict(weights), "rrf_k": rrf_k,
                        "avg_recall": r["avg_recall"], "avg_precision": r["avg_precision"]})
        if r["avg_recall"] > best_recall:
            best_recall, best_predictions = r["avg_recall"], predictions
        print(f"  [{ci:>3d}/{len(combos)}] {label:56s} Recall={r['avg_recall']:.4f} Precision={r['avg_precision']:.4f}")

    if args.no_eval:
        if args.save_best_predictions and best_predictions is not None:
            Path(args.save_best_predictions).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_best_predictions, "w", encoding="utf-8") as f:
                json.dump({q: {"answer": d} for q, d in best_predictions.items()}, f, ensure_ascii=False, indent=2)
            print(f"\nĐã lưu prediction (public test, chưa chấm điểm): {args.save_best_predictions}")
        else:
            print("[cảnh báo] --no-eval nhưng KHÔNG truyền --save-best-predictions -> không lưu gì cả.",
                  file=sys.stderr)
        return

    # Baseline = tổ hợp MỌI nguồn PHỤ (= nguồn có --weights truyền vào, không tính nguồn neo như v1
    # mặc định cố định 1.0) đều = 0 VÀ rrf_k = 60 (cấu hình chính thức #11b). Nếu lưới rrf-k không có
    # 60 thì lấy tổ hợp weight-0 tốt nhất làm mốc thay thế.
    #
    # ⚠️ BUG ĐÃ SỬA (2026-08-28, phát hiện THẬT khi chạy fine-sweep hướng #1 — KHÔNG phải suy đoán):
    # bản CŨ lọc `if 0.0 in weight_grids[name]` NGAY TRONG all(...) — khi lưới sweep của MỌI nguồn
    # phụ đều KHÔNG chứa 0.0 (vd fine-sweep quanh 1 đỉnh, chỉ còn giá trị dương), generator rỗng và
    # `all([])` = True MỘT CÁCH VÔ NGHĨA cho MỌI dòng, khiến zero_combos = TOÀN BỘ kết quả -> baseline
    # bị chọn = chính tổ hợp TỐT NHẤT -> in ra "Δ +0.0000" SAI HOÀN TOÀN (đã xảy ra thật: fine-sweep
    # hướng #1 in "Baseline 0.9368 ... Δ +0.0000" trong khi baseline THẬT đang dùng là 0.9319 của RRF
    # chính thức, không liên quan gì tới sweep này). Sửa: nếu MỘT nguồn phụ nào đó KHÔNG sweep qua 0.0,
    # KHÔNG có tổ hợp baseline hợp lệ trong lần chạy này — báo rõ, KHÔNG tự chọn liều một dòng bất kỳ.
    optional_names = list(weight_grids_raw.keys())  # nguồn NEO (không truyền --weights) luôn cố định
    missing_zero = [name for name in optional_names if 0.0 not in weight_grids[name]]
    if missing_zero:
        print(
            f"\n[cảnh báo] Lưới KHÔNG chứa 0.0 cho: {missing_zero} -> KHÔNG có tổ hợp baseline hợp lệ "
            f"('mọi nguồn phụ = 0') trong lần sweep NÀY, nên KHÔNG tự tính được Δ. Đây là chuyện BÌNH "
            f"THƯỜNG khi sweep lưới mịn quanh 1 đỉnh đã biết (không cần quét lại 0.0) — tự so sánh "
            f"THỦ CÔNG kết quả tốt nhất bên dưới với baseline đã biết từ lần sweep TRƯỚC (bao gồm cả "
            f"0.0).",
            file=sys.stderr,
        )
        zero_combos = []
    else:
        zero_combos = [r for r in results if all(r["weights"][name] == 0.0 for name in optional_names)]
    baseline = None
    for r in zero_combos:
        if r["rrf_k"] == 60:
            baseline = r
            break
    if baseline is None and zero_combos:
        baseline = max(zero_combos, key=lambda r: r["avg_recall"])

    results_sorted = sorted(results, key=lambda r: -r["avg_recall"])
    print("\n" + "=" * 84)
    print(f"{'Trọng số + rrf_k':56s} {'Recall':>9s} {'Precision':>10s}")
    print("-" * 84)
    for r in results_sorted[: args.show_top]:
        label = " ".join(f"{k}={v:g}" for k, v in r["weights"].items())
        label += f" rrf_k={r['rrf_k']}" if args.mode == "rrf" else f" mode=score norm={args.score_norm}"
        mark = "  <-- baseline" if (baseline is not None and r is baseline) else ""
        print(f"{label:56s} {r['avg_recall']:>9.4f} {r['avg_precision']:>10.4f}{mark}")
    print("=" * 84)

    best = results_sorted[0]
    if baseline is not None:
        delta = best["avg_recall"] - baseline["avg_recall"]
        print(f"\nBaseline (mọi nguồn phụ = 0): Recall {baseline['avg_recall']:.4f}")
        print(f"Tốt nhất                     : Recall {best['avg_recall']:.4f}  (Δ {delta:+.4f})")
        if delta >= 0.005:
            print("\n-> Cải thiện RÕ RỆT (>= +0.005). ĐÁNG xác nhận trên public test.")
            print("   ⚠️ NHỚ: cập nhật scripts/allowed-models.md + gửi đề xuất model mới cho BTC")
            print("      (trước 10 ngày so với hạn chót private test) nếu chốt dùng ensemble.")
        elif delta > 0:
            print(f"\n-> Cải thiện NHỎ (+{delta:.4f}, dưới ngưỡng +0.005). Theo bài học entity-boost")
            print("   (local +0.0007 nhưng public -0.0010), tín hiệu nhỏ như vậy KHÔNG đủ tin cậy để")
            print("   đánh đổi 1 lượt nộp + thủ tục đề xuất model với BTC.")
        else:
            print("\n-> KHÔNG tổ hợp nào vượt baseline -> ĐÓNG hướng ensemble, giữ nguyên cấu hình v1.")

    if args.save_best_predictions and best_predictions is not None:
        Path(args.save_best_predictions).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_best_predictions, "w", encoding="utf-8") as f:
            json.dump({qid: {"answer": docs} for qid, docs in best_predictions.items()}, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu prediction của tổ hợp TỐT NHẤT: {args.save_best_predictions}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(
                {"mode": args.mode, "score_norm": args.score_norm if args.mode == "score" else None,
                 "rrf_k": args.rrf_k, "top_k": args.top_k, "use_top_n": use_top_n,
                 "baseline": baseline, "best": best, "all": results_sorted},
                f, ensure_ascii=False, indent=2,
            )
        print(f"Đã lưu báo cáo: {args.report}")


# ---------------------------------------------------------------------------
# selftest cho phần fuse-theo-điểm (2026-08-28) — KHÔNG cần GPU/model/corpus thật, chạy trong vài giây.
# Theo đúng kỷ luật đã áp dụng cho bgem3_sparse.py: viết test TRƯỚC KHI giao, không suy đoán code đúng.
# ---------------------------------------------------------------------------

def cmd_selftest(args) -> None:
    import numpy as np
    import tempfile

    n_ok = 0

    def check(name: str, cond: bool):
        nonlocal n_ok
        status = "OK" if cond else "HỎNG"
        print(f"  [{status}] {name}")
        if not cond:
            raise SystemExit(f"[lỗi] selftest thất bại tại: {name}")
        n_ok += 1

    print("=== 1. save_ranks/load_ranks TƯƠNG THÍCH NGƯỢC (file KHÔNG có scores) ===")
    with tempfile.TemporaryDirectory() as td:
        p_old = str(Path(td) / "old.npz")
        ranks = np.array([[3, 1, 2], [0, 2, 1]], dtype=np.int32)
        save_ranks(p_old, ["q1", "q2"], ranks)  # KHÔNG truyền scores -> giống file sinh trước 2026-08-28
        qids, r2 = load_ranks(p_old)
        check("load_ranks() cũ vẫn đọc đúng ranks", qids == ["q1", "q2"] and np.array_equal(r2, ranks))
        qids3, r3, s3 = load_ranks_scores(p_old)
        check("load_ranks_scores() trên file cũ -> scores=None (không KeyError)", s3 is None)

        print("\n=== 2. save_ranks/load_ranks_scores VỚI scores (file mới) ===")
        p_new = str(Path(td) / "new.npz")
        scores = np.array([[0.9, 0.5, 0.4], [0.7, 0.6, 0.1]], dtype=np.float32)
        save_ranks(p_new, ["q1", "q2"], ranks, scores=scores)
        qids4, r4, s4 = load_ranks_scores(p_new)
        check("ranks round-trip đúng", np.array_equal(r4, ranks))
        check("scores round-trip đúng (sai số float32)", np.allclose(s4, scores, atol=1e-6))
        qids5, r5 = load_ranks(p_new)
        check("load_ranks() (đường cũ) vẫn đọc được file MỚI, bỏ qua key scores thừa",
              qids5 == ["q1", "q2"] and np.array_equal(r5, ranks))

    print("\n=== 3. normalize_scores ===")
    mm = normalize_scores(np.array([1.0, 2.0, 4.0]), "minmax")
    check("minmax: min->0, max->1", abs(mm[0] - 0.0) < 1e-9 and abs(mm[2] - 1.0) < 1e-9)
    check("minmax: điểm giữa đúng tỉ lệ (1/3)", abs(mm[1] - 1.0 / 3.0) < 1e-9)
    zs = normalize_scores(np.array([1.0, 2.0, 3.0]), "zscore")
    check("zscore: trung bình ≈ 0", abs(float(zs.mean())) < 1e-9)
    const = normalize_scores(np.array([5.0, 5.0, 5.0]), "minmax")
    check("minmax: mọi điểm bằng nhau -> toàn 0 (không chia 0/NaN)",
          np.all(const == 0.0) and not np.any(np.isnan(const)))
    const_z = normalize_scores(np.array([5.0, 5.0, 5.0]), "zscore")
    check("zscore: mọi điểm bằng nhau -> toàn 0 (không chia 0/NaN)",
          np.all(const_z == 0.0) and not np.any(np.isnan(const_z)))

    print("\n=== 4. fuse_one_question_by_score (CombSUM tay, đối chiếu công thức) ===")
    # nguồn A xếp hạng chunk [10, 20, 30] với điểm [0.9, 0.6, 0.3] (minmax -> [1.0, 0.5, 0.0])
    # nguồn B xếp hạng chunk [20, 10]     với điểm [5.0, 1.0]      (minmax -> [1.0, 0.0])
    # w_A=1.0, w_B=0.5 -> score(10) = 1.0*1.0 + 0.5*0.0 = 1.0
    #                     score(20) = 1.0*0.5 + 0.5*1.0 = 1.0
    #                     score(30) = 1.0*0.0            = 0.0   (B không xếp hạng 30 -> không đóng góp)
    rank_rows = {"A": np.array([10, 20, 30]), "B": np.array([20, 10])}
    score_rows = {"A": np.array([0.9, 0.6, 0.3]), "B": np.array([5.0, 1.0])}
    comb = fuse_one_question_by_score(rank_rows, score_rows, {"A": 1.0, "B": 0.5},
                                      {"A": 3, "B": 2}, "minmax")
    check("chunk 10: 1.0*1.0 + 0.5*0.0 = 1.0", abs(comb[10] - 1.0) < 1e-9)
    check("chunk 20: 1.0*0.5 + 0.5*1.0 = 1.0", abs(comb[20] - 1.0) < 1e-9)
    check("chunk 30: chỉ nguồn A, 1.0*0.0 = 0.0", abs(comb[30] - 0.0) < 1e-9)

    comb_w0 = fuse_one_question_by_score(rank_rows, score_rows, {"A": 1.0, "B": 0.0},
                                         {"A": 3, "B": 2}, "minmax")
    check("weight B=0 -> B KHÔNG đóng góp (chỉ có key của A trong kết quả)",
          set(comb_w0.keys()) == {10, 20, 30})

    print("\n=== 5. Nhánh RRF CŨ không bị đổi hành vi (regression guard) ===")
    rrf = fuse_one_question({"A": np.array([10, 20, 30])}, {"A": 1.0}, {"A": 3}, 60)
    expected = {10: 1.0 / 61, 20: 1.0 / 62, 30: 1.0 / 63}
    check("RRF 1 nguồn khớp đúng công thức 1/(k+rank)",
          all(abs(rrf[k] - v) < 1e-9 for k, v in expected.items()))

    print(f"\n=== ĐẠT {n_ok}/{n_ok} kiểm tra — logic fuse-theo-điểm và tương thích ngược đều ĐÚNG ===")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
        p.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
        p.add_argument("--limit-questions", type=int, default=None)
        p.add_argument("--question-batch-size", type=int, default=64)
        p.add_argument("--top-n", type=int, default=200,
                       help="Số chunk lưu lại mỗi câu hỏi. Mặc định 200 (>100 để Pha 2 thử được cả "
                            "top-n lớn hơn mà KHÔNG phải chạy lại Pha 1).")
        p.add_argument("--out", required=True)
        p.add_argument("--name", default=None, help="Chỉ để ghi nhớ, không ảnh hưởng kết quả.")

    p_rank = sub.add_parser("rank", help="Pha 1: sinh thứ hạng cho 1 model embedding (dùng cache có sẵn)")
    add_common(p_rank)
    p_rank.add_argument("--embedding-model", required=True)
    p_rank.add_argument("--embedding-cache", required=True)
    p_rank.add_argument("--allow-encode", action="store_true",
                        help="Cho phép encode lại corpus nếu thiếu cache (MẶC ĐỊNH TẮT - tốn nhiều giờ GPU).")
    p_rank.add_argument("--batch-size", type=int, default=64)
    p_rank.add_argument("--device", default="auto")
    p_rank.add_argument("--query-prefix", default="")
    p_rank.add_argument("--passage-prefix", default="")
    p_rank.add_argument("--query-instruct", default=None,
                        help="Chỉ truyền phần {task}; script tự ghép 'Instruct: {task}\\nQuery: ' "
                             "GIỐNG HỆT full_pipeline_eval.py.")
    p_rank.add_argument("--query-prompt-name", default=None)
    p_rank.add_argument("--passage-prompt-name", default=None)
    p_rank.add_argument("--max-seq-length", type=int, default=2048)
    p_rank.add_argument("--use-segmented-text", action="store_true",
                        help="Thêm 2026-08-28 cho model PhoBERT/RoBERTa family (vd DEk21_hcmute_embedding) "
                             "cần input ĐÃ TÁCH TỪ — dùng passage_segmented/question_segmented (đã có sẵn "
                             "trong data) thay vì passage/question thô. KHÔNG dùng cho v1/harrier.")
    p_rank.add_argument("--trust-remote-code", action="store_true")
    p_rank.add_argument("--fp16-inference", dest="fp16_inference", action="store_true", default=True)
    p_rank.add_argument("--no-fp16-inference", dest="fp16_inference", action="store_false")
    p_rank.set_defaults(func=cmd_rank)

    p_bm25 = sub.add_parser("rank-bm25", help="Pha 1: sinh thứ hạng cho nhánh BM25 (không cần GPU/model)")
    add_common(p_bm25)
    p_bm25.set_defaults(func=cmd_rank_bm25)

    p_rr = sub.add_parser(
        "rank-reranker",
        help="Pha 1: sinh thứ hạng từ cross-encoder reranker, để dùng như MỘT NGUỒN trong RRF "
             "(KHÔNG thay thế xếp hạng như full_pipeline_eval.py đang làm)",
    )
    add_common(p_rr)
    p_rr.add_argument("--reranker-model", required=True,
                      help="vd models/local_train/reranker-ft, models/public_train/reranker-ft-full, "
                           "hoặc BAAI/bge-reranker-v2-m3")
    p_rr.add_argument("--candidates", required=True,
                      help="File thứ hạng (.npz) làm nguồn ứng viên để chấm lại, thường là ranks_v1.npz")
    p_rr.add_argument("--batch-size", type=int, default=64)
    p_rr.add_argument("--device", default="auto")
    p_rr.add_argument("--max-length", type=int, default=None,
                      help="Cắt độ dài cặp (câu hỏi, đoạn văn). Chi phí attention tăng theo BÌNH PHƯƠNG "
                           "độ dài nên hạ từ 512 xuống 256 nhanh gần 4x cho phần attention. Mặc định: "
                           "giữ nguyên cấu hình của checkpoint.")
    p_rr.add_argument("--fp16-inference", dest="fp16_inference", action="store_true", default=True)
    p_rr.add_argument("--no-fp16-inference", dest="fp16_inference", action="store_false")
    p_rr.set_defaults(func=cmd_rank_reranker, top_n=100, question_batch_size=16)

    p_fuse = sub.add_parser("fuse", help="Pha 2: fuse RRF nhiều nhánh + sweep trọng số (không cần GPU)")
    p_fuse.add_argument("--rank", action="append", required=True, metavar="TEN=DUONG_DAN")
    p_fuse.add_argument("--weights", action="append", default=[], metavar="TEN=0,0.1,0.2",
                        help="Lưới trọng số cho 1 nguồn. Nguồn KHÔNG khai báo mặc định CỐ ĐỊNH 1.0.")
    p_fuse.add_argument("--use-top-n", action="append", default=[], metavar="TEN=100",
                        help="Số hạng dùng cho từng nguồn khi fuse (mặc định --default-top-n).")
    p_fuse.add_argument("--default-top-n", type=int, default=100,
                        help="Mặc định 100 = ĐÚNG dense-top-n/bm25-top-n của cấu hình chính thức.")
    p_fuse.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    p_fuse.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    p_fuse.add_argument("--dedup-groups", default=None,
                        help="data/processed/duplicate_doc_groups.json — bật hậu-xử-lý dedup+backfill: "
                             "2 doc_id là 2 lần crawl CÙNG 1 văn bản thì chỉ giữ 1, lấp slot còn lại "
                             "bằng ứng viên hạng thấp hơn (thêm 2026-08-23).")
    p_fuse.add_argument("--no-eval", action="store_true",
                        help="Dùng cho PUBLIC TEST (không có đáp án): chỉ sinh prediction, không chấm "
                             "điểm. Bắt buộc ĐÚNG 1 tổ hợp trọng số + rrf-k (đã chốt từ sweep trên val).")
    p_fuse.add_argument("--rrf-k", default="60", metavar="60,30,10",
                        help="Nhận NHIỀU giá trị cách nhau bởi dấu phẩy - rrf-k được sweep CÙNG trọng số. "
                             "rrf-k nhỏ làm đường cong RRF DỐC hơn -> nguồn phụ chỉ 'chen' được vào top-5 "
                             "bằng những ứng viên nó xếp RẤT CAO, phần đuôi rác bị loại - xem docstring. "
                             "CHỈ dùng khi --mode rrf (mặc định).")
    p_fuse.add_argument("--mode", choices=["rrf", "score"], default="rrf",
                        help="'rrf' (mặc định, KHÔNG đổi hành vi cũ) = cộng theo 1/(k+rank). "
                             "'score' (thêm 2026-08-28) = CombSUM trên điểm thô đã chuẩn hoá per-query "
                             "— cần file rank có lưu scores (mọi lệnh rank/rank-bm25/rank-reranker mới "
                             "đều tự lưu; file .npz cũ trước 2026-08-28 sẽ báo lỗi rõ ràng, chạy lại rank).")
    p_fuse.add_argument("--score-norm", choices=["minmax", "zscore"], default="minmax",
                        help="Cách chuẩn hoá điểm mỗi nguồn về cùng thang trước khi cộng, CHỈ dùng khi "
                             "--mode score. minmax: co giãn [0,1] theo min/max của top-N đã lưu. "
                             "zscore: (x-trung bình)/độ lệch chuẩn, ít nhạy điểm ngoại lai hơn.")
    p_fuse.add_argument("--top-k", type=int, default=5)
    p_fuse.add_argument("--show-top", type=int, default=20)
    p_fuse.add_argument("--report", default=None)
    p_fuse.add_argument("--save-best-predictions", default=None)
    p_fuse.set_defaults(func=cmd_fuse)

    p_self = sub.add_parser("selftest", help="Kiểm tra logic fuse-theo-điểm + tương thích ngược, KHÔNG cần GPU/data")
    p_self.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
