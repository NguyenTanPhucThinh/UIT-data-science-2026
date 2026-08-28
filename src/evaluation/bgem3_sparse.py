"""Nhánh SPARSE (lexical) của BAAI/bge-m3 làm nguồn thứ 4 cho ensemble RRF — thêm 2026-08-27.

ĐỘNG LỰC (xem docs/submission_log_task1.md, mục "Hướng còn lại"):
    Pipeline hiện tại có 3 nhánh: v1 (dense, XLM-R), harrier (dense, Qwen), BM25 (lexical THỐNG KÊ).
    bge-m3 sparse là lexical HỌC ĐƯỢC: model tự học trọng số cho từng token thay vì dùng tần suất
    tf-idf. Đây là LOẠI TÍN HIỆU MỚI, không phải biến thể của cái đã có — khác hẳn việc thêm một
    model dense nữa (đã thử 5 lần, đều thua v1).

    Lịch sử dự án cho thấy MỌI lần tăng điểm đều đến từ việc đổi TÍN HIỆU vào fusion hoặc CÁCH trộn
    chúng, chưa lần nào đến từ thêm tầng xử lý hay train lại. Hướng này thuộc đúng nhóm đó.

TUÂN THỦ: BAAI/bge-m3 ~568M tham số. Tổng ensemble v1(568M) + harrier(596M) + bge-m3(568M)
    = ~1,73 tỷ, VẪN DƯỚI hạn mức 4 tỷ (quy định mục 5). NHƯNG phải khai báo model này trong
    `scripts/allowed-models.md` và gửi đề xuất BTC trước 10 ngày hạn chót private test (mục 7)
    NẾU quyết định dùng nó trong pipeline chính thức.

VÌ SAO KHÔNG DÙNG THƯ VIỆN FlagEmbedding:
    Cách "chính chủ" là `FlagEmbedding.BGEM3FlagModel`. Nhưng cài FlagEmbedding kéo theo ràng buộc
    phiên bản transformers/peft riêng, RỦI RO làm hỏng môi trường .venv đang chạy tốt cho toàn bộ
    pipeline (sentence-transformers + fine-tune). Sparse head của bge-m3 thực chất CHỈ là 1 lớp
    Linear(1024 -> 1) lưu trong `sparse_linear.pt` (3,52 kB trên HuggingFace), nên script này tự
    tải đúng file đó và tự áp dụng — chỉ cần torch + transformers ĐÃ CÓ SẴN, không cài thêm gì.

CÔNG THỨC (đối chiếu source FlagEmbedding, bản inference):
    token_weights   = relu(sparse_linear(last_hidden_state))          # (B, L, 1), luôn >= 0
    sparse_emb      = zeros(B, vocab).scatter_reduce(index=input_ids,
                                                    src=token_weights, reduce="amax")
    sparse_emb[:, [cls, eos, pad, unk]] = 0                           # bỏ token đặc biệt
    score(q, d)     = Σ_t  w_q[t] * w_d[t]   (chỉ cộng trên token CHUNG) = tích vô hướng thưa

    Vì relu đảm bảo mọi trọng số >= 0, khởi tạo zeros + reduce="amax" là ĐÚNG (max với 0 không làm
    hỏng giá trị dương). Đây là chỗ dễ sai nhất nếu tự cài lại — đã kiểm chứng bằng unit test so
    với cách tính vét cạn (xem `selftest`).

DÙNG TEXT THÔ, KHÔNG DÙNG passage_segmented:
    Khác BM25 (dùng `passage_segmented` của VnCoreNLP), bge-m3 có tokenizer SentencePiece riêng và
    được train trên text THÔ. Cho ăn text đã tách từ (có dấu gạch dưới "bảo_hiểm_nhân_thọ") sẽ tách
    sai subword và làm hỏng kết quả. Script này CỐ Ý dùng `passage`/`question` thô.

QUY TRÌNH 3 BƯỚC:
    1. selftest  — kiểm tra logic sparse bằng model giả, KHÔNG cần GPU/tải model. Chạy TRƯỚC.
    2. encode    — encode corpus thành ma trận thưa, CHIA SHARD, CÓ THỂ CHẠY TIẾP nếu bị ngắt.
    3. rank      — sinh outputs/cache/ranks_bgem3sparse.npz, dùng thẳng cho `ensemble_fusion fuse`.

    Kèm `inspect` để MẮT THƯỜNG kiểm tra head hoạt động đúng (in ra token có trọng số cao nhất của
    một câu — phải là từ nội dung có nghĩa, không phải dấu câu/hư từ).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

VOCAB_FALLBACK = 250002  # XLM-RoBERTa; chỉ dùng nếu config không khai báo vocab_size


# ---------------------------------------------------------------------------
# Lõi tính sparse
# ---------------------------------------------------------------------------

def sparse_from_hidden(hidden_state, input_ids, attention_mask, sparse_linear,
                       unused_token_ids, vocab_size):
    """Tính sparse embedding (B, vocab) từ last_hidden_state.

    Tách riêng khỏi class M3Sparse để `selftest` gọi được mà KHÔNG cần tải model thật.
    """
    import torch

    # sparse_linear giữ ở float32 dù model chạy fp16 - phép nhân 1024x1 này rẻ, không đáng
    # đánh đổi rủi ro tràn số fp16 (bài học từ sự cố encode fp16 ngày 2026-08-23).
    tw = torch.relu(sparse_linear(hidden_state.float())).squeeze(-1)  # (B, L)
    tw = tw * attention_mask.to(tw.dtype)  # vị trí padding không đóng góp

    se = torch.zeros(hidden_state.shape[0], vocab_size, dtype=torch.float32, device=tw.device)
    se.scatter_reduce_(1, input_ids, tw, reduce="amax", include_self=True)
    if unused_token_ids:
        se[:, list(unused_token_ids)] = 0.0
    # detach BẮT BUỘC: sparse_linear là nn.Linear nên tham số mặc định requires_grad=True. Nếu gọi
    # hàm này NGOÀI torch.no_grad() thì tensor trả về mang theo đồ thị gradient -> .numpy() ném
    # RuntimeError. Đã dính đúng lỗi này khi viết selftest (2026-08-27).
    return se.detach()


class M3Sparse:
    def __init__(self, model_name: str, device: str, fp16: bool, max_length: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.max_length = max_length
        print(f"Load tokenizer + model: {model_name} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        if fp16 and device == "cuda":
            self.model = self.model.half()
            print("  đã cast model sang fp16 (sparse_linear VẪN giữ float32)")

        hidden = getattr(self.model.config, "hidden_size", 1024)
        self.vocab_size = int(getattr(self.model.config, "vocab_size", VOCAB_FALLBACK))

        # Tải ĐÚNG file sparse_linear.pt của repo model - đây là toàn bộ "sparse head".
        from huggingface_hub import hf_hub_download
        w_path = hf_hub_download(repo_id=model_name, filename="sparse_linear.pt")
        state = torch.load(w_path, map_location="cpu")
        self.sparse_linear = torch.nn.Linear(hidden, 1)
        self.sparse_linear.load_state_dict(state)
        self.sparse_linear = self.sparse_linear.to(device).float().eval()
        for prm in self.sparse_linear.parameters():
            prm.requires_grad_(False)
        print(f"  đã nạp sparse_linear.pt (Linear({hidden} -> 1)), vocab_size={self.vocab_size}")

        self.unused = [
            i for i in (
                self.tok.cls_token_id, self.tok.eos_token_id,
                self.tok.pad_token_id, self.tok.unk_token_id,
            ) if i is not None
        ]
        print(f"  token đặc biệt bị loại: {self.unused}")

    def encode(self, texts: List[str]):
        """Trả về tensor dày (B, vocab) trên CPU. Gọi theo batch nhỏ (vocab lớn)."""
        import torch

        enc = self.tok(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**enc)[0]
            se = sparse_from_hidden(
                hidden, enc["input_ids"], enc["attention_mask"],
                self.sparse_linear, self.unused, self.vocab_size,
            )
        return se.cpu()


def to_csr_block(dense_batch, min_weight: float):
    """(B, vocab) dày -> (indptr, indices, data) của khối CSR, chỉ giữ phần khác 0."""
    import numpy as np
    import scipy.sparse as sp

    arr = dense_batch.detach().cpu().numpy()
    if min_weight > 0:
        arr[arr < min_weight] = 0.0
    return sp.csr_matrix(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def cmd_encode(args) -> None:
    import numpy as np
    import scipy.sparse as sp

    from src.evaluation.full_pipeline_eval import load_chunks, resolve_device

    chunks = load_chunks(args.chunks)
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]
        print(f"[CHẾ ĐỘ THỬ] chỉ encode {len(chunks)} chunk đầu tiên")

    out_dir = Path(args.shard_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    shard = args.shard_size
    starts = list(range(0, len(chunks), shard))
    todo = [s for s in starts if not (out_dir / f"shard_{s:08d}.npz").exists()]
    print(f"Tổng {len(starts)} shard ({shard} chunk/shard); {len(starts)-len(todo)} shard ĐÃ CÓ, "
          f"còn {len(todo)} shard cần chạy.")
    if not todo:
        print("Không còn gì để encode. Sang bước `rank`.")
        return

    model = M3Sparse(args.model, device, args.fp16, args.max_length)

    import time
    t0 = time.time()
    for n_done, s in enumerate(todo, start=1):
        part = chunks[s : s + shard]
        texts = [c["passage"] for c in part]
        # SẮP XẾP THEO ĐỘ DÀI trước khi encode: tokenizer pad theo câu DÀI NHẤT trong batch, nên
        # trộn lẫn chunk ngắn với chunk dài làm phần lớn phép tính đổ vào padding vô ích. Đây chính
        # là thủ thuật sentence-transformers dùng sẵn bên trong (lý do encode dense corpus chỉ mất
        # ~1,5-2h dù max_length=2048). Không sắp xếp có thể chậm hơn NHIỀU LẦN.
        order = np.argsort([len(t) for t in texts], kind="stable")
        blocks, encoded_order = [], []
        for i in range(0, len(order), args.batch_size):
            sel = order[i : i + args.batch_size]
            blocks.append(to_csr_block(model.encode([texts[k] for k in sel]), args.min_weight))
            encoded_order.extend(int(k) for k in sel)
        # Trả về ĐÚNG thứ tự dòng gốc của corpus — sai bước này là mọi chỉ số chunk lệch hết.
        mat = sp.vstack(blocks, format="csr")[np.argsort(np.asarray(encoded_order))]
        np.savez_compressed(
            out_dir / f"shard_{s:08d}.npz",
            indptr=mat.indptr, indices=mat.indices, data=mat.data,
            shape=np.array(mat.shape, dtype=np.int64),
        )
        el = time.time() - t0
        eta = el / n_done * (len(todo) - n_done)
        print(f"  shard {s:08d}: {mat.shape[0]} chunk, {mat.nnz:,} phần tử khác 0 "
              f"({mat.nnz/mat.shape[0]:.0f}/chunk) — {n_done}/{len(todo)}, còn ~{eta/60:.0f} phút")

    print(f"\nXONG encode trong {(time.time()-t0)/60:.1f} phút. Shard nằm ở: {out_dir}")


def load_all_shards(shard_dir: str, n_expected: int):
    import numpy as np
    import scipy.sparse as sp

    files = sorted(Path(shard_dir).glob("shard_*.npz"))
    if not files:
        print(f"[lỗi] Không có shard nào trong {shard_dir} — chạy `encode` trước.", file=sys.stderr)
        sys.exit(1)
    mats = []
    for f in files:
        d = np.load(f)
        mats.append(sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"])))
    mat = sp.vstack(mats, format="csr")
    print(f"Đã nạp {len(files)} shard -> ma trận {mat.shape}, {mat.nnz:,} phần tử khác 0 "
          f"(~{mat.data.nbytes/1e9:.2f} GB RAM cho phần giá trị)")
    if mat.shape[0] != n_expected:
        print(
            f"[lỗi] Ma trận có {mat.shape[0]} dòng nhưng corpus có {n_expected} chunk — KHÔNG khớp.\n"
            f"      Nhiều khả năng encode bị ngắt giữa chừng hoặc đã dùng --limit-chunks.\n"
            f"      Dùng nhầm sẽ cho thứ hạng SAI HOÀN TOÀN. Chạy lại `encode` cho đủ.",
            file=sys.stderr,
        )
        sys.exit(1)
    return mat


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------

def cmd_rank(args) -> None:
    import numpy as np

    from src.evaluation.full_pipeline_eval import load_chunks, load_questions, resolve_device

    chunks = load_chunks(args.chunks)
    # PHẢI cắt corpus GIỐNG HỆT lúc encode, nếu không chốt chặn lệch số dòng sẽ nổ. Chỉ dùng khi
    # chạy thử; lần chạy thật để trống cả hai bên. (Thiếu tham số này là lý do smoke test ngày
    # 2026-08-28 dừng giữa chừng dù mọi thứ khác đều đúng.)
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]
        print(f"[CHẾ ĐỘ THỬ] chỉ dùng {len(chunks)} chunk đầu (phải KHỚP --limit-chunks lúc encode)")
    questions = load_questions(args.questions)
    if args.limit_questions:
        questions = questions[: args.limit_questions]
        print(f"[CHẾ ĐỘ THỬ] chỉ xếp hạng {len(questions)} câu hỏi đầu")

    doc_mat = load_all_shards(args.shard_dir, len(chunks))

    device = resolve_device(args.device)
    model = M3Sparse(args.model, device, args.fp16, args.max_length)

    print(f"Encode {len(questions)} câu hỏi (text THÔ, không dùng question_segmented)...")
    q_blocks = []
    for i in range(0, len(questions), args.batch_size):
        texts = [q["question"] for q in questions[i : i + args.batch_size]]
        q_blocks.append(to_csr_block(model.encode(texts), args.min_weight))
    import scipy.sparse as sp
    q_mat = sp.vstack(q_blocks, format="csr")
    print(f"  ma trận câu hỏi {q_mat.shape}, {q_mat.nnz:,} phần tử khác 0 "
          f"({q_mat.nnz/q_mat.shape[0]:.0f}/câu)")

    # Kẹp top_n theo số chunk thực có: nếu top_n > số chunk (chỉ xảy ra khi chạy thử với
    # --limit-chunks nhỏ) thì argpartition trả về ít phần tử hơn và phép gán sẽ ném
    # "could not broadcast input array" — đã dính đúng lỗi này trong test tích hợp (2026-08-27).
    n_chunk = doc_mat.shape[0]
    top_n = min(args.top_n, n_chunk)
    if top_n < args.top_n:
        print(f"[chú ý] top_n giảm {args.top_n} -> {top_n} vì corpus chỉ có {n_chunk} chunk.")
    all_ranks = np.zeros((len(questions), top_n), dtype=np.int32)
    qb = args.score_batch_size
    for start in range(0, len(questions), qb):
        qs = q_mat[start : start + qb]
        # (n_chunk x vocab) @ (vocab x b) = (n_chunk x b) — tích vô hướng thưa CHÍNH LÀ
        # công thức lexical matching của bge-m3 (tổng w_q[t]*w_d[t] trên token chung).
        scores = (doc_mat @ qs.T).toarray()
        for j in range(scores.shape[1]):
            col = scores[:, j]
            idx = np.argpartition(-col, min(top_n, len(col) - 1))[:top_n]
            all_ranks[start + j] = idx[np.argsort(-col[idx])]
        print(f"  đã chấm {min(start+qb, len(questions))}/{len(questions)} câu hỏi")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        question_ids=np.array([str(q["question_id"]) for q in questions]),
        ranks=all_ranks.astype(np.int32),
    )
    print(f"\nĐã lưu thứ hạng: {args.out}  (shape={all_ranks.shape})")

    # Recall đứng riêng — chỉ để phát hiện BUG, KHÔNG phải để quyết định giữ/bỏ hướng.
    gt = {q["question_id"]: q.get("answer_doc_ids") for q in questions}
    if any(gt.values()):
        doc_ids = [str(c["doc_id"]) for c in chunks]
        tot = 0.0
        n = 0
        for i, q in enumerate(questions):
            rel = set(str(x) for x in (q.get("answer_doc_ids") or []))
            if not rel:
                continue
            seen, top = set(), []
            for ci in all_ranks[i]:
                d = doc_ids[int(ci)]
                if d not in seen:
                    seen.add(d)
                    top.append(d)
                if len(top) >= args.top_k:
                    break
            tot += len(rel & set(top)) / len(rel)
            n += 1
        print(f"Recall@{args.top_k} ĐỨNG RIÊNG của bge-m3 sparse = {tot/n:.4f} ({n} câu có đáp án)")
        print("  ^ Chỉ dùng để BẮT BUG: nếu ra ~0.00 thì sparse head hỏng, ĐỪNG đem đi fuse.")
        print("    Thấp hơn v1 là BÌNH THƯỜNG và KHÔNG có nghĩa hướng này vô dụng — harrier cũng")
        print("    thua khi đứng riêng (0.9040 vs 0.9267) nhưng vẫn giúp ensemble (+0.0035 public).")


# ---------------------------------------------------------------------------
# inspect — kiểm tra bằng mắt
# ---------------------------------------------------------------------------

def cmd_inspect(args) -> None:
    from src.evaluation.full_pipeline_eval import load_questions, resolve_device

    # ƯU TIÊN đọc text TỪ FILE thay vì từ tham số dòng lệnh. Lý do (đã dính thật 2026-08-28):
    # PowerShell trên Windows truyền tham số non-ASCII sang Python bị hỏng mã (UTF-8 đọc thành
    # Latin-1) -> "Trường hợp" biến thành "TrÆ°á»ng há»£p" TRƯỚC KHI tới model, làm bài kiểm tra
    # bằng mắt trở nên vô nghĩa mà nhìn thoáng qua lại tưởng sparse head hỏng. Đọc từ file JSONL
    # bằng encoding="utf-8" thì không bao giờ dính lỗi này.
    if args.from_questions is not None:
        qs = load_questions(args.questions)
        if not 0 <= args.from_questions < len(qs):
            print(f"[lỗi] --from-questions phải trong [0, {len(qs)-1}]", file=sys.stderr)
            sys.exit(1)
        text = qs[args.from_questions]["question"]
    elif args.text:
        text = args.text
        if not any(ord(c) > 127 for c in text):
            print("[chú ý] --text không có ký tự tiếng Việt nào có dấu — nếu bạn VỪA gõ chữ có dấu "
                  "thì nó đã bị hỏng mã trên đường truyền. Dùng --from-questions cho chắc.")
    else:
        print("[lỗi] Cần --from-questions N (khuyến nghị) hoặc --text.", file=sys.stderr)
        sys.exit(1)

    model = M3Sparse(args.model, resolve_device(args.device), args.fp16, args.max_length)
    se = model.encode([text])[0]
    args.text = text
    vals, idx = se.topk(args.top)
    print(f"\n{args.top} token có TRỌNG SỐ CAO NHẤT cho câu:\n  {args.text!r}\n")
    for v, i in zip(vals.tolist(), idx.tolist()):
        print(f"  {model.tok.convert_ids_to_tokens([i])[0]:<24s} id={i:<8d} w={v:.4f}")
    print("\nĐÚNG nếu: phần lớn là TỪ NỘI DUNG có nghĩa (danh từ/thuật ngữ pháp lý).")
    print("SAI nếu : toàn dấu câu, <s>, </s>, <pad>, hoặc mọi trọng số xấp xỉ bằng nhau.")


# ---------------------------------------------------------------------------
# selftest — không cần GPU, không cần tải model
# ---------------------------------------------------------------------------

def cmd_selftest(args) -> None:
    import torch

    print("Kiểm tra logic sparse bằng model GIẢ (không tải bge-m3, không cần GPU)...\n")
    V, H, L = 20, 4, 6
    torch.manual_seed(0)
    lin = torch.nn.Linear(H, 1)
    hidden = torch.randn(2, L, H)
    input_ids = torch.tensor([[0, 7, 9, 7, 2, 1], [0, 5, 5, 5, 11, 2]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    unused = [0, 1, 2, 3]

    se = sparse_from_hidden(hidden, input_ids, attn, lin, unused, V)

    # Tham chiếu: vét cạn bằng vòng lặp Python thuần
    tw = (torch.relu(lin(hidden.float())).squeeze(-1) * attn).detach()
    ref = torch.zeros(2, V)
    for b in range(2):
        for pos in range(L):
            t = int(input_ids[b, pos])
            ref[b, t] = max(float(ref[b, t]), float(tw[b, pos]))
    for u in unused:
        ref[:, u] = 0.0

    assert torch.allclose(se, ref, atol=1e-6), "scatter_reduce KHÔNG khớp cách tính vét cạn"
    print("  ✓ scatter_reduce(amax) khớp chính xác cách tính vét cạn")
    assert (se[:, unused] == 0).all(), "token đặc biệt chưa bị loại"
    print("  ✓ token đặc biệt [cls, pad, eos, unk] đã bị loại về 0")
    assert (se >= 0).all(), "có trọng số âm — relu sai"
    print("  ✓ mọi trọng số >= 0 (relu đúng) -> khởi tạo zeros + amax là hợp lệ")
    # token 7 lặp 2 lần ở câu 0 -> phải lấy MAX chứ không phải tổng/giá trị cuối
    m = max(float(tw[0, 1]), float(tw[0, 3]))
    assert abs(float(se[0, 7]) - m) < 1e-6, "token lặp lại không lấy max"
    print("  ✓ token lặp lại lấy MAX (không phải tổng, không phải giá trị cuối)")
    # vị trí bị mask (pos 5 câu 0, token id 1) không đóng góp
    assert float(se[0, 1]) == 0.0
    print("  ✓ vị trí padding không đóng góp")

    # Tích vô hướng thưa == công thức lexical matching (tổng trên token CHUNG)
    import numpy as np
    import scipy.sparse as sp
    a, b = sp.csr_matrix(se[0:1].numpy()), sp.csr_matrix(se[1:2].numpy())
    dot = float((a @ b.T).toarray()[0, 0])
    manual = sum(
        float(se[0, t]) * float(se[1, t])
        for t in range(V) if se[0, t] > 0 and se[1, t] > 0
    )
    assert abs(dot - manual) < 1e-6, "tích vô hướng thưa != tổng trên token chung"
    print("  ✓ tích vô hướng thưa == Σ w_q[t]*w_d[t] trên token chung (đúng lexical matching)")

    print("\nTẤT CẢ ĐỀU ĐẠT. Logic sparse đúng — có thể chạy `encode`.")


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp_):
        sp_.add_argument("--model", default="BAAI/bge-m3")
        sp_.add_argument("--device", default="auto")
        sp_.add_argument("--fp16", action="store_true",
                         help="Chạy backbone ở fp16 (nhanh hơn ~2x trên GPU). sparse_linear LUÔN "
                              "giữ float32 nên không có rủi ro tràn số ở phần đầu ra.")
        sp_.add_argument("--max-length", type=int, default=2048,
                         help="Mặc định 2048 = GIỐNG cấu hình dense v1 đang dùng, tránh cắt cụt.")
        sp_.add_argument("--batch-size", type=int, default=8)
        sp_.add_argument("--min-weight", type=float, default=0.0,
                         help="Bỏ trọng số nhỏ hơn ngưỡng này để giảm RAM/dung lượng. Mặc định 0 "
                              "= GIỮ NGUYÊN như FlagEmbedding (không cắt). Chỉ tăng nếu thiếu RAM.")
        sp_.add_argument("--shard-dir", default="outputs/cache/bgem3_sparse_shards")

    p_e = sub.add_parser("encode", help="Bước 2: encode corpus thành ma trận thưa (CÓ THỂ CHẠY TIẾP)")
    common(p_e)
    p_e.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    p_e.add_argument("--shard-size", type=int, default=20000,
                     help="Số chunk mỗi shard. Shard đã có trên đĩa sẽ được BỎ QUA khi chạy lại — "
                          "đây là cơ chế chống mất trắng nếu job bị ngắt giữa chừng.")
    p_e.add_argument("--limit-chunks", type=int, default=None,
                     help="CHỈ để thử nhanh (vd 500). ĐỪNG dùng cho lần chạy thật.")
    p_e.set_defaults(func=cmd_encode)

    p_r = sub.add_parser("rank", help="Bước 3: sinh ranks_bgem3sparse.npz cho ensemble_fusion fuse")
    common(p_r)
    p_r.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    p_r.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    p_r.add_argument("--top-n", type=int, default=200)
    p_r.add_argument("--top-k", type=int, default=5)
    p_r.add_argument("--score-batch-size", type=int, default=16,
                     help="Số câu hỏi chấm điểm cùng lúc. Mỗi câu tốn ~1,4 MB RAM cho cột điểm "
                          "(351K chunk x float32) nên 16 câu ~ 22 MB — an toàn.")
    p_r.add_argument("--limit-questions", type=int, default=None)
    p_r.add_argument("--limit-chunks", type=int, default=None,
                     help="CHỈ để chạy thử: phải ĐẶT ĐÚNG BẰNG --limit-chunks đã dùng lúc encode, "
                          "nếu không chốt chặn lệch số dòng sẽ dừng chương trình.")
    p_r.add_argument("--out", default="outputs/cache/ranks_bgem3sparse.npz")
    p_r.set_defaults(func=cmd_rank)

    p_i = sub.add_parser("inspect", help="Kiểm tra bằng mắt: in token có trọng số cao nhất")
    common(p_i)
    p_i.add_argument("--from-questions", type=int, default=None,
                     help="KHUYẾN NGHỊ: lấy câu hỏi thứ N từ file --questions (đọc UTF-8 an toàn) "
                          "thay vì gõ --text trên dòng lệnh — tránh lỗi hỏng mã tiếng Việt của "
                          "PowerShell trên Windows.")
    p_i.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    p_i.add_argument("--text", default=None, help="Chỉ dùng khi text THUẦN ASCII. Chữ có dấu rất "
                                                 "dễ hỏng mã khi qua PowerShell — dùng --from-questions.")
    p_i.add_argument("--top", type=int, default=20)
    p_i.set_defaults(func=cmd_inspect)

    p_s = sub.add_parser("selftest", help="Bước 1: kiểm tra logic sparse (KHÔNG cần GPU/tải model)")
    p_s.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
