"""
CHẨN ĐOÁN model fine-tune cho Recall ~0.0046 (gần bằng ngẫu nhiên) — 2026-08-23.

Recall 0.0046 KHÔNG phải "model học kém" (học kém thì ra 0.7-0.9). Đoán ngẫu nhiên trên ~8.000
document cho ~0.0006, nên 0.0046 = model về cơ bản đã HỎNG. Cần tách bạch 4 khả năng, mỗi khả năng
có cách xử lý HOÀN TOÀN KHÁC NHAU:

  (A) TRÀN SỐ fp16 khi encode  -> embedding có NaN/inf -> similarity vô nghĩa.
      => Sửa cực rẻ: encode fp32. 9 giờ GPU KHÔNG mất trắng.
  (B) SỤP ĐỔ BIỂU DIỄN (representation collapse) -> mọi embedding gần như trùng nhau.
      => Nguyên nhân gần như chắc chắn: TripletLoss margin=5 trên embedding ĐÃ CHUẨN HOÁ. Khoảng
         cách Euclid tối đa chỉ 2 < 5 nên KHÔNG triplet nào từng "thoả mãn" -> gradient KHÔNG BAO
         GIỜ tắt -> model bị đẩy vô hạn, cuối cùng phá nát không gian biểu diễn.
         => Phải đổi loss (MultipleNegativesRankingLoss) hoặc đặt margin hợp lý (~0.5).
  (C) MODEL LƯU HỎNG (thiếu module Pooling/Normalize, sai config).
      => So modules.json với model gốc.
  (D) Model vẫn tốt, LỖI NẰM Ở SCRIPT ĐO.
      => Nếu (A)(B)(C) đều sạch thì phải soi lại quick_eval_finetune.py.

Chạy: python scripts/diagnose_ft_model.py --model models/aiteamvn-ft-light
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/aiteamvn-ft-light")
    p.add_argument("--base-model", default="AITeamVN/Vietnamese_Embedding")
    p.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    p.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    p.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    p.add_argument("--n-probe", type=int, default=200, help="Số chunk ngẫu nhiên để đo sụp đổ")
    args = p.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer
    from src.evaluation.full_pipeline_eval import load_chunks, load_questions
    from src.evaluation.evaluate import load_ground_truth

    print("=" * 74)
    print("(C) KIỂM TRA CẤU TRÚC MODEL ĐÃ LƯU")
    print("=" * 74)
    mj = Path(args.model) / "modules.json"
    if mj.exists():
        mods = [m["type"].split(".")[-1] for m in json.loads(mj.read_text(encoding="utf-8"))]
        print(f"  modules.json: {mods}")
        if "Normalize" not in mods:
            print("  ⚠️ THIẾU Normalize! (model gốc CÓ) -> nghi (C)")
        else:
            print("  ✓ Có đủ Transformer + Pooling + Normalize")
    else:
        print("  ⚠️ KHÔNG có modules.json -> nghi (C)")

    gt = load_ground_truth(args.ground_truth)
    questions = load_questions(args.questions)
    chunks = load_chunks(args.chunks)
    doc_ids = [str(c["doc_id"]) for c in chunks]

    # Lấy 1 câu hỏi có đáp án + chunk đúng của nó + vài chunk ngẫu nhiên
    rng = np.random.default_rng(42)
    probe_q, gold_idx = None, None
    for q in questions:
        qid = str(q["question_id"])
        item = gt.get(qid)
        gold = set(str(x) for x in (item if isinstance(item, list) else item.get("answer", [])))
        cand = [i for i, d in enumerate(doc_ids) if d in gold]
        if cand:
            probe_q, gold_idx = q["question"], cand[0]
            break
    rand_idx = rng.choice(len(chunks), size=args.n_probe, replace=False)
    texts = [chunks[gold_idx]["passage"]] + [chunks[int(i)]["passage"] for i in rand_idx]

    for label, path in (("BASE (gốc)", args.base_model), ("FINE-TUNE", args.model)):
        print()
        print("=" * 74)
        print(f"{label}: {path}")
        print("=" * 74)
        m = SentenceTransformer(path, device="cuda")
        m.max_seq_length = 2048

        for prec in ("fp16", "fp32"):
            mm = m.half() if prec == "fp16" else m.float()
            emb = mm.encode(texts, batch_size=16, convert_to_numpy=True,
                            normalize_embeddings=False, show_progress_bar=False).astype(np.float32)
            qe = mm.encode([probe_q], convert_to_numpy=True,
                           normalize_embeddings=False, show_progress_bar=False).astype(np.float32)

            n_nan = int(np.isnan(emb).sum() + np.isnan(qe).sum())
            n_inf = int(np.isinf(emb).sum() + np.isinf(qe).sum())
            norms = np.linalg.norm(emb, axis=1)
            print(f"\n  [{prec}] NaN={n_nan}  inf={n_inf}  "
                  f"norm: min={norms.min():.4f} trung vị={np.median(norms):.4f} max={norms.max():.4f}")
            if n_nan or n_inf:
                print(f"  ⚠️⚠️ CÓ NaN/inf ở {prec} -> ĐÚNG khả năng (A) TRÀN SỐ.")
                if prec == "fp16":
                    print("       -> Nếu fp32 sạch thì chỉ cần encode fp32, model VẪN DÙNG ĐƯỢC.")
                continue

            # (B) SỤP ĐỔ — 2 dạng KHÁC NHAU, phải kiểm tra CẢ HAI:
            #   B1. sụp về 1 ĐIỂM: mọi vector ≈ cùng nhau  -> cosine trung bình ≈ +1
            #   B2. sụp về 1 TRỤC: mọi vector ≈ ±v          -> |cosine| ≈ 1 nhưng TRUNG BÌNH có thể
            #       chỉ ~0.5 (vì lẫn cả +1 lẫn -1). Ngưỡng "trung bình > 0.95" KHÔNG bắt được dạng
            #       này — đúng ca đã xảy ra 2026-08-22 (trung bình 0.5756 nhưng model đã hỏng hoàn
            #       toàn). Vì vậy phải đo thêm |cosine| và SỐ CHIỀU HIỆU DỤNG.
            e = emb[1:] / np.maximum(np.linalg.norm(emb[1:], axis=1, keepdims=True), 1e-12)
            off = (e @ e.T)[~np.eye(len(e), dtype=bool)]
            frac_extreme = float((np.abs(off) > 0.99).mean())
            # Số chiều hiệu dụng = entropy của phổ giá trị kỳ dị (participation ratio).
            sv = np.linalg.svd(e - e.mean(0), compute_uv=False)
            p = (sv ** 2) / max((sv ** 2).sum(), 1e-12)
            eff_dim = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
            print(f"       cosine giữa chunk ngẫu nhiên: trung bình={off.mean():.4f}  "
                  f"|cos|>0.99 chiếm {frac_extreme*100:.1f}%")
            print(f"       số chiều hiệu dụng ≈ {eff_dim:.1f} / {e.shape[1]} "
                  f"(model lành mạnh: hàng chục trở lên; ≈1-2 = SỤP ĐỔ)")
            collapsed = off.mean() > 0.95 or frac_extreme > 0.5 or eff_dim < 3
            if collapsed:
                print("  ⚠️⚠️ SỤP ĐỔ BIỂU DIỄN -> khả năng (B). Không gian bị nén về 1 điểm/1 trục,")
                print("       cosine chỉ còn nhận ±1 nên KHÔNG còn thang liên tục để xếp hạng.")

            qn = qe[0] / max(np.linalg.norm(qe[0]), 1e-12)
            s_rand = e @ qn
            print(f"       cos(câu hỏi, {args.n_probe} chunk ngẫu nhiên): trung bình={s_rand.mean():.4f} "
                  f"độ lệch chuẩn={s_rand.std():.4f} max={s_rand.max():.4f}")
            if s_rand.mean() < -0.3:
                print("       ⚠️ Trung bình ÂM MẠNH: dấu hiệu negative bị đẩy về ĐỐI CỰC của query —")
                print("          đúng lời giải thoái hoá của TripletLoss khi margin > khoảng cách tối đa.")
            # LƯU Ý: KHÔNG so câu hỏi với "chunk đầu tiên của document đúng" nữa. Chunk đầu thường là
            # phần mở đầu hành chính (quốc hiệu, số hiệu, căn cứ...) chứ KHÔNG phải đoạn trả lời câu
            # hỏi — baseline (Recall 0.92) cũng chỉ xếp nó hạng 9/200, nên phép thử đó vô nghĩa.
        del m

    print("\n" + "=" * 74)
    print("KẾT LUẬN: đọc các cảnh báo ⚠️ ở trên. Nếu fp32 sạch mà fp16 có NaN -> chỉ cần đổi sang")
    print("fp32 khi encode. Nếu cosine giữa chunk ngẫu nhiên >0.95 -> sụp đổ, phải đổi loss.")
    print("=" * 74)


if __name__ == "__main__":
    main()
