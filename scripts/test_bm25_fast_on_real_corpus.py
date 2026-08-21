"""Kiểm tra --bm25-fast (bản vector hoá BM25 mới, 2026-08-17) TRÊN DỮ LIỆU THẬT của dự án, nhưng
CHỈ trên 1 PHẦN corpus trước (--limit) - để đo tốc độ/RAM thật trên MÁY BẠN trước khi cam kết chạy
full 351K chunk. ĐỘNG LỰC: --bm25-workers (song song qua tiến trình) đã gây MemoryError thật với
worker=16; bản --bm25-fast (ma trận thưa, không multiprocessing) đã verify ĐÚNG công thức trên dữ
liệu giả lập nhỏ, NHƯNG khi tự test ở quy mô ~200K document/50 triệu phần tử trong môi trường của
Claude cũng bị hết bộ nhớ (môi trường đó giới hạn RAM thấp hơn máy bạn nhiều, nên KHÔNG chắc máy bạn
cũng gặp vấn đề - nhưng cần tự đo trên máy bạn trước khi tin, không đoán mò).

Cách dùng (từ thư mục gốc UIT-data-science-2026):
    python scripts/test_bm25_fast_on_real_corpus.py --limit 50000
    python scripts/test_bm25_fast_on_real_corpus.py --limit 150000
    python scripts/test_bm25_fast_on_real_corpus.py            (KHÔNG giới hạn - full 351K, chỉ chạy
                                                                  sau khi đã thử --limit an toàn)

QUAN TRỌNG: mở Task Manager (Ctrl+Shift+Esc) trước khi chạy, để mắt tới cột Memory trong lúc script
chạy - nếu thấy RAM tăng vọt gần chạm giới hạn máy, Ctrl+C dừng ngay (an toàn, không hỏng gì, chỉ
chưa xây xong ma trận).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.evaluation.full_pipeline_eval import build_bm25_fast_matrix, bm25_scores_batch_fast, _naive_tokenize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ dùng N chunk ĐẦU của corpus (test nhanh/an toàn trước khi chạy full)")
    parser.add_argument("--n-questions", type=int, default=20, help="Số câu hỏi dùng để so sánh độ chính xác")
    args = parser.parse_args()

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("Thiếu rank_bm25. Chạy: pip install rank_bm25", file=sys.stderr)
        sys.exit(1)

    print(f"Đọc {args.chunks}...")
    chunks = []
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
            if args.limit and len(chunks) >= args.limit:
                break
    print(f"Dùng {len(chunks)} chunk (limit={args.limit or 'KHÔNG - full corpus'}).")

    print("Tokenize cho BM25 (giống build_bm25_index() thật)...")
    t0 = time.time()
    tokenized = [
        c["passage_segmented"].split() if c.get("passage_segmented") else _naive_tokenize(c["passage"])
        for c in chunks
    ]
    print(f"  Xong sau {time.time()-t0:.1f}s.")

    print("Build BM25Okapi gốc (rank_bm25) - bước NÀY vốn đã có sẵn trong pipeline, không phải chi phí MỚI...")
    t0 = time.time()
    bm25 = BM25Okapi(tokenized)
    t_build_orig = time.time() - t0
    print(f"  Xong sau {t_build_orig:.1f}s.")

    print("\n>>> BƯỚC QUAN TRỌNG - theo dõi Task Manager (cột Memory) trong lúc bước này chạy <<<")
    print(">>> Ctrl+C ngay nếu RAM tăng vọt gần chạm giới hạn máy - an toàn, không hỏng gì. <<<\n")
    t0 = time.time()
    bm25_fast = build_bm25_fast_matrix(bm25)
    t_build_fast = time.time() - t0
    print(f"build_bm25_fast_matrix xong sau {t_build_fast:.1f}s (thêm {t_build_fast:.1f}s so với "
          f"chi phí BM25Okapi gốc {t_build_orig:.1f}s vốn đã có sẵn).")

    print(f"\nĐọc {args.n_questions} câu hỏi đầu từ {args.questions} để so sánh...")
    questions = []
    with open(args.questions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            questions.append(q)
            if len(questions) >= args.n_questions:
                break

    q_tokens_batch = [
        q["question_segmented"].split() if q.get("question_segmented") else _naive_tokenize(q["question"])
        for q in questions
    ]

    t0 = time.time()
    for t in q_tokens_batch:
        bm25.get_scores(t)
    t_orig_per_q = (time.time() - t0) / len(q_tokens_batch)

    t0 = time.time()
    scores_fast = bm25_scores_batch_fast(bm25_fast, q_tokens_batch)
    t_fast_total = time.time() - t0

    print("\n" + "=" * 70)
    print(f"Tốc độ get_scores() GỐC (rank_bm25, tuần tự)   : {t_orig_per_q*1000:.1f} ms/câu")
    print(f"Tốc độ bm25_scores_batch_fast (--bm25-fast)     : {t_fast_total/len(q_tokens_batch)*1000:.2f} ms/câu (tính CẢ batch)")
    print(f"Tốc độ nhanh hơn khoảng: ~{t_orig_per_q/(t_fast_total/len(q_tokens_batch)):.0f} lần")
    print("=" * 70)

    print("\nSo sánh ĐỘ CHÍNH XÁC (top-5 doanh nghiệp/chunk index có khớp nhau không):")
    import numpy as np

    all_match = True
    max_rel_diff = 0.0
    for i, t in enumerate(q_tokens_batch):
        s_orig = np.asarray(bm25.get_scores(t))
        s_fast = scores_fast[i]
        top5_orig = np.argsort(-s_orig)[:5]
        top5_fast = np.argsort(-s_fast)[:5]
        match = np.array_equal(top5_orig, top5_fast)
        all_match = all_match and match
        denom = np.maximum(np.abs(s_orig), 1e-8)
        rel = (np.abs(s_orig - s_fast) / denom).max()
        max_rel_diff = max(max_rel_diff, rel)
        if not match:
            print(f"  ⚠️ Câu {i}: top-5 KHÁC NHAU! orig={top5_orig.tolist()} fast={top5_fast.tolist()}")

    print(f"\nTất cả top-5 khớp nhau: {all_match}")
    print(f"Sai số tương đối lớn nhất: {max_rel_diff:.2e}")
    if all_match:
        print("✅ PASS trên dữ liệu thật (subset đã test) - an toàn để thử quy mô lớn hơn hoặc full corpus.")
    else:
        print("❌ CÓ SAI LỆCH - DỪNG LẠI, gửi lại toàn bộ output này để kiểm tra trước khi dùng --bm25-fast thật.")
        sys.exit(1)


if __name__ == "__main__":
    main()
