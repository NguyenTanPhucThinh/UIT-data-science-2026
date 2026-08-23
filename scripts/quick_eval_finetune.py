"""
ĐO NHANH model embedding vừa fine-tune, KHÔNG cần encode lại toàn bộ corpus (2026-08-22).

VẤN ĐỀ: encode đủ 351.394 chunk bằng model mới tốn vài GIỜ GPU. Nếu model mới tệ (như dòng #12) thì
đó là vài giờ đổ đi. Cần biết SỚM nó có triển vọng hay không.

CÁCH LÀM: chỉ encode một TẬP CON corpus, gồm:
  (a) MỌI chunk thuộc document là đáp án đúng của val_split_20  -> để Recall đo được
  (b) union top-N chunk mà retriever HIỆN TẠI (v1) xếp cho từng câu hỏi val -> đây chính là những
      "kẻ gây nhiễu" THẬT mà model phải vượt qua, không phải nhiễu ngẫu nhiên dễ dãi.
Tập con này thường ~50-80 nghìn chunk (1/5 corpus) -> nhanh hơn ~5-7 lần.

⚠️ CÁCH ĐỌC KẾT QUẢ - RẤT QUAN TRỌNG:
  - Recall trên tập con LUÔN CAO HƠN trên corpus đầy đủ (ít kẻ nhiễu hơn). TUYỆT ĐỐI KHÔNG so con số
    này với 0.9200/0.9319 của các lần chạy trước.
  - Chỉ so Δ giữa 2 model đo trên CÙNG tập con. Baseline lấy thẳng từ cache có sẵn (không encode lại)
    nên phép so là công bằng tuyệt đối: cùng chunk, cùng câu hỏi, chỉ khác model.
  - Đây là đo DENSE-ONLY (không BM25, không harrier) - cô lập đúng thứ đang thay đổi là embedding.
  - Kết quả là CẬN DƯỚI của cải thiện: tập ứng viên bị giới hạn trong top-N của model CŨ, nên nếu
    model mới tìm được tài liệu nằm ngoài đó thì phần lợi ích ấy KHÔNG được tính ở đây.

Cách dùng:
    python scripts/quick_eval_finetune.py \
        --new-model models/aiteamvn-ft-light-ckpt/checkpoint-1500 \
        --top-n 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new-model", required=True, help="Thư mục model vừa fine-tune (hoặc checkpoint-N)")
    p.add_argument("--baseline-cache", default="outputs/cache/embcompare_aiteamvn.npz")
    p.add_argument("--ranks", default="outputs/cache/ranks_v1.npz", help="Thứ hạng của v1 trên val")
    p.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    p.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    p.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    p.add_argument("--top-n", type=int, default=50, help="Lấy top-N của v1 mỗi câu hỏi làm kẻ nhiễu")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--device", default="auto")
    p.add_argument("--report", default="outputs/predictions/quick_eval_ft.json")
    args = p.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer
    from src.evaluation.full_pipeline_eval import load_chunks, load_questions, resolve_device
    from src.evaluation.evaluate import load_ground_truth

    gt = load_ground_truth(args.ground_truth)
    questions = load_questions(args.questions)
    chunks = load_chunks(args.chunks)
    doc_ids = [str(c["doc_id"]) for c in chunks]

    d = np.load(args.ranks, allow_pickle=False)
    rank_qids = [str(q) for q in d["question_ids"]]
    ranks = d["ranks"]
    if [str(q["question_id"]) for q in questions] != rank_qids:
        print(f"[lỗi] Thứ tự question_id trong {args.ranks} khác {args.questions}.", file=sys.stderr)
        sys.exit(1)

    # --- Dựng tập con ---
    gold_docs = set()
    for qid, item in gt.items():
        gold_docs.update(str(x) for x in (item if isinstance(item, list) else item.get("answer", [])))

    subset = set()
    for row in ranks[:, : args.top_n]:
        subset.update(int(x) for x in row)
    n_from_ranks = len(subset)
    for i, dcid in enumerate(doc_ids):
        if dcid in gold_docs:
            subset.add(i)
    subset = np.array(sorted(subset), dtype=np.int64)

    print(f"Tập con: {len(subset):,} chunk / {len(chunks):,} toàn corpus ({len(subset)/len(chunks)*100:.1f}%)")
    print(f"  - từ top-{args.top_n} của v1: {n_from_ranks:,} chunk")
    print(f"  - thêm vào vì thuộc document đáp án: {len(subset) - n_from_ranks:,} chunk")
    sub_doc_ids = [doc_ids[i] for i in subset]

    def recall_of(sims) -> tuple:
        """sims: (n_questions, n_subset). Gộp max theo doc rồi lấy top-k."""
        tot = 0.0
        for qi, q in enumerate(questions):
            qid = str(q["question_id"])
            rel = set(str(x) for x in (gt[qid] if isinstance(gt[qid], list) else gt[qid]["answer"]))
            if not rel:
                continue
            row = sims[qi]
            best = {}
            top_idx = np.argpartition(-row, min(400, len(row) - 1))[:400]
            for j in top_idx:
                dcid = sub_doc_ids[j]
                if dcid not in best or row[j] > best[dcid]:
                    best[dcid] = row[j]
            top = [dc for dc, _ in sorted(best.items(), key=lambda x: -x[1])[: args.top_k]]
            tot += len(rel & set(top)) / len(rel)
        return tot / len(questions)

    device = resolve_device(args.device)
    results = {}

    # --- BASELINE: lấy thẳng từ cache, KHÔNG encode lại ---
    print("\n[1/2] Baseline (AITeamVN gốc) — đọc embedding từ cache có sẵn, không encode lại...")
    cache = np.load(args.baseline_cache, allow_pickle=True)
    base_emb = cache["embeddings"][subset]
    base_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding", device=device)
    base_model.max_seq_length = args.max_seq_length
    if device == "cuda":
        base_model = base_model.half()
    qemb = base_model.encode([q["question"] for q in questions], batch_size=args.batch_size,
                             convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=True).astype(np.float32)
    results["baseline"] = recall_of(qemb @ base_emb.T.astype(np.float32))
    print(f"  Recall (dense-only, tập con) = {results['baseline']:.4f}")
    del base_model, base_emb, cache

    # --- MODEL MỚI: phải encode tập con ---
    print(f"\n[2/2] Model fine-tune: {args.new_model} — encode {len(subset):,} chunk...")
    new_model = SentenceTransformer(args.new_model, device=device)
    new_model.max_seq_length = args.max_seq_length
    if device == "cuda":
        new_model = new_model.half()
    texts = [chunks[i]["passage"] for i in subset]
    new_emb = new_model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    qemb2 = new_model.encode([q["question"] for q in questions], batch_size=args.batch_size,
                             convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=True).astype(np.float32)
    results["finetuned"] = recall_of(qemb2 @ new_emb.T)
    print(f"  Recall (dense-only, tập con) = {results['finetuned']:.4f}")

    delta = results["finetuned"] - results["baseline"]
    print("\n" + "=" * 66)
    print(f"{'Baseline (AITeamVN gốc)':38s} {results['baseline']:.4f}")
    print(f"{'Fine-tune ' + Path(args.new_model).name:38s} {results['finetuned']:.4f}   Δ {delta:+.4f}")
    print("=" * 66)
    print("⚠️ Con số tuyệt đối CAO HƠN thực tế (tập con ít kẻ nhiễu) — CHỈ đọc cột Δ.")
    if delta >= 0.010:
        print("\n-> CẢI THIỆN RÕ. ĐÁNG bỏ vài giờ encode toàn bộ corpus để đo thật.")
    elif delta > 0.002:
        print(f"\n-> Cải thiện NHỎ ({delta:+.4f}). Nhớ đây là CẬN DƯỚI, nhưng theo quy tắc 2026-08-21")
        print("   thì mức này chưa đủ tin cậy — cân nhắc train thêm rồi đo lại checkpoint sau.")
    else:
        print(f"\n-> KHÔNG cải thiện ({delta:+.4f}). KHÔNG nên encode toàn corpus. Dừng train, đổi hướng")
        print("   (vd MultipleNegativesRankingLoss thay TripletLoss, hoặc tăng learning rate).")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"model": args.new_model, "subset_size": int(len(subset)),
                   "corpus_size": len(chunks), "top_n": args.top_n, **results, "delta": delta},
                  f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu: {args.report}")


if __name__ == "__main__":
    main()
