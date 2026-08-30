"""FINE-TUNE PHORANKER VỚI SEMI-HARD NEGATIVE — theo đúng công thức arXiv 2507.14619 (2026-08-28).

===================== VÌ SAO "SEMI-HARD" MỚI LÀ MẤU CHỐT =====================
Bài arXiv 2507.14619 (top-3 SoICT Hackathon 2024, truy hồi văn bản luật tiếng Việt) đo ba cách
đào negative trên CÙNG kiến trúc bi-encoder → PhoRanker:

    easy negative (lấy ngẫu nhiên từ TOÀN corpus)        : MRR@10 56.75
    hard negative (lấy top ứng viên của retriever)       : MRR@10 47.96   <- TỆ NHẤT
    SEMI-HARD (ngẫu nhiên TRONG ứng viên, trừ đáp án)    : MRR@10 78.21   <- TỐT NHẤT

=> Chênh lệch KHỔNG LỒ (47.96 -> 78.21) chỉ do đổi cách chọn negative. Và đây ĐÚNG cơ chế đã giết
4 lần fine-tune embedding của team: cả 4 lần đều dùng HARD negative (top-ranked của v1), lần 4 còn
lọc thêm theo cosine mà vẫn -0.1189. Log dự án ghi giả thuyết "false-negative contamination" đã bị
bác bỏ — bài báo này đưa ra lời giải KHÁC: hard negative gây BẤT ỔN HUẤN LUYỆN, không phải nhiễm bẩn.
Team CHƯA TỪNG thử semi-hard.

ĐỊNH NGHĨA CHÍNH XÁC (để không lặp lại nhầm lẫn):
    - Lấy top-K document ứng viên của ensemble cho câu hỏi (K = --candidate-pool).
    - LOẠI BỎ mọi document là đáp án đúng.
    - Từ phần CÒN LẠI, lấy NGẪU NHIÊN n cái (n = --n-negatives). KHÔNG lấy n cái đầu bảng.
  Chính chữ "ngẫu nhiên trong ứng viên" là điểm khác biệt với hard negative.

===================== CHỐNG RÒ RỈ =====================
CHỈ dùng câu hỏi của train_split_80 (5602 câu, ranks có sẵn trong ranks_hold_*.npz — đã xác minh
giao với val_split_20 = 0 câu). Script có chốt chặn CỨNG, từ chối chạy nếu phát hiện câu val.

CÁCH DÙNG:
    python -m src.training.finetune_phoranker selftest            # không GPU
    python -m src.training.finetune_phoranker preview --out ...   # xem cặp train, KHÔNG train
    python -m src.training.finetune_phoranker train --output-dir ...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

try:
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.full_pipeline_eval import load_chunks, load_questions, resolve_device
    from src.evaluation.ensemble_fusion import load_ranks, fuse_one_question
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.full_pipeline_eval import load_chunks, load_questions, resolve_device
    from src.evaluation.ensemble_fusion import load_ranks, fuse_one_question

PROD_WEIGHTS = {"v1": 1.0, "harrier": 0.2, "bm25": 0.1}
PROD_RRF_K = 60


def mine_negatives(cand_docs, gold, n, strategy, rng):
    """cand_docs: doc_id đã xếp hạng giảm dần. Trả về n negative theo chiến lược.

    'semi-hard' (mặc định, theo bài báo): NGẪU NHIÊN trong ứng viên sau khi loại đáp án đúng.
    'hard'     : n ứng viên ĐẦU BẢNG sau khi loại đáp án — đúng cách đã thất bại 4/4 ở team.
    'easy'     : không dùng ứng viên, nơi gọi tự bốc ngẫu nhiên toàn corpus.
    """
    pool = [d for d in cand_docs if d not in gold]
    if strategy == "hard":
        return pool[:n]
    if strategy == "semi-hard":
        if len(pool) <= n:
            return pool
        return rng.sample(pool, n)
    raise ValueError(f"strategy không hợp lệ: {strategy!r}")


def build_pairs(args, rng):
    chunks = load_chunks(args.chunks)
    doc_ids = [c["doc_id"] for c in chunks]
    field_p = "passage_segmented" if args.use_segmented_text else "passage"
    if field_p not in chunks[0]:
        print(f"[cảnh báo] chunk thiếu {field_p} -> dùng 'passage'.", file=sys.stderr)
        field_p = "passage"

    questions = load_questions(args.questions)
    qmap = {str(q["question_id"]): q for q in questions}

    # ---- CHỐT CHẶN RÒ RỈ ----
    val_gt = load_ground_truth(args.val_ground_truth)
    leaked = [q for q in qmap if q in val_gt]
    if leaked:
        print(f"[lỗi] {len(leaked)} câu hỏi TRAIN nằm trong val_split_20 -> rò rỉ, "
              f"kết quả đo sẽ vô nghĩa. Ví dụ: {leaked[:5]}", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ Chốt chặn rò rỉ: 0/{len(qmap)} câu train nằm trong val_split_20")

    sources, ref_qids = {}, None
    for name, path in [("v1", args.rank_v1), ("harrier", args.rank_harrier), ("bm25", args.rank_bm25)]:
        qids, ranks = load_ranks(path)
        if ref_qids is None:
            ref_qids = qids
        elif qids != ref_qids:
            print(f"[lỗi] {path} có question_id khác các file trước.", file=sys.stderr)
            sys.exit(1)
        sources[name] = ranks

    use_top = {n: 100 for n in sources}
    pairs, stats = [], Counter()
    for qi, qid in enumerate(ref_qids):
        q = qmap.get(qid)
        if q is None:
            stats["thiếu_text"] += 1
            continue
        gold = set(str(x) for x in (q.get("answer_doc_ids") or []))
        if not gold:
            stats["không_có_đáp_án"] += 1
            continue
        qt = q.get("question_segmented" if args.use_segmented_text else "question") or q.get("question", "")

        fused = fuse_one_question({n: sources[n][qi] for n in sources}, PROD_WEIGHTS, use_top, PROD_RRF_K)
        best = {}
        for ci, s in fused.items():
            d = doc_ids[ci]
            if d not in best or s > best[d][1]:
                best[d] = (ci, s)
        ranked = [d for d, _ in sorted(best.items(), key=lambda x: -x[1][1])][: args.candidate_pool]

        # POSITIVE: chunk đại diện của tài liệu đúng. Chỉ lấy nếu tài liệu đúng CÓ trong pool —
        # nếu không có thì ta không biết chunk nào liên quan, thêm vào sẽ là nhãn nhiễu.
        n_pos = 0
        for g in gold:
            if g in best:
                pairs.append({"q": qt, "p": chunks[best[g][0]][field_p], "label": 1.0, "qid": qid})
                n_pos += 1
        if n_pos == 0:
            stats["đáp_án_ngoài_pool"] += 1
            continue
        stats["câu_dùng_được"] += 1

        for d in mine_negatives(ranked, gold, args.n_negatives, args.negative_strategy, rng):
            pairs.append({"q": qt, "p": chunks[best[d][0]][field_p], "label": 0.0, "qid": qid})

    print(f"\n  Thống kê: {dict(stats)}")
    n_pos = sum(1 for p in pairs if p["label"] == 1.0)
    print(f"  {len(pairs):,} cặp ({n_pos:,} positive / {len(pairs)-n_pos:,} negative), "
          f"chiến lược negative = {args.negative_strategy!r}, n={args.n_negatives}")
    return pairs


def cmd_preview(args) -> None:
    rng = random.Random(args.seed)
    pairs = build_pairs(args, rng)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs[: args.limit]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nĐã ghi {min(len(pairs), args.limit)} cặp đầu vào {args.out} để KIỂM TRA BẰNG MẮT.")
    print("Hãy xem vài dòng: positive phải THỰC SỰ liên quan câu hỏi, negative phải là văn bản")
    print("KHÁC nhưng cùng chủ đề (nếu negative toàn thứ chẳng liên quan gì -> pool sai).")


def cmd_train(args) -> None:
    try:
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader
    except ImportError:
        print("[lỗi] Thiếu sentence-transformers/torch.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    pairs = build_pairs(args, rng)
    if not pairs:
        print("[lỗi] Không sinh được cặp huấn luyện nào.", file=sys.stderr)
        sys.exit(1)

    examples = [InputExample(texts=[p["q"], p["p"]], label=p["label"]) for p in pairs]
    rng.shuffle(examples)
    device = resolve_device(args.device)
    print(f"\nLoad model gốc: {args.base_model} (max_length={args.max_length}, device={device})")
    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length, device=device)

    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    warmup = int(len(loader) * args.epochs * 0.1)
    print(f"Huấn luyện {args.epochs} epoch, batch={args.batch_size}, "
          f"{len(loader)} bước/epoch, warmup={warmup}")
    model.fit(train_dataloader=loader, epochs=args.epochs, warmup_steps=warmup,
              optimizer_params={"lr": args.lr}, output_path=args.output_dir,
              use_amp=args.fp16, show_progress_bar=True)
    model.save(args.output_dir)
    print(f"\nĐã lưu model: {args.output_dir}")
    print("Bước tiếp: chấm lại val bằng rerank_blend score --reranker-model " + args.output_dir)


def cmd_selftest(args) -> None:
    ok = True

    def check(cond, label):
        nonlocal ok
        print(f"  {'OK ' if cond else 'SAI'} {label}")
        if not cond:
            ok = False

    print("SELFTEST đào negative (không cần GPU/dữ liệu)")
    cands = [f"D{i}" for i in range(20)]
    gold = {"D3", "D7"}

    hard = mine_negatives(cands, gold, 5, "hard", random.Random(0))
    check(hard == ["D0", "D1", "D2", "D4", "D5"], "hard = n ứng viên ĐẦU BẢNG, đã loại đáp án")
    check(not (set(hard) & gold), "hard KHÔNG chứa đáp án đúng")

    rng = random.Random(42)
    semi = mine_negatives(cands, gold, 5, "semi-hard", rng)
    check(len(semi) == 5 and len(set(semi)) == 5, "semi-hard trả đúng n, không trùng")
    check(not (set(semi) & gold), "semi-hard KHÔNG chứa đáp án đúng")
    check(semi != ["D0", "D1", "D2", "D4", "D5"], "semi-hard KHÁC hard (không phải top đầu bảng)")

    # phải THỰC SỰ ngẫu nhiên: nhiều seed khác nhau phải cho nhiều tập khác nhau
    seen = {tuple(sorted(mine_negatives(cands, gold, 5, "semi-hard", random.Random(s)))) for s in range(30)}
    check(len(seen) > 10, f"semi-hard phân tán trên nhiều seed ({len(seen)} tập khác nhau/30)")

    # phủ khắp pool chứ không dồn về đầu bảng
    pos = Counter()
    for s in range(300):
        for d in mine_negatives(cands, gold, 5, "semi-hard", random.Random(s)):
            pos[int(d[1:])] += 1
    late = sum(v for k, v in pos.items() if k >= 10)
    check(late > 300, f"semi-hard lấy CẢ ứng viên cuối bảng ({late} lần từ hạng >=10)")

    small = mine_negatives(["D0", "D1"], {"D0"}, 5, "semi-hard", random.Random(0))
    check(small == ["D1"], "pool nhỏ hơn n -> trả hết phần còn lại, không lỗi")
    check(mine_negatives(["D0"], {"D0"}, 5, "semi-hard", random.Random(0)) == [],
          "pool rỗng sau khi loại đáp án -> trả rỗng, không lỗi")

    try:
        mine_negatives(cands, gold, 5, "linh-tinh", random.Random(0))
        check(False, "chiến lược sai phải báo lỗi")
    except ValueError:
        check(True, "chiến lược sai báo lỗi rõ ràng")

    print("\n" + ("TẤT CẢ ĐỀU ĐÚNG" if ok else "CÓ LỖI"))
    sys.exit(0 if ok else 1)


def add_common(p):
    p.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    p.add_argument("--questions", default="data/processed/train_split_80_questions.jsonl")
    p.add_argument("--val-ground-truth", default="data/processed/val_split_20.json",
                   help="CHỈ để chốt chặn rò rỉ — không bao giờ dùng để huấn luyện")
    p.add_argument("--rank-v1", default="outputs/cache/ranks_hold_v1.npz")
    p.add_argument("--rank-harrier", default="outputs/cache/ranks_hold_harrier.npz")
    p.add_argument("--rank-bm25", default="outputs/cache/ranks_hold_bm25.npz")
    p.add_argument("--candidate-pool", type=int, default=50,
                   help="Lấy top-K document làm pool đào negative (bài báo dùng top-90 chunk)")
    p.add_argument("--n-negatives", type=int, default=5, help="Bài báo dùng n=5 (tốt nhất)")
    p.add_argument("--negative-strategy", choices=["semi-hard", "hard"], default="semi-hard")
    p.add_argument("--use-segmented-text", action="store_true", default=True)
    p.add_argument("--no-segmented-text", dest="use_segmented_text", action="store_false")
    p.add_argument("--seed", type=int, default=42)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("preview", help="sinh cặp train và GHI RA để kiểm tra mắt, KHÔNG huấn luyện")
    add_common(pp)
    pp.add_argument("--limit", type=int, default=60)
    pp.add_argument("--out", default="outputs/predictions/phoranker_train_preview.jsonl")
    pp.set_defaults(func=cmd_preview)

    pt = sub.add_parser("train", help="fine-tune cross-encoder — CẦN GPU")
    add_common(pt)
    pt.add_argument("--base-model", default="itdainb/PhoRanker")
    pt.add_argument("--output-dir", default="models/phoranker_semihard")
    pt.add_argument("--epochs", type=int, default=2)
    pt.add_argument("--batch-size", type=int, default=16)
    pt.add_argument("--lr", type=float, default=2e-5)
    pt.add_argument("--max-length", type=int, default=256)
    pt.add_argument("--device", default="auto")
    pt.add_argument("--fp16", action="store_true", default=True)
    pt.add_argument("--no-fp16", dest="fp16", action="store_false")
    pt.set_defaults(func=cmd_train)

    ps = sub.add_parser("selftest")
    ps.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
