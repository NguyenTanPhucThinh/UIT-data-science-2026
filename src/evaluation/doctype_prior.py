"""PRIOR THEO LOẠI VĂN BẢN — hướng mới, KHÁC PHẠM TRÙ với mọi thứ đã thử (2026-08-28).

===================== VÌ SAO HƯỚNG NÀY KHÁC HẲN 15 HƯỚNG TRƯỚC =====================
Mọi hướng đã thử (đổi model, thêm nhánh ensemble, RRF->CombSUM, sweep trọng số, sweep k1/b,
rerank) đều trả lời CÙNG MỘT câu hỏi: "văn bản này GIỐNG câu hỏi tới mức nào?" — tức đều là
tín hiệu NGỮ NGHĨA giữa query và passage. Chúng thất bại/không đáng kể vì tín hiệu ngữ nghĩa
đã bão hoà (trần pool 99%, Recall@5 0.9319 — xem diagnose_ceiling.py).

Hướng này trả lời câu hỏi KHÁC HẲN: "loại văn bản này có XU HƯỚNG được chọn làm đáp án không?"
— một prior TIÊN NGHIỆM về document, KHÔNG phụ thuộc câu hỏi, nên KHÔNG thể suy ra từ bất kỳ
embedding nào. Đo trên dữ liệu thật (2026-08-28):

    loại          % corpus   % ground truth   lift
    Luật             6.56          26.10      3.98x   <- được chọn NHIỀU GẤP 4 LẦN tỉ lệ tự nhiên
    Nghị định       55.09          52.49      0.95x
    Thông tư        11.78           8.81      0.75x
    Quyết định      14.46           7.77      0.54x
    Công văn         5.11           1.32      0.26x   <- gần như KHÔNG BAO GIỜ là đáp án
    Chỉ thị          0.55           0.15      0.27x

Prior này ỔN ĐỊNH: phân bố trên train (6.116 nhãn) và val (1.521 nhãn) gần trùng khít
(Luật 26.10% vs 27.48%; Công văn 1.32% vs 0.72%) => không phải nhiễu chia tập.

Ca sai THẬT minh hoạ (question_id=54496, "Thời hạn nộp hồ sơ tự quyết toán thuế TNCN"):
đáp án đúng là Luật Quản lý thuế 2019, nhưng pipeline trả về 2 Công văn đúng chủ đề ở hạng
1-2. Ngữ nghĩa không phân biệt được; prior loại văn bản thì phân biệt được ngay.

===================== CHỐNG RÒ RỈ DỮ LIỆU (BẮT BUỘC) =====================
Prior ước lượng CHỈ TỪ train_split_80, đo CHỈ TRÊN val_split_20. Nếu ước lượng trên chính
val rồi đo trên val thì chắc chắn ra số đẹp GIẢ. Script này KHÔNG cho phép làm thế.

===================== CÔNG THỨC =====================
    score'(d) = score(d) + beta * log( P(type(d) | là đáp án, TRAIN) / P(type(d) | corpus) )
beta=0 => y hệt pipeline hiện tại (đối chứng bắt buộc phải ra đúng 0.9319).

Kèm kiểm định GHÉP CẶP McNemar (b = số câu được cứu, c = số câu bị hỏng) — với n=1398 thì
sai số chuẩn của Recall là ±0.013, nên so 2 con số Recall rời rạc KHÔNG kết luận được gì ở
mức Δ nhỏ; đếm b/c thì kết luận được.

CÁCH DÙNG:  python -m src.evaluation.doctype_prior
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .evaluate import load_ground_truth
    from .full_pipeline_eval import load_chunks
    from .ensemble_fusion import load_ranks, load_dup_groups, fuse_one_question
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.full_pipeline_eval import load_chunks
    from src.evaluation.ensemble_fusion import load_ranks, load_dup_groups, fuse_one_question


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    total = sum(math.comb(n, i) for i in range(lo + 1))
    return min(1.0, 2.0 * total / (2.0 ** n))


def fit_prior(train_gt: dict, doc_type_of: dict, corpus_counts: Counter, smooth: float = 1.0):
    """log-lift của mỗi loại văn bản, ƯỚC LƯỢNG CHỈ TỪ TRAIN. Làm trơn Laplace để loại hiếm
    không cho ra log(0). Loại chưa từng gặp -> lift 0 (trung tính), KHÔNG phạt mù."""
    gold_counts = Counter()
    for _, item in train_gt.items():
        for d in item:
            gold_counts[doc_type_of.get(str(d))] += 1
    tot_gold = sum(gold_counts.values())
    tot_corpus = sum(corpus_counts.values())
    types = set(corpus_counts) | set(gold_counts)
    prior = {}
    for t in types:
        p_gold = (gold_counts[t] + smooth) / (tot_gold + smooth * len(types))
        p_corp = (corpus_counts[t] + smooth) / (tot_corpus + smooth * len(types))
        prior[t] = math.log(p_gold / p_corp)
    return prior, gold_counts


def aggregate_with_prior(rrf_score, doc_ids, doc_type_of, prior, beta, top_k, group_of=None):
    best = {}
    for chunk_idx, score in rrf_score.items():
        d = doc_ids[chunk_idx]
        if d not in best or score > best[d]:
            best[d] = score
    if beta != 0.0:
        best = {d: s + beta * prior.get(doc_type_of.get(d), 0.0) for d, s in best.items()}
    ranked = sorted(best.items(), key=lambda x: -x[1])
    if not group_of:
        return [d for d, _ in ranked[:top_k]]
    out, seen = [], set()
    for d, _ in ranked:
        g = group_of.get(d)
        if g is not None:
            if g in seen:
                continue
            seen.add(g)
        out.append(d)
        if len(out) >= top_k:
            break
    return out[:top_k]


def load_gt_multi(paths):
    """Gộp nhiều file ground-truth (dùng khi sinh submission: ước lượng prior trên TOÀN BỘ nhãn
    có được = train_split_80 + val_split_20, vì public test rời hẳn cả hai nên KHÔNG rò rỉ)."""
    merged = {}
    for p in paths:
        for qid, docs in load_ground_truth(p).items():
            merged[qid] = docs
    return merged


def cmd_submit(args):
    """Sinh submission.json cho PUBLIC TEST: tái lập ĐÚNG cấu hình đã đạt 0.9208 rồi cộng prior."""
    doc_type_of, corpus_counts = {}, Counter()
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            doc_type_of[str(o["doc_id"])] = o.get("doc_type")
            corpus_counts[o.get("doc_type")] += 1

    fit_paths = [p for p in args.fit_gt.split(",") if p.strip()]
    print(f"Ước lượng prior từ: {fit_paths}")
    prior, gold_counts = fit_prior(load_gt_multi(fit_paths), doc_type_of, corpus_counts)
    tot_g, tot_c = sum(gold_counts.values()), sum(corpus_counts.values())
    for t in sorted(prior, key=lambda x: -prior[x]):
        print(f"   {str(t):<14s}{100*corpus_counts[t]/tot_c:8.2f}% corpus "
              f"{100*gold_counts[t]/tot_g:8.2f}% GT   log-lift {prior[t]:+.3f}")

    chunks = load_chunks(args.chunks)
    doc_ids = [c["doc_id"] for c in chunks]
    group_of = load_dup_groups(args.dedup_groups)

    sources, ref_qids = {}, None
    for name, path in [("v1", args.rank_v1), ("harrier", args.rank_harrier), ("bm25", args.rank_bm25)]:
        qids, ranks = load_ranks(path)
        if ref_qids is None:
            ref_qids = qids
        elif qids != ref_qids:
            print(f"[lỗi] {path} có danh sách question_id KHÁC — sẽ sinh sai bài nộp.", file=sys.stderr)
            sys.exit(1)
        sources[name] = ranks
    print(f"  {len(ref_qids)} câu hỏi public test")

    weights = {"v1": 1.0, "harrier": 0.2, "bm25": 0.1}
    use_top = {n: 100 for n in sources}

    preds_base, preds_prior = {}, {}
    for qi, qid in enumerate(ref_qids):
        combined = fuse_one_question({n: sources[n][qi] for n in sources}, weights, use_top, 60)
        preds_base[qid] = aggregate_with_prior(combined, doc_ids, doc_type_of, prior, 0.0,
                                               args.top_k, group_of)
        preds_prior[qid] = aggregate_with_prior(combined, doc_ids, doc_type_of, prior, args.beta,
                                                args.top_k, group_of)

    # --- CHỐT CHẶN định dạng cho CẢ HAI bản -----------------------------------------
    for label, preds in [("baseline", preds_base), ("prior", preds_prior)]:
        bad = [q for q, d in preds.items() if len(d) != args.top_k or len(set(d)) != len(d)]
        if bad:
            print(f"[lỗi] bản {label}: {len(bad)} câu sai định dạng (không đúng {args.top_k} doc, "
                  f"hoặc trùng doc_id). Ví dụ: {bad[:5]}", file=sys.stderr)
            sys.exit(1)
    print(f"  ✓ Định dạng hợp lệ: cả 2 bản đều đúng {args.top_k} doc_id, không trùng lặp")

    # --- Đối chiếu THÔNG TIN với artifact cũ (KHÔNG chặn) ----------------------------
    # 2026-08-28: đã xác minh outputs/predictions/submission_ensemble.json (21-08) KHÔNG tái lập
    # được từ code+cache hiện tại ở BẤT KỲ tổ hợp (top-n, rrf_k, trọng số) nào đã thử — nó sinh ra
    # từ ranks cũ hơn. Vì vậy đây CHỈ là thông tin tham khảo, KHÔNG dùng làm chốt chặn.
    if args.baseline_submission and Path(args.baseline_submission).exists():
        with open(args.baseline_submission, encoding="utf-8") as f:
            old = json.load(f)
        old_ans = {q: v["answer"] if isinstance(v, dict) else v for q, v in old.items()}
        same = sum(1 for q in old_ans if old_ans[q] == preds_base.get(q))
        print(f"  [thông tin] so với artifact cũ {args.baseline_submission}: khớp {same}/{len(old_ans)} câu."
              f"\n              (đã biết artifact đó KHÔNG tái lập được — dùng bản baseline sinh MỚI"
              f" bên dưới làm mốc, đừng dùng con số 0.9208 cũ.)")

    changed = sum(1 for q in preds_prior if preds_prior[q] != preds_base[q])
    changed_set = sum(1 for q in preds_prior if set(preds_prior[q]) != set(preds_base[q]))
    print(f"  Prior (beta={args.beta:g}) đổi THỨ TỰ top-5 ở {changed}/{len(ref_qids)} câu, "
          f"đổi TẬP tài liệu ở {changed_set}/{len(ref_qids)} câu")

    for path, preds in [(args.out_baseline, preds_base), (args.out, preds_prior)]:
        if not path:
            continue
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({q: {"answer": d} for q, d in preds.items()}, f, ensure_ascii=False, indent=2)
        print(f"Đã ghi: {path}")
    print("\nNộp CẢ HAI để đo Δ public sạch (baseline sinh mới vs prior), đừng so với 0.9208 cũ.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    sp = sub.add_parser("submit", help="sinh submission.json cho PUBLIC TEST (áp prior)")
    sp.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    sp.add_argument("--corpus", default="data/processed/corpus.jsonl")
    sp.add_argument("--fit-gt",
                    default="data/processed/train_split_80.json,data/processed/val_split_20.json",
                    help="Ước lượng prior trên TOÀN BỘ nhãn có được — public test rời hẳn nên "
                         "không rò rỉ, và nhiều nhãn hơn thì prior chính xác hơn")
    sp.add_argument("--dedup-groups", default=None)
    sp.add_argument("--rank-v1", default="outputs/cache/ranks_public_v1.npz")
    sp.add_argument("--rank-harrier", default="outputs/cache/ranks_public_harrier.npz")
    sp.add_argument("--rank-bm25", default="outputs/cache/ranks_public_bm25.npz")
    sp.add_argument("--baseline-submission", default="outputs/predictions/submission_ensemble.json",
                    help="Bản nộp cũ đã đạt 0.9208 — dùng làm chốt chặn tái lập ở beta=0")
    sp.add_argument("--beta", type=float, default=0.0005)
    sp.add_argument("--top-k", type=int, default=5)
    sp.add_argument("--out", default="outputs/predictions/submission_doctype_prior.json")
    sp.add_argument("--out-baseline", default="outputs/predictions/submission_baseline_regen.json",
                    help="Bản beta=0 sinh MỚI — nộp kèm để có mốc public sạch, vì artifact cũ "
                         "submission_ensemble.json đã xác minh là không tái lập được")
    sp.set_defaults(func=cmd_submit)

    ap.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--train-gt", default="data/processed/train_split_80.json",
                    help="CHỈ dùng để ước lượng prior — KHÔNG bao giờ để đo điểm")
    ap.add_argument("--ground-truth", default="data/processed/val_split_20.json",
                    help="CHỈ dùng để đo điểm — KHÔNG bao giờ để ước lượng prior")
    ap.add_argument("--dedup-groups", default=None)
    ap.add_argument("--rank-v1", default="outputs/cache/ranks_v1.npz")
    ap.add_argument("--rank-harrier", default="outputs/cache/ranks_harrier.npz")
    ap.add_argument("--rank-bm25", default="outputs/cache/ranks_bm25.npz")
    ap.add_argument("--betas", default="0,0.0005,0.001,0.002,0.003,0.005,0.008,0.012,0.02")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--report", default="outputs/predictions/doctype_prior_report.json")
    args = ap.parse_args()

    if getattr(args, "func", None):
        args.func(args)
        return

    if Path(args.train_gt).resolve() == Path(args.ground_truth).resolve():
        print("[lỗi] --train-gt và --ground-truth TRÙNG NHAU -> rò rỉ dữ liệu, kết quả vô nghĩa.",
              file=sys.stderr)
        sys.exit(1)

    doc_type_of, corpus_counts = {}, Counter()
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            t = o.get("doc_type")
            doc_type_of[str(o["doc_id"])] = t
            corpus_counts[t] += 1

    train_gt = load_ground_truth(args.train_gt)
    prior, gold_counts = fit_prior(train_gt, doc_type_of, corpus_counts)

    print("=" * 78)
    print("PRIOR ƯỚC LƯỢNG TỪ TRAIN_SPLIT_80 (val chưa hề được chạm vào)")
    print("=" * 78)
    tot_g, tot_c = sum(gold_counts.values()), sum(corpus_counts.values())
    print(f"   {'loại':<14s}{'% corpus':>10s}{'% GT train':>12s}{'log-lift':>10s}")
    for t in sorted(prior, key=lambda x: -prior[x]):
        print(f"   {str(t):<14s}{100*corpus_counts[t]/tot_c:10.2f}{100*gold_counts[t]/tot_g:12.2f}"
              f"{prior[t]:+10.3f}")

    chunks = load_chunks(args.chunks)
    doc_ids = [c["doc_id"] for c in chunks]
    gt = load_ground_truth(args.ground_truth)
    group_of = load_dup_groups(args.dedup_groups)

    sources, ref_qids = {}, None
    for name, path in [("v1", args.rank_v1), ("harrier", args.rank_harrier), ("bm25", args.rank_bm25)]:
        qids, ranks = load_ranks(path)
        ref_qids = ref_qids or qids
        sources[name] = ranks

    weights = {"v1": 1.0, "harrier": 0.2, "bm25": 0.1}
    use_top = {n: 100 for n in sources}

    # fuse 1 lần, tái dùng cho mọi beta (fuse không phụ thuộc beta) - tiết kiệm nhiều thời gian
    fused = {}
    for qi, qid in enumerate(ref_qids):
        if qid not in gt:
            continue
        fused[qid] = fuse_one_question({n: sources[n][qi] for n in sources}, weights, use_top, 60)

    print()
    print("=" * 78)
    print("ĐO TRÊN VAL_SPLIT_20 (prior đã cố định, không nhìn val)")
    print("=" * 78)
    print(f"   {'beta':>8s} {'Recall@5':>10s} {'Δ':>9s} {'b(cứu)':>8s} {'c(hỏng)':>8s} {'p':>9s}")
    baseline_correct, rows, base_recall = None, [], None
    for beta in [float(x) for x in args.betas.split(",") if x.strip()]:
        correct, rec_sum = {}, 0.0
        for qid, combined in fused.items():
            gold = set(gt[qid])
            top5 = aggregate_with_prior(combined, doc_ids, doc_type_of, prior, beta, args.top_k, group_of)
            correct[qid] = bool(gold & set(top5))
            rec_sum += len(gold & set(top5)) / len(gold)
        recall = rec_sum / len(fused)
        if baseline_correct is None:
            baseline_correct, base_recall = dict(correct), recall
        b = sum(1 for q in correct if correct[q] and not baseline_correct[q])
        c = sum(1 for q in correct if not correct[q] and baseline_correct[q])
        p = mcnemar_p(b, c)
        rows.append({"beta": beta, "recall": recall, "delta": recall - base_recall,
                     "b_saved": b, "c_broken": c, "mcnemar_p": p})
        print(f"   {beta:8.4f} {recall:10.4f} {recall-base_recall:+9.4f} {b:8d} {c:8d} {p:9.4f}")

    best = max(rows, key=lambda r: r["recall"])
    print()
    print("=" * 78)
    print("KẾT LUẬN")
    print("=" * 78)
    print(f"   Tốt nhất: beta={best['beta']:g} -> Recall {best['recall']:.4f} (Δ{best['delta']:+.4f}), "
          f"cứu {best['b_saved']} câu / hỏng {best['c_broken']} câu, p={best['mcnemar_p']:.4f}")
    if best["delta"] <= 0:
        print("   => Prior KHÔNG giúp gì. Đóng hướng này.")
    elif best["mcnemar_p"] < 0.05 and best["b_saved"] > best["c_broken"]:
        print("   => Cải thiện CÓ Ý NGHĨA THỐNG KÊ theo kiểm định ghép cặp (không chỉ là Δ đẹp).")
        print("      Đây là loại thay đổi ảnh hưởng ĐỀU lên mọi câu hỏi (prior toàn cục), khác hẳn")
        print("      entity-boost (chỉ chạm 2/1398 câu) đã đổi dấu khi lên public.")
        print("      => ĐỦ ĐIỀU KIỆN cân nhắc nộp public, nhưng đọc kỹ đường cong beta trước:")
        print("         nếu đỉnh nhọn/lệch biên lưới thì quét thêm; đỉnh nên nằm trên CAO NGUYÊN.")
    else:
        print("   => Δ dương nhưng CHƯA đủ bằng chứng ghép cặp (b không vượt trội c, hoặc p>=0.05).")
        print("      KHÔNG kết luận 'cải thiện', KHÔNG đốt lượt nộp.")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"prior": {str(k): v for k, v in prior.items()}, "sweep": rows}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nĐã lưu: {args.report}")


if __name__ == "__main__":
    main()
