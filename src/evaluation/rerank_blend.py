"""RERANK CROSS-ENCODER TRỘN ĐIỂM Ở CẤP DOCUMENT — hướng cuối cùng còn lại (2026-08-28).

===================== VÌ SAO THỬ LẠI RERANKER DÙ ĐÃ 4/4 ÂM =====================
Đối chiếu 4 phép thử reranker cũ với công thức SOTA (arXiv 2507.14619, top-3 SoICT Hackathon 2024)
cho thấy chúng KHÁC HẲN nhau ở 3 điểm, nên "4/4 âm" KHÔNG bác bỏ được công thức SOTA:

  | yếu tố       | SOTA                                   | 4 phép thử cũ của team              |
  |--------------|----------------------------------------|-------------------------------------|
  | model        | PhoRanker (chuyên tiếng Việt)          | bge-reranker-v2-m3 (đa ngữ), AITeamVN|
  | huấn luyện   | FINE-TUNE với SEMI-HARD negative       | dùng thẳng bản pretrained            |
  | ghép vào     | rerank top-N ứng viên của retriever    | thay thứ hạng RRF / làm nguồn RRF phụ|

Và chẩn đoán (diagnose_ceiling.py) chỉ đúng vào đây: 85.7% lỗi còn lại là lỗi XẾP HẠNG với đáp án
ĐÃ nằm trong pool (Recall@100 = 0.9890 vs Recall@5 = 0.9319) — +0.057 dư địa mà CHỈ rerank chạm tới.

===================== THIẾT KẾ, RÚT TỪ ĐÚNG CÁC THẤT BẠI CŨ =====================
1. TRỘN ĐIỂM, KHÔNG THAY THẾ. Log dự án đã chẩn đoán: `rerank_candidate_details()` cũ xếp lại THUẦN
   theo điểm reranker, VỨT điểm ensemble — tức reranker chạy ở "trọng số vô hạn", đúng kiểu thất bại
   đã thấy ở harrier w=1.5. Ở đây: score = norm(ensemble) + alpha*norm(reranker), alpha dò được.
2. alpha=0 PHẢI TÁI LẬP BASELINE CHÍNH XÁC. Chuẩn hoá min-max per-query là phép biến đổi ĐƠN ĐIỆU
   nên KHÔNG đổi thứ hạng — alpha=0 cho ra đúng top-5 của baseline. `selftest` kiểm tra điều này.
3. CẤP DOCUMENT, không phải cấp chunk. Lỗi cần sửa là "tài liệu đúng ở hạng 6-10" (đo ở cấp document),
   nên rerank đúng top-N DOCUMENT của ensemble, mỗi document lấy chunk đại diện (chunk thắng phép
   max lúc gộp). Rẻ hơn nhiều so với rerank toàn bộ chunk pool, và đúng hạt hơn.
4. KIỂM ĐỊNH GHÉP CẶP (McNemar) chứ không chỉ Δ Recall — sàn nhiễu local là ±0.0132 (n=1398).

⚠️ RỦI RO ĐÃ BIẾT TRƯỚC: PhoRanker có max_length=256 (họ PhoBERT/RoBERTa, trần kiến trúc CỨNG —
   đúng lý do đã đóng hẳn DEk21). Chunk của ta tới 2000 ký tự nên sẽ bị CẮT đáng kể. Đây là lý do
   phải chạy `score` với model PRETRAINED TRƯỚC (rẻ) rồi mới quyết định có fine-tune hay không.

CÁCH DÙNG (từ thư mục GỐC repo):
    python -m src.evaluation.rerank_blend selftest                    # không cần GPU
    python -m src.evaluation.rerank_blend score --out ...             # GPU, chấm cặp
    python -m src.evaluation.rerank_blend blend --rerank ...          # CPU, sweep alpha + McNemar
    python -m src.evaluation.rerank_blend submit --rerank ... --alpha ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .evaluate import load_ground_truth
    from .full_pipeline_eval import load_chunks, load_questions, resolve_device
    from .ensemble_fusion import load_ranks, load_dup_groups, fuse_one_question
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.full_pipeline_eval import load_chunks, load_questions, resolve_device
    from src.evaluation.ensemble_fusion import load_ranks, load_dup_groups, fuse_one_question


PROD_WEIGHTS = {"v1": 1.0, "harrier": 0.2, "bm25": 0.1}
PROD_RRF_K = 60


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2.0 ** n))


def minmax(vals: Sequence[float]):
    """Chuẩn hoá min-max per-query. ĐƠN ĐIỆU TĂNG -> KHÔNG đổi thứ hạng khi dùng một mình.
    Đây chính là thứ bảo đảm alpha=0 tái lập baseline chính xác."""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.0] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def docs_with_best_chunk(rrf_score: Dict[int, float], doc_ids: Sequence[str],
                         top_n_docs: int) -> List[Tuple[str, int, float]]:
    """Gộp chunk -> document theo max (GIỐNG HỆT aggregate_max của pipeline chính), nhưng GIỮ LẠI
    chỉ số chunk đại diện — chunk đã THẮNG phép max — để đưa đúng đoạn văn đó cho cross-encoder."""
    best: Dict[str, Tuple[int, float]] = {}
    for chunk_idx, score in rrf_score.items():
        d = doc_ids[chunk_idx]
        cur = best.get(d)
        if cur is None or score > cur[1]:
            best[d] = (chunk_idx, score)
    ranked = sorted(((d, ci, s) for d, (ci, s) in best.items()), key=lambda x: -x[2])
    return ranked[:top_n_docs]


def load_sources(rank_paths: Dict[str, str]):
    sources, ref_qids = {}, None
    for name, path in rank_paths.items():
        qids, ranks = load_ranks(path)
        if ref_qids is None:
            ref_qids = qids
        elif qids != ref_qids:
            print(f"[lỗi] {path} có danh sách question_id KHÁC các file trước — sẽ ghép nhầm câu hỏi.",
                  file=sys.stderr)
            sys.exit(1)
        sources[name] = ranks
    return sources, ref_qids


def build_candidates(sources, ref_qids, doc_ids, top_n_docs):
    """Danh sách ứng viên document cho MỌI câu hỏi, dùng ĐÚNG cấu hình production."""
    out = {}
    use_top = {n: 100 for n in sources}
    for qi, qid in enumerate(ref_qids):
        fused = fuse_one_question({n: sources[n][qi] for n in sources}, PROD_WEIGHTS, use_top, PROD_RRF_K)
        out[qid] = docs_with_best_chunk(fused, doc_ids, top_n_docs)
    return out


# ---------------------------------------------------------------------------
# score — chấm cặp (query, chunk đại diện) bằng cross-encoder
# ---------------------------------------------------------------------------

def cmd_score(args) -> None:
    import numpy as np

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("[lỗi] Thiếu sentence-transformers.", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks(args.chunks)
    doc_ids = [c["doc_id"] for c in chunks]
    questions = load_questions(args.questions)
    qtext = {}
    for q in questions:
        qid = str(q["question_id"])
        if args.use_segmented_text:
            t = q.get("question_segmented")
            if not t:
                print(f"[cảnh báo] câu {qid} thiếu question_segmented -> dùng bản thô.", file=sys.stderr)
                t = q.get("question", "")
        else:
            t = q.get("question", "")
        qtext[qid] = t

    field = "passage_segmented" if args.use_segmented_text else "passage"
    if args.use_segmented_text and field not in chunks[0]:
        print(f"[cảnh báo] chunk thiếu {field} -> dùng 'passage'.", file=sys.stderr)
        field = "passage"

    sources, ref_qids = load_sources({"v1": args.rank_v1, "harrier": args.rank_harrier,
                                      "bm25": args.rank_bm25})
    cands = build_candidates(sources, ref_qids, doc_ids, args.top_n_docs)

    n_pairs = sum(len(v) for v in cands.values())
    print(f"{len(ref_qids)} câu hỏi x tối đa {args.top_n_docs} document = {n_pairs:,} cặp cross-encoder.")
    print(f"Model: {args.reranker_model} (max_length={args.max_length}, segmented={args.use_segmented_text})")
    print("LƯU Ý: cross-encoder KHÔNG cache được — đây là bước đắt nhất pipeline.")

    device = resolve_device(args.device)
    model = CrossEncoder(args.reranker_model, max_length=args.max_length, device=device)

    # Số tham số — cần cho quy định mục 5 (tổng pipeline < 4 tỷ). Ensemble hiện tại ~1,16 tỷ
    # (v1 568M + harrier 596M), nên còn dư ~2,84 tỷ cho reranker.
    try:
        n_par = sum(p.numel() for p in model.model.parameters())
        print(f"  Số tham số reranker: {n_par/1e6:.1f}M — tổng pipeline ~{(1160+n_par/1e6)/1000:.2f} tỷ "
              f"({'DƯỚI' if 1160 + n_par/1e6 < 4000 else '⚠️ VƯỢT'} hạn mức 4 tỷ, quy định mục 5)")
        print("  ⚠️ Thêm model mới vào pipeline chính thức PHẢI gửi đề xuất BTC trước 10 ngày so với")
        print("     hạn chót private test (quy định mục 7) — xem dòng A2 phan_cong_thu_nghiem_task1.md")
    except Exception:
        pass

    all_docs, all_scores = [], []
    flat_pairs, flat_meta = [], []
    for qid in ref_qids:
        for d, ci, _ in cands[qid]:
            flat_pairs.append([qtext.get(qid, ""), chunks[ci][field]])
            flat_meta.append(qid)

    print(f"Đang chấm {len(flat_pairs):,} cặp...")
    scores = model.predict(flat_pairs, batch_size=args.batch_size, show_progress_bar=True)

    by_q: Dict[str, List[float]] = {}
    k = 0
    for qid in ref_qids:
        n = len(cands[qid])
        by_q[qid] = [float(x) for x in scores[k:k + n]]
        k += n
    assert k == len(flat_pairs), "lệch số cặp — không được xảy ra"

    max_n = max(len(cands[q]) for q in ref_qids)
    doc_arr = np.full((len(ref_qids), max_n), "", dtype=object)
    sc_arr = np.full((len(ref_qids), max_n), np.nan, dtype=np.float32)
    ens_arr = np.full((len(ref_qids), max_n), np.nan, dtype=np.float32)
    for i, qid in enumerate(ref_qids):
        for j, (d, ci, es) in enumerate(cands[qid]):
            doc_arr[i, j] = d
            sc_arr[i, j] = by_q[qid][j]
            ens_arr[i, j] = es

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, question_ids=np.array(ref_qids),
                        docs=doc_arr.astype(str), rerank=sc_arr, ensemble=ens_arr)
    print(f"\nĐã lưu điểm rerank: {args.out}  (shape={sc_arr.shape})")


# ---------------------------------------------------------------------------
# blend / submit
# ---------------------------------------------------------------------------

def load_rerank(path):
    import numpy as np
    d = np.load(path, allow_pickle=False)
    return [str(q) for q in d["question_ids"]], d["docs"], d["rerank"], d["ensemble"]


def blended_top5(docs_row, rr_row, ens_row, alpha, top_k, group_of=None):
    import numpy as np
    keep = [j for j in range(len(docs_row)) if str(docs_row[j]) != "" and not np.isnan(ens_row[j])]
    if not keep:
        return []
    ens_n = minmax([float(ens_row[j]) for j in keep])
    rr_n = minmax([float(rr_row[j]) for j in keep])
    scored = [(str(docs_row[j]), ens_n[i] + alpha * rr_n[i]) for i, j in enumerate(keep)]
    scored.sort(key=lambda x: -x[1])
    if not group_of:
        return [d for d, _ in scored[:top_k]]
    out, seen = [], set()
    for d, _ in scored:
        g = group_of.get(d)
        if g is not None:
            if g in seen:
                continue
            seen.add(g)
        out.append(d)
        if len(out) >= top_k:
            break
    return out[:top_k]


def cmd_blend(args) -> None:
    qids, docs, rr, ens = load_rerank(args.rerank)
    gt = load_ground_truth(args.ground_truth)
    group_of = load_dup_groups(args.dedup_groups)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    print("=" * 78)
    print("TRỘN ĐIỂM: score = norm(ensemble) + alpha * norm(reranker), gộp ở cấp DOCUMENT")
    print("=" * 78)
    print("  alpha=0 PHẢI ra đúng Recall baseline (min-max là phép đơn điệu, không đổi thứ hạng).")
    print(f"  {'alpha':>8s} {'Recall@5':>10s} {'Δ':>9s} {'b(cứu)':>8s} {'c(hỏng)':>8s} {'p':>9s}")

    base_correct, base_recall, rows = None, None, []
    for alpha in alphas:
        correct, tot, n = {}, 0.0, 0
        for i, qid in enumerate(qids):
            if qid not in gt:
                continue
            gold = set(gt[qid])
            top5 = blended_top5(docs[i], rr[i], ens[i], alpha, args.top_k, group_of)
            hit = len(gold & set(top5))
            correct[qid] = hit > 0
            tot += hit / len(gold)
            n += 1
        recall = tot / n
        if base_correct is None:
            base_correct, base_recall = dict(correct), recall
        b = sum(1 for q in correct if correct[q] and not base_correct[q])
        c = sum(1 for q in correct if not correct[q] and base_correct[q])
        p = mcnemar_p(b, c)
        rows.append({"alpha": alpha, "recall": recall, "delta": recall - base_recall,
                     "b_saved": b, "c_broken": c, "mcnemar_p": p})
        print(f"  {alpha:8.3f} {recall:10.4f} {recall-base_recall:+9.4f} {b:8d} {c:8d} {p:9.4f}")

    best = max(rows, key=lambda r: r["recall"])
    print()
    print(f"  Tốt nhất: alpha={best['alpha']:g} -> {best['recall']:.4f} ({best['delta']:+.4f}), "
          f"cứu {best['b_saved']} / hỏng {best['c_broken']}, p={best['mcnemar_p']:.4f}")
    print()
    print("  ⚠️ NHẮC LẠI BÀI HỌC 28-08: Δ local KHÔNG dự đoán được dấu của Δ public ở vùng <0.01")
    print("     (3/3 lần đổi dấu, kể cả khi McNemar có ý nghĩa). Chỉ nộp nếu Δ ĐỦ LỚN (>=0.02),")
    print("     và ĐỪNG chọn alpha bằng argmax của sweep này — chọn thận trọng hơn đỉnh (bài học #21).")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu: {args.report}")


def cmd_submit(args) -> None:
    qids, docs, rr, ens = load_rerank(args.rerank)
    group_of = load_dup_groups(args.dedup_groups)
    preds, base = {}, {}
    for i, qid in enumerate(qids):
        preds[qid] = blended_top5(docs[i], rr[i], ens[i], args.alpha, args.top_k, group_of)
        base[qid] = blended_top5(docs[i], rr[i], ens[i], 0.0, args.top_k, group_of)
    bad = [q for q, d in preds.items() if len(d) != args.top_k or len(set(d)) != len(d)]
    if bad:
        print(f"[lỗi] {len(bad)} câu sai định dạng. Ví dụ {bad[:5]}", file=sys.stderr)
        sys.exit(1)
    changed = sum(1 for q in preds if preds[q] != base[q])
    print(f"  alpha={args.alpha:g} đổi top-5 ở {changed}/{len(qids)} câu ({100*changed/len(qids):.1f}%)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({q: {"answer": d} for q, d in preds.items()}, f, ensure_ascii=False, indent=2)
    print(f"Đã ghi bài nộp: {args.out}")


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def cmd_selftest(args) -> None:
    import numpy as np
    ok = True

    def check(cond, label):
        nonlocal ok
        print(f"  {'OK ' if cond else 'SAI'} {label}")
        if not cond:
            ok = False

    print("SELFTEST rerank_blend (không cần GPU/dữ liệu thật)")

    check(minmax([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0], "minmax khi mọi điểm bằng nhau -> toàn 0")
    m = minmax([1.0, 3.0, 2.0])
    check(abs(m[0]) < 1e-9 and abs(m[1] - 1.0) < 1e-9 and abs(m[2] - 0.5) < 1e-9, "minmax đúng công thức")
    check(sorted(range(3), key=lambda i: -m[i]) == [1, 2, 0], "minmax GIỮ NGUYÊN thứ hạng (đơn điệu)")

    # gộp chunk->document giữ đúng chunk thắng max
    rrf = {0: 0.5, 1: 0.9, 2: 0.3, 3: 0.7}
    doc_ids = ["A", "A", "B", "B"]
    got = docs_with_best_chunk(rrf, doc_ids, 5)
    check(got == [("B", 3, 0.7), ("A", 1, 0.9)] or got == [("A", 1, 0.9), ("B", 3, 0.7)],
          "gộp document lấy đúng chunk điểm cao nhất")
    check(got[0] == ("A", 1, 0.9), "xếp document giảm dần theo điểm")

    # alpha=0 phải tái lập ĐÚNG thứ hạng theo ensemble, bất kể điểm reranker thế nào
    docs_row = np.array(["D1", "D2", "D3", "D4", "D5", "D6"])
    ens_row = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
    rr_row = np.array([-9.0, 9.0, -9.0, 9.0, -9.0, 9.0], dtype=np.float32)  # cố tình NGƯỢC hẳn
    base = blended_top5(docs_row, rr_row, ens_row, 0.0, 5)
    check(base == ["D1", "D2", "D3", "D4", "D5"], "alpha=0 tái lập CHÍNH XÁC thứ hạng ensemble")
    hi = blended_top5(docs_row, rr_row, ens_row, 100.0, 5)
    check(hi[0] == "D2", "alpha rất lớn -> reranker chiếm ưu thế (cơ chế có tác dụng)")
    # alpha vừa: tính tay. norm(ens)=[1,.8,.6,.4,.2,0] ; norm(rr)=[0,1,0,1,0,1]
    # alpha=0.25 -> [1.0, 1.05, 0.6, 0.65, 0.2, 0.25] -> D2,D1,D4,D3,D6
    mid = blended_top5(docs_row, rr_row, ens_row, 0.25, 5)
    check(mid == ["D2", "D1", "D4", "D3", "D6"], "alpha vừa -> trộn đúng công thức (đối chiếu tay)")

    # NaN / ô trống bị loại đúng
    docs_row2 = np.array(["D1", "D2", ""])
    ens_row2 = np.array([0.9, 0.8, np.nan], dtype=np.float32)
    rr_row2 = np.array([1.0, 2.0, np.nan], dtype=np.float32)
    check(blended_top5(docs_row2, rr_row2, ens_row2, 0.0, 5) == ["D1", "D2"], "bỏ đúng ô trống/NaN")

    # dedup group
    g = {"D1": 0, "D2": 0}
    check(blended_top5(docs_row, rr_row, ens_row, 0.0, 3, g) == ["D1", "D3", "D4"],
          "dedup bỏ document cùng nhóm, lấp slot bằng ứng viên sau")

    # McNemar
    from math import isclose
    check(isclose(mcnemar_p(10, 0), 0.001953125, rel_tol=1e-9), "McNemar b=10,c=0")
    check(mcnemar_p(0, 0) == 1.0 and mcnemar_p(5, 5) == 1.0, "McNemar trường hợp biên")

    print("\n" + ("TẤT CẢ ĐỀU ĐÚNG" if ok else "CÓ LỖI — DỪNG, đừng chạy trên dữ liệu thật"))
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("score", help="chấm (query, chunk đại diện) bằng cross-encoder — CẦN GPU")
    ps.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    ps.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    ps.add_argument("--rank-v1", default="outputs/cache/ranks_v1.npz")
    ps.add_argument("--rank-harrier", default="outputs/cache/ranks_harrier.npz")
    ps.add_argument("--rank-bm25", default="outputs/cache/ranks_bm25.npz")
    ps.add_argument("--reranker-model", default="itdainb/PhoRanker")
    ps.add_argument("--max-length", type=int, default=256,
                    help="PhoRanker là họ PhoBERT/RoBERTa -> 256 là TRẦN KIẾN TRÚC, không nâng được")
    ps.add_argument("--use-segmented-text", action="store_true", default=True,
                    help="PhoRanker BẮT BUỘC input đã tách từ (xác minh qua model card)")
    ps.add_argument("--no-segmented-text", dest="use_segmented_text", action="store_false")
    ps.add_argument("--top-n-docs", type=int, default=30,
                    help="Số DOCUMENT đầu bảng của ensemble được chấm lại. 30 phủ 97.6% (Recall@20=0.9760, @30=0.9822)")
    ps.add_argument("--batch-size", type=int, default=64)
    ps.add_argument("--device", default="auto")
    ps.add_argument("--out", default="outputs/cache/rerank_val.npz")
    ps.set_defaults(func=cmd_score)

    pb = sub.add_parser("blend", help="sweep alpha + McNemar trên tập CÓ đáp án — CPU")
    pb.add_argument("--rerank", default="outputs/cache/rerank_val.npz")
    pb.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    pb.add_argument("--dedup-groups", default=None)
    pb.add_argument("--alphas", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,3.0")
    pb.add_argument("--top-k", type=int, default=5)
    pb.add_argument("--report", default="outputs/predictions/rerank_blend_report.json")
    pb.set_defaults(func=cmd_blend)

    pu = sub.add_parser("submit", help="sinh submission.json cho public test — CPU")
    pu.add_argument("--rerank", default="outputs/cache/rerank_public.npz")
    pu.add_argument("--dedup-groups", default=None)
    pu.add_argument("--alpha", type=float, required=True)
    pu.add_argument("--top-k", type=int, default=5)
    pu.add_argument("--out", default="outputs/predictions/submission_phoranker.json")
    pu.set_defaults(func=cmd_submit)

    pt = sub.add_parser("selftest")
    pt.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
