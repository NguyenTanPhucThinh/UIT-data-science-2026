"""CHẨN ĐOÁN TRẦN THẬT CỦA PIPELINE — trả lời câu hỏi "còn dư địa hay đã bão hoà?" (2026-08-28).

VÌ SAO CẦN FILE NÀY: suốt ~15 thử nghiệm vừa qua, MỌI hướng đều thuộc CÙNG một phạm trù —
"thêm/đổi/đánh lại trọng số một tín hiệu xếp hạng trên cùng một pool ứng viên". Chưa ai đo xem
lỗi còn lại THUỘC LOẠI NÀO. Có đúng 3 loại, và chúng đòi hỏi 3 loại giải pháp KHÁC HẲN NHAU:

  (A) LỖI RECALL   — tài liệu đúng KHÔNG hề có trong pool top-100 của bất kỳ nguồn nào.
                     => rerank/fusion/đổi trọng số VĨNH VIỄN không sửa được (không thể xếp hạng
                        thứ mình chưa từng lấy về). Chỉ mở rộng/đổi CÁCH LẤY VỀ mới sửa được.
  (B) LỖI XẾP HẠNG — tài liệu đúng CÓ trong pool nhưng nằm ngoài top-5.
                     => đây LÀ đất của rerank/fusion. Nếu (B) nhỏ thì mọi hướng đã thử tất yếu
                        phải cho kết quả ~0, KHÔNG phải vì làm sai mà vì không còn gì để lấy.
  (C) TRẦN DỮ LIỆU — ground truth sai/thiếu, không cách nào đúng được.

Tỉ lệ A/B/C quyết định TOÀN BỘ chiến lược còn lại. Script này KHÔNG đề xuất gì cả, chỉ đo.

CÁCH DÙNG (từ thư mục GỐC repo):
    python -m src.evaluation.diagnose_ceiling
    python -m src.evaluation.diagnose_ceiling --dump-failures outputs/predictions/failures.json

Thuần CPU, dùng lại cache ranks_*.npz có sẵn, KHÔNG encode lại gì, chạy trong ~1-2 phút.
"""
from __future__ import annotations

import argparse
import json
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


def rank_all_documents(rrf_score, doc_ids, group_of=None):
    """Giống aggregate_max() nhưng KHÔNG cắt top_k — trả về TOÀN BỘ document đã xếp hạng.
    Cần bản đầy đủ để biết tài liệu đúng đang nằm ở hạng bao nhiêu (5? 12? 80?)."""
    best = {}
    for chunk_idx, score in rrf_score.items():
        doc_id = doc_ids[chunk_idx]
        if doc_id not in best or score > best[doc_id]:
            best[doc_id] = score
    ranked = sorted(best.items(), key=lambda x: -x[1])
    if not group_of:
        return [d for d, _ in ranked]
    out, seen = [], set()
    for doc_id, _ in ranked:
        g = group_of.get(doc_id)
        if g is not None:
            if g in seen:
                continue
            seen.add(g)
        out.append(doc_id)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    ap.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    ap.add_argument("--dedup-groups", default=None)
    ap.add_argument("--rank-v1", default="outputs/cache/ranks_v1.npz")
    ap.add_argument("--rank-harrier", default="outputs/cache/ranks_harrier.npz")
    ap.add_argument("--rank-bm25", default="outputs/cache/ranks_bm25.npz")
    ap.add_argument("--dump-failures", default=None)
    args = ap.parse_args()

    print("Đang nạp chunks (file lớn, ~1 phút)...")
    chunks = load_chunks(args.chunks)
    doc_ids = [c["doc_id"] for c in chunks]
    print(f"  {len(chunks):,} chunk / {len(set(doc_ids)):,} document")

    gt = load_ground_truth(args.ground_truth)
    group_of = load_dup_groups(args.dedup_groups)

    sources = {}
    ref_qids = None
    for name, path in [("v1", args.rank_v1), ("harrier", args.rank_harrier), ("bm25", args.rank_bm25)]:
        qids, ranks = load_ranks(path)
        if ref_qids is None:
            ref_qids = qids
        sources[name] = ranks
        print(f"  {name:8s} top-{ranks.shape[1]} x {len(qids)} câu hỏi")

    qids = [q for q in ref_qids if q in gt]
    print(f"\n{len(qids)} câu hỏi có ground truth.\n")

    # ---- 0. Cấu trúc ground truth -------------------------------------------------
    gt_sizes = Counter(len(gt[q]) for q in qids)
    print("=" * 78)
    print("0. GROUND TRUTH CÓ MẤY TÀI LIỆU ĐÚNG MỖI CÂU?")
    print("=" * 78)
    for size in sorted(gt_sizes):
        n = gt_sizes[size]
        print(f"  {size} tài liệu đúng : {n:5d} câu ({100*n/len(qids):5.1f}%)")
    avg_gt = sum(len(gt[q]) for q in qids) / len(qids)
    print(f"  Trung bình: {avg_gt:.3f} tài liệu đúng/câu")
    print(f"  => Trần Precision@5 lý thuyết = {avg_gt/5:.4f} (Precision KHÔNG phải chỉ số đáng tối ưu)")

    # ---- 1. Trần pool: tài liệu đúng có NẰM TRONG pool không? ---------------------
    weights = {"v1": 1.0, "harrier": 0.2, "bm25": 0.1}
    pool_hit = {name: 0 for name in sources}
    pool_hit["UNION"] = 0
    full_cover = {name: 0 for name in sources}
    full_cover["UNION"] = 0

    rank_of_gt = {}          # qid -> hạng tốt nhất của 1 tài liệu GT trong xếp hạng document cuối
    doc_ranking_cache = {}
    recall_at = defaultdict(float)
    DEPTHS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200, 500]

    for qi, qid in enumerate(ref_qids):
        if qid not in gt:
            continue
        gold = set(gt[qid])

        # pool theo từng nguồn (document nào chạm tới được qua top-100 chunk)
        union_docs = set()
        for name, ranks in sources.items():
            docs = {doc_ids[int(i)] for i in ranks[qi]}
            union_docs |= docs
            if gold & docs:
                pool_hit[name] += 1
            if gold <= docs:
                full_cover[name] += 1
        if gold & union_docs:
            pool_hit["UNION"] += 1
        if gold <= union_docs:
            full_cover["UNION"] += 1

        # xếp hạng document THẬT của cấu hình production
        rank_rows = {name: sources[name][qi] for name in sources}
        combined = fuse_one_question(rank_rows, weights, {n: 100 for n in sources}, 60)
        ranked_docs = rank_all_documents(combined, doc_ids, group_of)
        doc_ranking_cache[qid] = ranked_docs

        pos = {d: i + 1 for i, d in enumerate(ranked_docs)}
        gold_positions = sorted(pos[d] for d in gold if d in pos)
        rank_of_gt[qid] = gold_positions[0] if gold_positions else None

        for depth in DEPTHS:
            hit = sum(1 for p in gold_positions if p <= depth)
            recall_at[depth] += hit / len(gold)

    n = len(qids)
    print()
    print("=" * 78)
    print("1. TÀI LIỆU ĐÚNG CÓ NẰM TRONG POOL ỨNG VIÊN KHÔNG? (top-100 chunk mỗi nguồn)")
    print("=" * 78)
    print("   'chạm được' = ít nhất 1 tài liệu đúng có mặt; 'phủ hết' = TẤT CẢ tài liệu đúng có mặt")
    print(f"   {'nguồn':10s} {'chạm được':>12s} {'phủ hết':>12s}")
    for name in ["v1", "harrier", "bm25", "UNION"]:
        print(f"   {name:10s} {100*pool_hit[name]/n:11.2f}% {100*full_cover[name]/n:11.2f}%")
    print()
    print(f"   ==> TRẦN TUYỆT ĐỐI của mọi kỹ thuật fusion/rerank trên pool hiện tại: {100*full_cover['UNION']/n:.2f}%")
    print( "       (không kỹ thuật xếp hạng lại nào vượt được con số này — không thể xếp hạng")
    print( "        thứ chưa bao giờ được lấy về)")

    # ---- 2. Recall theo độ sâu ----------------------------------------------------
    print()
    print("=" * 78)
    print("2. RECALL THEO ĐỘ SÂU — cấu hình production (v1=1, harrier=0.2, bm25=0.1, rrf_k=60)")
    print("=" * 78)
    prev = 0.0
    for depth in DEPTHS:
        r = recall_at[depth] / n
        gain = r - prev
        bar = "#" * int(r * 50)
        print(f"   Recall@{depth:<4d} = {r:.4f}  (+{gain:.4f})  {bar}")
        prev = r

    r5 = recall_at[5] / n
    r100 = recall_at[100] / n
    print()
    print(f"   Recall@5   = {r5:.4f}   <- điểm đang tối ưu")
    print(f"   Recall@100 = {r100:.4f}   <- trần nếu rerank HOÀN HẢO trong pool")
    print(f"   KHOẢNG TRỐNG rerank có thể ăn được = {r100 - r5:.4f}")

    # ---- 3. Phân loại lỗi ---------------------------------------------------------
    fail_notinpool, fail_ranking, fails = [], [], []
    for qid in qids:
        gold = set(gt[qid])
        ranked = doc_ranking_cache[qid]
        top5 = set(ranked[:5])
        if gold & top5:
            continue  # coi là đạt (ít nhất 1 tài liệu đúng lọt top-5)
        fails.append(qid)
        r = rank_of_gt[qid]
        if r is None:
            fail_notinpool.append(qid)
        else:
            fail_ranking.append((qid, r))

    print()
    print("=" * 78)
    print("3. PHÂN LOẠI CÁC CÂU ĐANG SAI (không có tài liệu đúng nào trong top-5)")
    print("=" * 78)
    print(f"   Tổng số câu sai: {len(fails)} / {n}  ({100*len(fails)/n:.2f}%)")
    print()
    print(f"   (A) LỖI RECALL  — tài liệu đúng KHÔNG có trong pool : {len(fail_notinpool):4d} "
          f"({100*len(fail_notinpool)/max(len(fails),1):5.1f}% số câu sai)")
    print( "       => fusion/rerank/đổi trọng số KHÔNG BAO GIỜ sửa được nhóm này")
    print(f"   (B) LỖI XẾP HẠNG — có trong pool, hạng 6-100          : {len(fail_ranking):4d} "
          f"({100*len(fail_ranking)/max(len(fails),1):5.1f}% số câu sai)")
    print( "       => đây là TOÀN BỘ đất còn lại cho fusion/rerank")
    if fail_ranking:
        buckets = Counter()
        for _, r in fail_ranking:
            if r <= 10: buckets["hạng 6-10"] += 1
            elif r <= 20: buckets["hạng 11-20"] += 1
            elif r <= 50: buckets["hạng 21-50"] += 1
            else: buckets["hạng >50"] += 1
        print("       Phân bố hạng của tài liệu đúng trong nhóm (B):")
        for k in ["hạng 6-10", "hạng 11-20", "hạng 21-50", "hạng >50"]:
            if buckets[k]:
                print(f"         {k:12s}: {buckets[k]:4d}")

    # ---- 4. Sàn nhiễu thống kê ----------------------------------------------------
    print()
    print("=" * 78)
    print("4. SÀN NHIỄU CỦA PHÉP ĐO LOCAL (vì sao mọi Δ nhỏ đều 'không đáng kể')")
    print("=" * 78)
    import math
    se = math.sqrt(r5 * (1 - r5) / n)
    print(f"   n = {n} câu, Recall = {r5:.4f}")
    print(f"   Sai số chuẩn của CHÍNH con số này: {se:.4f}")
    print(f"   Khoảng tin cậy 95% (không ghép cặp): ±{1.96*se:.4f}  -> [{r5-1.96*se:.4f}, {r5+1.96*se:.4f}]")
    print()
    print(f"   => Mọi cải thiện < ~{1.96*se:.3f} nằm TRONG sai số của chính phép đo khi so 2 con số rời rạc.")
    print( "      Đây là lý do TOÁN HỌC (không phải xui) khiến quy tắc >=0.01-0.02 đã đúc kết từ")
    print( "      kinh nghiệm là đúng: với 1398 câu, ta VẬT LÝ không phân biệt nổi cải thiện nhỏ hơn thế.")
    print()
    print( "   LỐI THOÁT: so sánh GHÉP CẶP (cùng tập câu hỏi) nhạy hơn NHIỀU — chỉ cần đếm số câu")
    print( "   ĐỔI KẾT QUẢ (b = sai->đúng, c = đúng->sai), dùng McNemar. Nếu b+c nhỏ (vd 20 câu)")
    print( "   thì 12-8 là nhiễu, nhưng 18-2 lại có ý nghĩa THẬT dù Δ chỉ +0.011.")
    print( "   => Từ giờ mọi so sánh A/B nên in ra (b, c) chứ không chỉ 2 con số Recall.")

    if args.dump_failures:
        out = {
            "recall_at_5": r5, "recall_at_100": r100,
            "pool_ceiling_union": full_cover["UNION"] / n,
            "fail_not_in_pool": fail_notinpool,
            "fail_ranking": [{"qid": q, "rank_of_gold": r} for q, r in fail_ranking],
        }
        Path(args.dump_failures).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_failures, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu danh sách câu sai: {args.dump_failures}")


if __name__ == "__main__":
    main()
