"""ĐO DƯ ĐỊA CỦA HƯỚNG "HỌ VĂN BẢN" — câu hỏi cấu trúc CUỐI CÙNG chưa được đo (2026-08-28).

BỐI CẢNH: đọc tay 20 ca sai cho thấy vài ca thuộc dạng "cùng một văn bản, khác PHIÊN BẢN/NĂM"
(đáp án Luật Thanh tra 2010, pipeline trả về Luật Thanh tra 2022) hoặc "văn bản gốc vs văn bản
SỬA ĐỔI nó" (đáp án Thông tư 59/2015, pipeline trả về Thông tư 06/2021 sửa đổi Thông tư 59/2015).

Nếu dạng này CHIẾM TỈ LỆ LỚN trong 60 ca lỗi xếp hạng thì có một cơ chế CẤU TRÚC chưa khai thác:
khi top-5 đã chứa 1 văn bản, "anh em cùng họ" của nó là ứng viên rất mạnh cho slot còn lại — và
đây KHÔNG phải đặc trưng học từ thói quen gán nhãn (không đoán chọn bản nào), nên KHÔNG dính bẫy
đã làm prior doc_type thất bại: nó chỉ THÊM ứng viên vào slot trống, không xếp lại theo thiên kiến.

Nếu dạng này HIẾM -> hướng này chết, và không còn cơ chế cấu trúc nào để thử.

Script CHỈ ĐO, không đề xuất.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

try:
    from .evaluate import load_ground_truth
    from .full_pipeline_eval import load_chunks
    from .ensemble_fusion import load_ranks, fuse_one_question, aggregate_max
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.full_pipeline_eval import load_chunks
    from src.evaluation.ensemble_fusion import load_ranks, fuse_one_question, aggregate_max


def raw_title(meta: dict) -> str:
    if meta.get("name"):
        return str(meta["name"])
    link = meta.get("link") or ""
    m = re.search(r"/([^/]+?)-\d+\.aspx", unquote(link))
    return m.group(1).replace("-", " ") if m else ""


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def family_key(title: str) -> str:
    """Khoá 'họ văn bản': bỏ NĂM và mọi chữ số -> 'Luat Thanh tra 2022' và 'Luat thanh tra 2010'
    cùng khoá 'luat thanh tra'. Cố ý thô: chỉ cần đủ để ĐẾM tỉ lệ, không phải để dùng thật."""
    t = strip_accents(title.lower())
    t = re.sub(r"\b(19|20)\d{2}\b", " ", t)     # bỏ năm
    t = re.sub(r"\d+", " ", t)                   # bỏ mọi số (số hiệu văn bản)
    t = re.sub(r"[^a-z\s]", " ", t)
    return " ".join(t.split())


def doc_number(title: str) -> str:
    """Trích 'số hiệu' dạng 59-2015-TT-BLDTBXH / 26-2008-QH12 để bắt quan hệ SỬA ĐỔI:
    tên của Thông tư 06/2021 CÓ CHỨA chuỗi '59-2015-TT-BLDTBXH' của văn bản nó sửa đổi."""
    m = re.search(r"(\d{1,4})[-/](\d{4})[-/]([A-Za-z]{2,5}\d{0,3}(?:-[A-Za-z]{2,12}\d{0,3})?)",
                  title.replace(" ", "-"))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}".lower() if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    ap.add_argument("--rank-v1", default="outputs/cache/ranks_v1.npz")
    ap.add_argument("--rank-harrier", default="outputs/cache/ranks_harrier.npz")
    ap.add_argument("--rank-bm25", default="outputs/cache/ranks_bm25.npz")
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    meta = {}
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            meta[str(o["doc_id"])] = o
    title_of = {d: raw_title(m) for d, m in meta.items()}
    fam_of = {d: family_key(t) for d, t in title_of.items()}
    num_of = {d: doc_number(t) for d, t in title_of.items()}

    chunks = load_chunks(args.chunks)
    doc_ids = [c["doc_id"] for c in chunks]
    gt = load_ground_truth(args.ground_truth)

    sources, ref_qids = {}, None
    for name, path in [("v1", args.rank_v1), ("harrier", args.rank_harrier), ("bm25", args.rank_bm25)]:
        qids, ranks = load_ranks(path)
        ref_qids = ref_qids or qids
        sources[name] = ranks
    W = {"v1": 1.0, "harrier": 0.2, "bm25": 0.1}
    U = {n: 100 for n in sources}

    n_q = n_fail = 0
    same_family = 0          # đáp án cùng HỌ với 1 doc trong top-5 (khác năm/số hiệu)
    amend_link = 0           # đáp án và 1 doc top-5 có quan hệ SỬA ĐỔI (số hiệu lồng trong tên)
    wasted_slots = 0         # top-5 có >=2 doc CÙNG HỌ -> lãng phí slot
    fam_hit_but_missed = []
    slot_waste_all = 0

    for qi, qid in enumerate(ref_qids):
        if qid not in gt:
            continue
        n_q += 1
        gold = set(gt[qid])
        c = fuse_one_question({n: sources[n][qi] for n in sources}, W, U, 60)
        top5 = aggregate_max(c, doc_ids, 5, None)

        fams = [fam_of.get(d, "") for d in top5]
        dup = sum(v - 1 for v in Counter([f for f in fams if f]).values() if v > 1)
        if dup:
            slot_waste_all += 1

        if gold & set(top5):
            continue
        n_fail += 1
        if dup:
            wasted_slots += 1

        g_fams = {fam_of.get(d, "") for d in gold if fam_of.get(d)}
        g_nums = {num_of.get(d, "") for d in gold if num_of.get(d)}
        hit_fam = bool(g_fams & {f for f in fams if f})
        hit_amend = False
        for d in top5:
            t = strip_accents(title_of.get(d, "").lower()).replace(" ", "-")
            if any(g and g in t for g in g_nums):
                hit_amend = True
            gt_titles = [strip_accents(title_of.get(x, "").lower()).replace(" ", "-") for x in gold]
            if num_of.get(d) and any(num_of[d] in gtt for gtt in gt_titles):
                hit_amend = True
        if hit_fam:
            same_family += 1
        if hit_amend:
            amend_link += 1
        if (hit_fam or hit_amend) and len(fam_hit_but_missed) < args.examples:
            fam_hit_but_missed.append((qid, sorted(gold), top5))

    print("=" * 78)
    print("DƯ ĐỊA CỦA CƠ CHẾ 'HỌ VĂN BẢN' (đo trên val_split_20)")
    print("=" * 78)
    print(f"  {n_q} câu hỏi, {n_fail} câu SAI (không có đáp án trong top-5)\n")
    print(f"  (a) Đáp án CÙNG HỌ với 1 văn bản đã nằm trong top-5 : {same_family:3d}/{n_fail} "
          f"({100*same_family/max(n_fail,1):.1f}% số câu sai)")
    print( "      (cùng tên văn bản, khác NĂM/số hiệu — vd Luật Thanh tra 2010 vs 2022)")
    print(f"  (b) Đáp án có quan hệ SỬA ĐỔI với 1 doc trong top-5 : {amend_link:3d}/{n_fail} "
          f"({100*amend_link/max(n_fail,1):.1f}%)")
    print( "      (số hiệu văn bản này lồng trong tên văn bản kia)")
    print(f"  (c) Top-5 LÃNG PHÍ slot (>=2 doc cùng họ) ở câu SAI : {wasted_slots:3d}/{n_fail} "
          f"({100*wasted_slots/max(n_fail,1):.1f}%)")
    print(f"      — trên TOÀN BỘ {n_q} câu: {slot_waste_all} câu ({100*slot_waste_all/n_q:.1f}%)")
    union = same_family + amend_link  # xấp xỉ trên (có thể trùng nhau)
    print()
    print(f"  ==> TRẦN TRÊN của cơ chế này (a hoặc b, đếm nới rộng): tối đa ~{union}/{n_fail} câu")
    print(f"      = tối đa +{union/n_q:.4f} Recall NẾU sửa được HOÀN HẢO 100% (không thực tế)")

    if fam_hit_but_missed:
        print("\n" + "=" * 78)
        print("VÍ DỤ CỤ THỂ")
        print("=" * 78)
        for qid, gold, top5 in fam_hit_but_missed:
            print(f"\n[{qid}] đáp án: {[title_of.get(d,'')[:70] for d in gold]}")
            for i, d in enumerate(top5, 1):
                print(f"   {i}. {title_of.get(d,'')[:70]}")


if __name__ == "__main__":
    main()
