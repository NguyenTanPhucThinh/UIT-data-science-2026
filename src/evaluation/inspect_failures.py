"""XEM TẬN MẮT CÁC CÂU ĐANG SAI — bước cuối của chuỗi chẩn đoán (2026-08-28).

Chuỗi chẩn đoán đã có:
  diagnose_ceiling.py -> 85.7% câu sai là LỖI XẾP HẠNG (tài liệu đúng CÓ trong pool,
                          32/60 nằm ngay hạng 6-10). Trần pool = 99.00%.
  test_length_bias.py -> giả thuyết "thiên lệch độ dài khi gộp max" BỊ BÁC BỎ
                          (tỉ lệ đè/đúng chỉ 1.14x, 54.3% ~ tung đồng xu, penalty giảm đơn điệu).

Còn lại 2 khả năng, và chỉ ĐỌC THẬT mới phân biệt được:
  (B1) Tài liệu đè và tài liệu đúng CÙNG chủ đề, khác nhau ở chi tiết mà embedding không
       thể thấy (văn bản khung vs văn bản chuyên biệt, còn hiệu lực vs hết hiệu lực,
       Luật vs Nghị định vs Thông tư hướng dẫn). => CÓ tín hiệu khai thác được, nhưng là
       tín hiệu CẤU TRÚC/METADATA, không phải tín hiệu ngữ nghĩa văn bản.
  (B2) Ground truth chọn 1 trong nhiều văn bản đều hợp lý như nhau. => TRẦN DỮ LIỆU,
       không tối ưu thêm được, và mọi Δ nhỏ đo được trên nhóm này chỉ là nhiễu.

Script CHỈ XUẤT DỮ LIỆU để người đọc, KHÔNG tự kết luận.

CÁCH DÙNG:  python -m src.evaluation.inspect_failures --limit 25 --out outputs/predictions/failures_readable.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

try:
    from .evaluate import load_ground_truth
    from .full_pipeline_eval import load_chunks
    from .ensemble_fusion import load_ranks, load_dup_groups, fuse_one_question, aggregate_max
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth
    from src.evaluation.full_pipeline_eval import load_chunks
    from src.evaluation.ensemble_fusion import load_ranks, load_dup_groups, fuse_one_question, aggregate_max


def title_of(meta: dict) -> str:
    """Tên văn bản: ưu tiên field name, fallback suy từ slug URL (đa số name=null)."""
    if meta.get("name"):
        return str(meta["name"])
    link = meta.get("link") or ""
    m = re.search(r"/([^/]+?)-\d+\.aspx", unquote(link))
    if m:
        return m.group(1).replace("-", " ")
    return f"(không có tên) {link[-60:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    ap.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    ap.add_argument("--rank-v1", default="outputs/cache/ranks_v1.npz")
    ap.add_argument("--rank-harrier", default="outputs/cache/ranks_harrier.npz")
    ap.add_argument("--rank-bm25", default="outputs/cache/ranks_bm25.npz")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--out", default="outputs/predictions/failures_readable.txt")
    args = ap.parse_args()

    meta = {}
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            meta[str(o["doc_id"])] = {"name": o.get("name"), "link": o.get("link"),
                                      "doc_type": o.get("doc_type"), "len": o.get("passage_length")}

    qtext = {}
    try:
        with open(args.questions, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                qid = str(o.get("question_id") or o.get("qid") or o.get("id") or "")
                if qid:
                    qtext[qid] = o.get("question") or ""
    except FileNotFoundError:
        print(f"[cảnh báo] không thấy {args.questions} — sẽ không in được câu hỏi", file=sys.stderr)

    chunks = load_chunks(args.chunks)
    doc_ids = [c["doc_id"] for c in chunks]
    gt = load_ground_truth(args.ground_truth)

    sources, ref_qids = {}, None
    for name, path in [("v1", args.rank_v1), ("harrier", args.rank_harrier), ("bm25", args.rank_bm25)]:
        qids, ranks = load_ranks(path)
        ref_qids = ref_qids or qids
        sources[name] = ranks

    weights = {"v1": 1.0, "harrier": 0.2, "bm25": 0.1}
    use_top = {n: 100 for n in sources}

    out_lines = []
    n_written = 0
    for qi, qid in enumerate(ref_qids):
        if qid not in gt or n_written >= args.limit:
            continue
        gold = set(gt[qid])
        rank_rows = {n: sources[n][qi] for n in sources}
        combined = fuse_one_question(rank_rows, weights, use_top, 60)
        top5 = aggregate_max(combined, doc_ids, 5, None)
        if gold & set(top5):
            continue
        n_written += 1
        out_lines.append("=" * 100)
        out_lines.append(f"[{n_written}] question_id={qid}")
        out_lines.append(f"CÂU HỎI: {qtext.get(qid, '(không đọc được)')[:400]}")
        out_lines.append("")
        out_lines.append("  >>> TÀI LIỆU ĐÚNG (ground truth) — KHÔNG lọt top-5:")
        for d in sorted(gold):
            m = meta.get(d, {})
            out_lines.append(f"      [{d}] ({m.get('doc_type')}, {m.get('len')} ký tự)")
            out_lines.append(f"          {title_of(m)[:150]}")
        out_lines.append("")
        out_lines.append("  >>> 5 TÀI LIỆU PIPELINE TRẢ VỀ (đã đè tài liệu đúng):")
        for i, d in enumerate(top5, 1):
            m = meta.get(d, {})
            out_lines.append(f"      {i}. [{d}] ({m.get('doc_type')}, {m.get('len')} ký tự)")
            out_lines.append(f"          {title_of(m)[:150]}")
        out_lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"Đã ghi {n_written} ca sai vào {args.out}")


if __name__ == "__main__":
    main()
