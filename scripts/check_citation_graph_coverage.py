"""
Do "tran ceiling" cua y tuong citation-graph augmentation TRUOC KHI viet injection logic vao
full_pipeline_eval.py - tranh lang phi cong neu do thi khong thuc su cuu duoc bao nhieu tai lieu bi
miss. Cung logic nhu check_complementarity.py da dung truoc do (so sanh truc tiep predictions da co
san voi ground truth, khong can chay lai model/GPU).

Cau hoi can tra loi: trong so cac tai lieu dung bi MIEU boi predictions hien tai, bao nhieu % la
"hang xom" (1-hop) trong outputs/citation_graph.json cua MOT tai lieu DA CO trong top-5 du doan cua
chinh cau hoi do? Neu ty le nay cao (dac biet o n_relevant>=2, nhom yeu nhat) thi dang viet Phan 2
(bom vao pipeline). Neu thap thi do thi qua thua de co ich, can nghi huong khac.

Cach dung (tu thu muc goc UIT-data-science-2026, SAU KHI da chay build_citation_graph.py):
    python scripts/check_citation_graph_coverage.py

Khong can GPU, chay vai giay.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_ground_truth(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(qid): [str(d) for d in (item.get("answer") or [])] for qid, item in raw.items()}


def load_predictions(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    preds = {}
    for qid, item in raw.items():
        answer = item.get("answer") if isinstance(item, dict) else None
        preds[str(qid)] = [str(d) for d in (answer or [])]
    return preds


def load_graph(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_default_predictions() -> tuple[str, bool]:
    """Giong het logic trong analyze_multi_relevant_misses.py - uu tien file predictions THAT dung
    bm25-weight=0.1 (chinh thuc), fallback sang bm25=0.4 neu chua co."""
    official = Path("outputs/predictions/sweep_bm25_weight/bm25_weight_0.1.json")
    if official.exists():
        return str(official), False
    fallback = Path("outputs/predictions/aiteamvn_rerank_test.hybrid_fused.json")
    return str(fallback), True


def main() -> None:
    default_pred, _ = find_default_predictions()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    parser.add_argument("--predictions", default=default_pred)
    parser.add_argument("--graph", default="outputs/citation_graph.json")
    args = parser.parse_args()

    _, is_fallback = find_default_predictions()
    if args.predictions == default_pred and is_fallback:
        print(
            "[CANH BAO] Dung fallback predictions (bm25-weight=0.4), khong phai ban chinh thuc 0.1 - "
            "van dung tam duoc, chenh lech khong dang ke."
        )

    gt = load_ground_truth(args.ground_truth)
    preds = load_predictions(args.predictions)
    graph = load_graph(args.graph)

    # Gom missed doc theo n_relevant, kiem tra tung doc bi miss co phai "hang xom" (1-hop trong graph)
    # cua MOT trong cac doc DA CO trong top-5 du doan cua chinh cau hoi do hay khong.
    stats_by_n = defaultdict(lambda: {"n_questions_with_miss": 0, "n_missed": 0, "n_rescuable": 0})
    total_missed = 0
    total_rescuable = 0
    rescued_examples = []

    for qid, correct_ids in gt.items():
        if not correct_ids:
            continue
        predicted_ids = preds.get(qid, [])
        missed = [d for d in correct_ids if d not in predicted_ids]
        if not missed:
            continue

        n_rel = len(correct_ids)
        bucket = stats_by_n[n_rel]
        bucket["n_questions_with_miss"] += 1

        # Hop cac hang xom cua TAT CA doc da co trong top-5 du doan cua cau hoi nay
        neighbor_pool = set()
        for pred_doc in predicted_ids:
            neighbor_pool.update(graph.get(pred_doc, []))

        for missed_doc in missed:
            bucket["n_missed"] += 1
            total_missed += 1
            if missed_doc in neighbor_pool:
                bucket["n_rescuable"] += 1
                total_rescuable += 1
                if len(rescued_examples) < 10:
                    # tim doc nao trong predicted_ids la nguon cua canh nay, de kiem tra bang mat
                    source = next((p for p in predicted_ids if missed_doc in graph.get(p, [])), "?")
                    rescued_examples.append((qid, source, missed_doc))

    print("=" * 70)
    print(f"Nguon predictions: {args.predictions}")
    print(f"Nguon graph: {args.graph}")
    print("=" * 70)
    print(f"{'n_relevant':>10s} {'cau hoi bi miss':>16s} {'tong doc bi miss':>17s} {'cuu duoc qua graph':>19s} {'ty le':>8s}")
    for n_rel in sorted(stats_by_n):
        b = stats_by_n[n_rel]
        rate = b["n_rescuable"] / b["n_missed"] * 100 if b["n_missed"] else 0.0
        print(f"{n_rel:>10d} {b['n_questions_with_miss']:>16d} {b['n_missed']:>17d} {b['n_rescuable']:>19d} {rate:>7.1f}%")
    print("-" * 70)
    overall_rate = total_rescuable / total_missed * 100 if total_missed else 0.0
    print(f"{'TONG':>10s} {'':>16s} {total_missed:>17d} {total_rescuable:>19d} {overall_rate:>7.1f}%")
    print("=" * 70)

    print("\nVi du duoc 'cuu' (doc_id du doan -> doc_id bi miss, qua graph):")
    for qid, source, missed_doc in rescued_examples:
        print(f"  Cau hoi {qid}: du doan co {source}, graph noi toi {missed_doc} (dang bi miss)")

    print(
        "\nDIEN GIAI: ty le nay la 'tran ceiling' NEU chi bom 1 tai lieu/cau hoi bat ky lich su xep "
        "hang - THUC TE se thap hon vi con phai chon BOM VAO DAU (thay the doc nao trong top-5) va co "
        "the lam mat 1 doc dung khac dang co san. Neu ty le o day da THAP (<15-20%) thi khong dang "
        "viet Phan 2. Neu CAO (>30-40%, dac biet o n_relevant>=2) thi dang thu."
    )


if __name__ == "__main__":
    main()
