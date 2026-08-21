"""
Debug 1 CAU HOI CU THE cua apply_entity_boost_injection.py - vi sao ket qua tong the KHONG DOI
(0.9303 = 0.9303 tuyet doi) du da bom 15 cau hoi, va vi sao chinh case da dung de thiet ke y tuong
(28816, "Tổng công ty Giấy Việt Nam") van chua duoc sua.

In ra: entity trich xuat duoc, top-5 hien tai (co tieu de), VA TOAN BO cac van ban trong corpus co
tieu de chua cum entity do (khong chi 1 cai dau tien nhu script chinh) - de xem co bi CHON NHAM giua
nhieu ung vien hay khong co ung vien nao ca.

Cach dung:
    python scripts/debug_entity_case.py --qid 28816
"""
from __future__ import annotations

import argparse
import json

from apply_entity_boost_injection import extract_entities, normalize, load_titles, load_ground_truth_questions, load_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", default="28816")
    parser.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    parser.add_argument("--predictions", default="outputs/predictions/sweep_bm25_weight/bm25_weight_0.1.json")
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    args = parser.parse_args()

    question_texts = load_ground_truth_questions(args.ground_truth)
    preds = load_predictions(args.predictions)
    titles = load_titles(args.chunks)
    titles_norm = {doc_id: normalize(t) for doc_id, t in titles.items()}

    with open(args.ground_truth, encoding="utf-8") as f:
        gt_raw = json.load(f)
    correct_ids = [str(d) for d in (gt_raw.get(args.qid, {}).get("answer") or [])]

    q_text = question_texts.get(args.qid, "")
    print(f"Cau hoi {args.qid}: {q_text}")
    print(f"Tai lieu DUNG (ground truth): {correct_ids}")
    for doc_id in correct_ids:
        print(f"  - {doc_id}: {titles.get(doc_id, '(KHONG CO trong titles - kiem tra doc_id)')[:200]}")

    entities = extract_entities(q_text)
    print(f"\nEntity trich xuat duoc: {entities}")

    predicted_ids = preds.get(args.qid, [])
    print(f"\nTop-5 HIEN TAI (truoc injection):")
    for doc_id in predicted_ids:
        mark = " <-- DUNG" if doc_id in correct_ids else ""
        print(f"  - {doc_id}{mark}: {titles.get(doc_id, '?')[:150]}")

    for e in entities:
        e_norm = normalize(e)
        print(f"\n--- Tim TAT CA van ban co tieu de chua '{e}' ---")
        matches = [(doc_id, t) for doc_id, t in titles_norm.items() if e_norm in t]
        print(f"Tong so tim thay: {len(matches)}")
        for doc_id, _ in matches[:20]:
            mark = " <-- DUNG (ground truth)" if doc_id in correct_ids else ""
            in_top5 = " [DA CO trong top-5]" if doc_id in predicted_ids else ""
            print(f"  - {doc_id}{mark}{in_top5}: {titles.get(doc_id, '?')[:200]}")


if __name__ == "__main__":
    main()
