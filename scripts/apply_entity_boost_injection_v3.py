"""
Ban v3 cua entity-boost injection - SUA loi da debug o v2: khi 1 ten thuc the trung trong TIEU DE cua
NHIEU van ban khac nhau (vd case 28816 "Tổng công ty Giấy Việt Nam" trung trong CA 3 van ban:
103038/129075/200437), v2 chon ung vien theo THU TU XUAT HIEN trong file chunks_corpus.jsonl (hoan
toan ngau nhien, khong lien quan gi den do lien quan that) - dan den bom NHAM van ban 103038 (mot cong
van huong dan, khong lien quan) thay vi 129075 (dung, ground truth) cho chinh case dung de thiet ke y
tuong nay. Day la nguyen nhan hop ly khien Recall tong the KHONG DOI (0.9303=0.9303) du da bom 15 cau.

Fix: khi co NHIEU ung vien trung ten (ambiguous), CHAM DIEM tung ung vien bang do tuong dong dense
(cosine similarity) giua CAU HOI va ung vien do (dung lai model + cache nhu ban citation-graph-filtered,
KHONG train/fine-tune gi them) - CHON ung vien diem CAO NHAT. Neu top-5 da day, CHI thay the hang 5 neu
ung vien tot nhat > diem hang 5 hien tai (tranh bom mu nhu bai hoc citation-graph).

Cach dung (tu thu muc goc UIT-data-science-2026):
    python scripts/apply_entity_boost_injection_v3.py

Sau do do lai bang evaluate.py nhu moi lan khac, so sanh voi ban goc VA ban v2 (khong loc):
    python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/sweep_bm25_weight/bm25_weight_0.1.json
    python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/entity_boost_injected.json
    python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/entity_boost_injected_v3.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from apply_entity_boost_injection import (
    extract_entities,
    normalize,
    load_titles,
    load_ground_truth_questions,
    load_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    parser.add_argument("--predictions", default="outputs/predictions/sweep_bm25_weight/bm25_weight_0.1.json")
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--embedding-cache", default="outputs/cache/embcompare_aiteamvn.npz")
    parser.add_argument("--embedding-model", default="AITeamVN/Vietnamese_Embedding")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--output", default="outputs/predictions/entity_boost_injected_v3.json")
    parser.add_argument("--max-inject", type=int, default=1)
    parser.add_argument(
        "--margin", type=float, default=0.0,
        help="Chi THAY THE hang 5 neu diem ung vien tot nhat > diem hang 5 + margin",
    )
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Thieu sentence-transformers. Chay: pip install sentence-transformers torch", file=sys.stderr)
        sys.exit(1)
    import numpy as np
    import torch

    question_texts = load_ground_truth_questions(args.ground_truth)
    preds = load_predictions(args.predictions)
    print(f"Doc tieu de (chunk dau) tu {args.chunks}...")
    titles = load_titles(args.chunks)
    titles_norm = {doc_id: normalize(t) for doc_id, t in titles.items()}
    print(f"  {len(titles_norm)} van ban.")

    print(f"Load cache embedding corpus tu {args.embedding_cache}...")
    cache = np.load(args.embedding_cache, allow_pickle=True)
    corpus_emb = cache["embeddings"]
    corpus_doc_ids = cache["doc_ids"]
    doc_to_indices: dict = {}
    for idx, doc_id in enumerate(corpus_doc_ids):
        doc_to_indices.setdefault(str(doc_id), []).append(idx)
    print(f"  {len(corpus_emb)} chunk, {len(doc_to_indices)} van ban rieng biet trong cache.")

    # --- Buoc 1: xac dinh cau hoi nao CAN xu ly (co entity, entity VANG MAT khoi top-5, VA co it nhat
    # 1 ung vien trong corpus) - giong dung logic o v2, chi khac la GOM ung vien lai thay vi bom luon.
    per_qid_candidates: dict = {}
    n_with_entity = 0
    n_already_covered = 0
    n_missing_from_top5 = 0
    n_ambiguous = 0  # co >= 2 ung vien trung ten - day la truong hop v2 chon SAI (theo thu tu file)
    for qid, predicted_ids in preds.items():
        q_text = question_texts.get(qid, "")
        entities = extract_entities(q_text)
        if not entities:
            continue
        n_with_entity += 1
        already_covered = any(
            normalize(e) in titles_norm.get(doc_id, "") for doc_id in predicted_ids for e in entities
        )
        if already_covered:
            n_already_covered += 1
            continue
        n_missing_from_top5 += 1
        candidates = set()
        for e in entities:
            e_norm = normalize(e)
            for doc_id, t in titles_norm.items():
                if doc_id in predicted_ids:
                    continue
                if e_norm in t:
                    candidates.add(doc_id)
        if candidates:
            if len(candidates) > 1:
                n_ambiguous += 1
            per_qid_candidates[qid] = list(candidates)

    print(f"So cau hoi phat hien duoc >=1 thuc the: {n_with_entity}")
    print(f"  - Da co san trong top-5: {n_already_covered}")
    print(f"  - VANG MAT khoi top-5: {n_missing_from_top5}")
    print(f"    - Co ung vien trong corpus: {len(per_qid_candidates)}")
    print(f"    - MO HO (>=2 van ban trung ten - day la cho v2 co the chon SAI): {n_ambiguous}")

    # --- Buoc 2: encode CHI cac cau hoi co ung vien can cham diem.
    qids_to_encode = list(per_qid_candidates.keys())
    print(f"\nSo cau hoi can encode de cham diem ung vien: {len(qids_to_encode)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Load embedding model {args.embedding_model} (device={device})...")
    model = SentenceTransformer(args.embedding_model, device=device)
    model.max_seq_length = args.max_seq_length

    texts = [args.query_prefix + question_texts.get(qid, "") for qid in qids_to_encode]
    if texts:
        query_emb = model.encode(
            texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
        ).astype(np.float32)
    else:
        query_emb = np.zeros((0, 0), dtype=np.float32)
    qid_to_vec = {qid: query_emb[i] for i, qid in enumerate(qids_to_encode)}

    def doc_score(vec, doc_id: str) -> float:
        idxs = doc_to_indices.get(doc_id)
        if not idxs:
            return -1.0
        return float(corpus_emb[idxs].dot(vec).max())

    # --- Buoc 3: voi moi cau hoi, CHON ung vien diem cao nhat (thay vi "dau tien tim thay" nhu v2),
    # roi chi bom neu ung vien do THUC SU tot hon hang 5 hien tai (neu top-5 da day).
    n_injected_questions = 0
    n_total_injected = 0
    n_rejected_by_margin = 0
    examples = []
    output = {}
    for qid, predicted_ids in preds.items():
        result = list(predicted_ids)
        vec = qid_to_vec.get(qid)
        candidates = per_qid_candidates.get(qid)
        if vec is not None and candidates:
            scored = sorted(((doc_score(vec, c), c) for c in candidates), reverse=True)
            best_score, best_doc = scored[0]
            n_inj = 0
            if len(result) >= 5:
                rank5_doc = result[-1]
                rank5_score = doc_score(vec, rank5_doc)
                if best_score > rank5_score + args.margin:
                    result[-1] = best_doc
                    n_inj = 1
                else:
                    n_rejected_by_margin += 1
            else:
                result.append(best_doc)
                n_inj = 1
            if n_inj > 0:
                n_injected_questions += 1
                n_total_injected += n_inj
                if len(examples) < 15:
                    examples.append((qid, len(candidates), best_doc, best_score))
        output[qid] = {"answer": result}

    print("=" * 70)
    print(f"So cau hoi THUC SU bi sua doi (da bom, sau khi chon ung vien tot nhat + loc margin): "
          f"{n_injected_questions} ({n_injected_questions/len(preds)*100:.1f}%)")
    print(f"Tong so tai lieu da bom: {n_total_injected}")
    print(f"So cau hoi bi TU CHOI vi ung vien tot nhat KHONG hon hang 5 hien tai: {n_rejected_by_margin}")
    print("=" * 70)
    print("\nVi du da bom (qid, so_ung_vien_trung_ten, doc_id_duoc_chon, diem):")
    for qid, n_cand, doc_id, score in examples:
        print(f"  Cau hoi {qid}: {n_cand} ung vien trung ten -> chon doc_id={doc_id} (diem={score:.4f})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu predictions MOI (v3, co cham diem chon ung vien) tai: {args.output}")
    print(
        "\nBuoc tiep theo - do va so sanh Recall CA 3 ban (goc / v2 khong loc / v3 co loc+chon dung):\n"
        f"  python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions {args.predictions}\n"
        f"  python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/entity_boost_injected.json\n"
        f"  python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions {args.output}\n"
        "Gui lai CA 3 ket qua (dong 'Recall trung binh' + bang breakdown theo n_relevant)."
    )


if __name__ == "__main__":
    main()
