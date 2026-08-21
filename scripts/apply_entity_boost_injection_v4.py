"""
Ban v4 - tach 2 cau hoi rieng ma v3 dang GOP LAM MOT:
  (1) "trong nhieu ung vien TRUNG TEN, ung vien nao dung?" - v3 da giai bang diem dense, GIU LAI o day.
  (2) "co nen BOM khong?" - v3 dung CHINH model dense (AITeamVN/Vietnamese_Embedding) de tra loi cau
      nay, nhung day la VONG LAP TU MAU THUAN: cac case entity-boost nham toi la NHUNG CASE MA CHINH
      MODEL DENSE DO DA BI NHAM (giua cac van ban "khuon mau" cung cau truc, khac ten co quan) - nen
      khi dung LAI no de "tham dinh" ung vien dung, no gan nhu chac chan se tu choi dung nhung case can
      sua (da xay ra: 15/15 bi tu choi o v3, Recall khong doi). Dung 1 tin hieu da SAI de kiem tra lai
      chinh no khong co y nghia.

v4 lam 2 viec:
  (a) THEM co --always-inject: bo dieu kien margin, BOM THANG ung vien diem cao nhat (van chon dung
      bang diem dense trong so cac ung vien TRUNG TEN - phan sua o v3 van giu), de tach rieng xem PHAN
      CHON DUNG UNG VIEN (khong lien quan gi den margin) co giup Recall tang khong.
  (b) THEM debug in ra DIEM SO THO cua tung ung vien (--debug-qid, mac dinh 28816) de xem thuc su model
      dense danh gia van ban DUNG (129075) thap hon hang 5 hien tai bao nhieu - de biet day la truong
      hop "gan bang nhau, sit sao" hay "chenh lech lon, model that su khong phan biet duoc".

Cach dung (tu thu muc goc UIT-data-science-2026):
    python scripts/apply_entity_boost_injection_v4.py --always-inject
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
    parser.add_argument("--output", default="outputs/predictions/entity_boost_injected_v4.json")
    parser.add_argument("--max-inject", type=int, default=1)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument(
        "--always-inject", action="store_true",
        help="Bo qua dieu kien margin - BOM THANG ung vien diem cao nhat (van dung diem de CHON dung "
             "trong so cac ung vien trung ten, chi khong dung diem do de QUYET DINH co bom hay khong)",
    )
    parser.add_argument("--debug-qid", default="28816", help="In diem so tho cua tung ung vien cho qid nay de doi chieu")
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

    per_qid_candidates: dict = {}
    n_with_entity = 0
    n_already_covered = 0
    n_missing_from_top5 = 0
    n_ambiguous = 0
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
    print(f"    - MO HO (>=2 van ban trung ten): {n_ambiguous}")

    qids_to_encode = list(per_qid_candidates.keys())
    if args.debug_qid and args.debug_qid not in per_qid_candidates and args.debug_qid in preds:
        qids_to_encode.append(args.debug_qid)  # dam bao debug-qid duoc encode du khong co candidate (vd de kiem tra)
    print(f"\nSo cau hoi can encode: {len(qids_to_encode)}")

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

    # --- Debug: in diem so tho cho 1 qid cu the, DOC LAP voi logic bom o duoi ---
    if args.debug_qid in qid_to_vec:
        print("\n" + "=" * 70)
        print(f"DEBUG diem so cho qid={args.debug_qid}: {question_texts.get(args.debug_qid, '')}")
        vec = qid_to_vec[args.debug_qid]
        predicted_ids = preds.get(args.debug_qid, [])
        print("Diem cac van ban dang o top-5 hien tai:")
        for doc_id in predicted_ids:
            print(f"    {doc_id}: {doc_score(vec, doc_id):.4f}  ({titles.get(doc_id, '?')[:80]!r})")
        cands = per_qid_candidates.get(args.debug_qid, [])
        print("Diem cac ung vien tim duoc qua entity match:")
        for doc_id in cands:
            print(f"    {doc_id}: {doc_score(vec, doc_id):.4f}  ({titles.get(doc_id, '?')[:80]!r})")
        print("=" * 70)

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
            if len(result) >= 5 and not args.always_inject:
                rank5_doc = result[-1]
                rank5_score = doc_score(vec, rank5_doc)
                if best_score > rank5_score + args.margin:
                    result[-1] = best_doc
                    n_inj = 1
                else:
                    n_rejected_by_margin += 1
            elif len(result) >= 5:
                result[-1] = best_doc
                n_inj = 1
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
    print(f"Che do: {'LUON BOM (bo margin)' if args.always_inject else 'CO LOC margin'}")
    print(f"So cau hoi THUC SU bi sua doi: {n_injected_questions} ({n_injected_questions/len(preds)*100:.1f}%)")
    print(f"Tong so tai lieu da bom: {n_total_injected}")
    if not args.always_inject:
        print(f"So cau hoi bi TU CHOI vi ung vien tot nhat KHONG hon hang 5: {n_rejected_by_margin}")
    print("=" * 70)
    print("\nVi du da bom (qid, so_ung_vien_trung_ten, doc_id_duoc_chon, diem):")
    for qid, n_cand, doc_id, score in examples:
        print(f"  Cau hoi {qid}: {n_cand} ung vien -> chon doc_id={doc_id} (diem={score:.4f})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu predictions MOI tai: {args.output}")
    print(
        "\nBuoc tiep theo - do Recall:\n"
        f"  python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions {args.output}\n"
        "Gui lai ket qua (dong 'Recall trung binh' + bang breakdown) VA toan bo phan DEBUG diem so o tren."
    )


if __name__ == "__main__":
    main()
