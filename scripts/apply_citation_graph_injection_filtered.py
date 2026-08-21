"""
Ban CO LOC cua apply_citation_graph_injection.py - CHI bom tai lieu lien ket qua citation graph NEU
diem tuong dong dense (cosine similarity) giua CAU HOI va tai lieu do CAO HON diem cua tai lieu dang
o hang thap nhat (hang 5, sap bi thay the) - tranh tinh trang bom mu nhu ban truoc (da do: Recall
GIAM tu 0.9303 xuong 0.9201, vi bom ca nhung tai lieu lien ket ve mat PHAP LY nhung khong lien quan
gi den CAU HOI cu the).

Dung LAI model pretrained dang dung chinh thuc (AITeamVN/Vietnamese_Embedding, KHONG train/fine-tune
gi them) + cache embedding corpus da co san (outputs/cache/embcompare_aiteamvn.npz, KHONG encode lai
corpus). Phan MOI DUY NHAT can tinh: encode cau hoi cua val_split_20 (nhanh, vai giay - vai phut, chi
~1398 cau hoi NGAN, khac han encode ca corpus 351K chunk tung ton hang gio) - va thuc te chi can
encode nhung cau hoi CO KHA NANG bi injection (co it nhat 1 hang xom moi trong graph), it hon 1398.

Cach dung (tu thu muc goc UIT-data-science-2026, SAU KHI da co outputs/citation_graph.json):
    python scripts/apply_citation_graph_injection_filtered.py

Sau do do lai bang evaluate.py, so sanh CA 3 ban (goc / injection tho khong loc / injection da loc)
de xem loc co giup lat nguoc ket qua AM -0.0102 da do truoc do khong.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_ground_truth_questions(path: str) -> dict:
    """val_split_20.json co schema {qid: {"question":..., "answer":[...]}} - CHI lay question text o
    day, KHONG dung field answer cho quyet dinh gi (khong ro ri ground truth vao logic injection)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(qid): item.get("question", "") for qid, item in raw.items()}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    parser.add_argument("--predictions", default="outputs/predictions/sweep_bm25_weight/bm25_weight_0.1.json")
    parser.add_argument("--graph", default="outputs/citation_graph.json")
    parser.add_argument("--embedding-cache", default="outputs/cache/embcompare_aiteamvn.npz")
    parser.add_argument("--embedding-model", default="AITeamVN/Vietnamese_Embedding")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--output", default="outputs/predictions/citation_graph_injected_filtered.json")
    parser.add_argument("--max-inject", type=int, default=1)
    parser.add_argument(
        "--margin", type=float, default=0.0,
        help="Chi bom neu diem_candidate > diem_hang5 + margin (0 = chi can nhinh hon 1 chut cung duoc)",
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
    graph = load_graph(args.graph)

    print(f"Load cache embedding corpus tu {args.embedding_cache}...")
    cache = np.load(args.embedding_cache, allow_pickle=True)
    corpus_emb = cache["embeddings"]
    corpus_doc_ids = cache["doc_ids"]
    doc_to_indices: dict = {}
    for idx, doc_id in enumerate(corpus_doc_ids):
        doc_to_indices.setdefault(str(doc_id), []).append(idx)
    print(f"  {len(corpus_emb)} chunk, {len(doc_to_indices)} van ban rieng biet trong cache.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Load embedding model {args.embedding_model} (device={device})...")
    model = SentenceTransformer(args.embedding_model, device=device)
    model.max_seq_length = args.max_seq_length

    # Chi encode CAC CAU HOI thuc su co the bi injection (co it nhat 1 doc trong top-5 co hang xom
    # trong graph CHUA co trong danh sach) - tranh encode thua cho cau hoi chac chan khong doi.
    qids_to_encode = []
    for qid, predicted_ids in preds.items():
        for anchor in predicted_ids:
            neighbors = graph.get(anchor, [])
            if any(n not in predicted_ids for n in neighbors):
                qids_to_encode.append(qid)
                break

    print(f"So cau hoi can encode (co kha nang bi injection): {len(qids_to_encode)}/{len(preds)}")
    texts = [args.query_prefix + question_texts.get(qid, "") for qid in qids_to_encode]
    query_emb = model.encode(
        texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)
    qid_to_vec = {qid: query_emb[i] for i, qid in enumerate(qids_to_encode)}

    def doc_score(vec, doc_id: str) -> float:
        idxs = doc_to_indices.get(doc_id)
        if not idxs:
            return -1.0  # khong co trong cache/corpus - coi nhu diem thap nhat, khong the cham
        return float(corpus_emb[idxs].dot(vec).max())

    n_questions_modified = 0
    n_total_injected = 0
    n_candidates_rejected = 0
    output = {}
    for qid, predicted_ids in preds.items():
        result = list(predicted_ids)
        n_injected = 0
        vec = qid_to_vec.get(qid)
        if vec is not None and result:
            for anchor in predicted_ids:
                if n_injected >= args.max_inject:
                    break
                for neighbor in graph.get(anchor, []):
                    if neighbor in result:
                        continue
                    if len(result) >= 5:
                        rank5_doc = result[-1]
                        rank5_score = doc_score(vec, rank5_doc)
                        cand_score = doc_score(vec, neighbor)
                        if cand_score <= rank5_score + args.margin:
                            n_candidates_rejected += 1
                            continue  # ung vien KEM hon hang 5 hien tai theo diem dense - KHONG bom
                        result[-1] = neighbor
                    else:
                        result.append(neighbor)
                    n_injected += 1
                    break
        if n_injected > 0:
            n_questions_modified += 1
            n_total_injected += n_injected
        output[qid] = {"answer": result}

    print(f"\nTong so cau hoi: {len(preds)}")
    print(f"So cau hoi bi sua doi (SAU KHI loc bang diem dense): {n_questions_modified} ({n_questions_modified/len(preds)*100:.1f}%)")
    print(f"Tong so tai lieu da bom: {n_total_injected}")
    print(f"So ung vien bi TU CHOI vi diem dense KEM hon hang 5 hien tai: {n_candidates_rejected}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu predictions MOI (sau injection + loc) tai: {args.output}")
    print(
        "\nBuoc tiep theo - do lai bang evaluate.py, so sanh CA 3 ban (goc / injection tho khong loc "
        "/ injection da loc):\n"
        f"  python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions {args.predictions}\n"
        f"  python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/citation_graph_injected.json\n"
        f"  python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions {args.output}\n"
        "Gui lai CA 3 ket qua (dac biet dong 'Recall trung binh') de so sanh."
    )


if __name__ == "__main__":
    main()
