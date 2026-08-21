"""
Ap dung citation-graph injection THAT vao 1 file predictions da co san (KHONG can chay lai
full_pipeline_eval.py, khong can GPU, khong can encode lai corpus) - roi xuat ra file predictions moi
de do lai bang evaluate.py va so sanh truoc/sau.

Khac voi check_citation_graph_coverage.py (chi do "tran ceiling" GIA DINH bom la mien phi) - script
nay mo phong DUNG THUC TE se lam khi tich hop that: neu top-5 da day, phai THAY THE 1 tai lieu dang co
(hang thap nhat) bang tai lieu lien ket qua graph - nen co the LAM MAT 1 tai lieu dung khac dang co
san. Do Recall SAU khi ap dung se phan anh dung net effect that, khong phai uoc luong ly tuong nua.

Chinh sach injection (bao thu, de thu nghiem dau tien):
- Voi moi cau hoi, duyet cac tai lieu du doan tu hang cao xuong thap (rank 1 -> 5)
- Voi moi tai lieu, tra graph xem co "hang xom" (van ban lien ket) nao CHUA co trong danh sach khong
- Neu co, BOM vao: neu danh sach chua du 5 thi them vao cuoi, neu da du 5 thi THAY THE vi tri hang
  THAP NHAT (rank 5, it tin cay nhat) bang tai lieu lien ket do
- Toi da bom 1 tai lieu/cau hoi (--max-inject de doi neu muon thu nhieu hon)

Cach dung (tu thu muc goc UIT-data-science-2026, SAU KHI da co outputs/citation_graph.json):
    python scripts/apply_citation_graph_injection.py

Sau do do lai bang lenh co san (giong moi lan do local khac trong du an):
    python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/sweep_bm25_weight/bm25_weight_0.1.json
    python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/citation_graph_injected.json
So sanh 2 ket qua Recall - CHI submit public neu ban co tang ro ret (theo dung nguyen tac da dung
xuyen suot du an, xem submission-log-task1.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def inject(predicted_ids: list, graph: dict, max_inject: int) -> tuple:
    """Tra ve (danh_sach_moi, so_luong_da_bom). Chinh sach: duyet anchor tu hang cao -> thap, thay
    the hang THAP NHAT bang hang xom dau tien tim duoc chua co trong danh sach."""
    if not predicted_ids:
        return predicted_ids, 0
    result = list(predicted_ids)
    n_injected = 0
    for anchor in predicted_ids:  # duyet theo thu tu hang goc (rank 1 truoc)
        if n_injected >= max_inject:
            break
        for neighbor in graph.get(anchor, []):
            if neighbor in result:
                continue
            if len(result) >= 5:
                result[-1] = neighbor  # thay the hang THAP NHAT (it tin cay nhat)
            else:
                result.append(neighbor)
            n_injected += 1
            break
    return result, n_injected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", default="outputs/predictions/sweep_bm25_weight/bm25_weight_0.1.json")
    parser.add_argument("--graph", default="outputs/citation_graph.json")
    parser.add_argument("--output", default="outputs/predictions/citation_graph_injected.json")
    parser.add_argument("--max-inject", type=int, default=1, help="So tai lieu toi da duoc bom them/cau hoi")
    args = parser.parse_args()

    preds = load_predictions(args.predictions)
    graph = load_graph(args.graph)

    n_questions_modified = 0
    n_total_injected = 0
    output = {}
    for qid, predicted_ids in preds.items():
        new_ids, n_inj = inject(predicted_ids, graph, args.max_inject)
        if n_inj > 0:
            n_questions_modified += 1
            n_total_injected += n_inj
        output[qid] = {"answer": new_ids}

    print(f"Tong so cau hoi: {len(preds)}")
    print(f"So cau hoi bi sua doi boi injection: {n_questions_modified} ({n_questions_modified/len(preds)*100:.1f}%)")
    print(f"Tong so tai lieu da bom (thay the hoac them moi): {n_total_injected}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu predictions MOI (sau injection) tai: {args.output}")
    print(
        "\nBuoc tiep theo - do va so sanh Recall TRUOC/SAU (dung lenh evaluate.py co san):\n"
        f"  python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json "
        f"--predictions {args.predictions}\n"
        f"  python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json "
        f"--predictions {args.output}\n"
        "Gui lai CA 2 ket qua (dac biet dong 'Recall trung binh' va bang breakdown theo n_relevant) "
        "de so sanh va quyet dinh co submit public khong."
    )


if __name__ == "__main__":
    main()
