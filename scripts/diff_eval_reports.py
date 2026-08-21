"""
So sanh 2 report cua evaluate.py (--report ...json) de tim CHINH XAC nhung cau hoi nao bi
GIAM Recall khi doi model (vd pretrained -> fine-tuned), roi in ra cau hoi + tai lieu DUNG +
tai lieu DU DOAN cua CA 2 model de doc thu cong, tim nguyen nhan cu the (sai kieu gi: du doan
sai hoan toan / du doan gan dung nhung lech doc khac / thieu du doan...).

*** LUU Y QUAN TRONG - TRANH KET LUAN SAI ***
Neu model B la ban fine-tune tren TOAN BO train.json (bao gom ca val_split_20 - vd
models/vietnamese-embedding-aiteamvn-ft-full, ban da nop submission_v5), ket qua Recall cua no
tren val_split_20 se BI THOI PHONG (model da "nhin thay" dap an dung khi train) - KHONG dung
de so sanh cong bang voi model A (pretrained, chua tung thay data nay). Chi dung script nay de
so sanh 2 model CUNG khong bi ro ri du lieu tren val_split_20 - vi du:
    A = AITeamVN pretrained (chua fine-tune gi)
    B = AITeamVN fine-tune GIAI DOAN 1 (chi tren 80% train, val_split_20 la 20% con lai, CHUA
        tung "thay" trong luc train) - dung run_pipeline_finetune_aiteamvn_stage1.ps1 de tao.

Cach chay (tu thu muc GOC UIT-data-science-2026), vi du so sanh pretrained (co san) voi ban
fine-tune Giai doan 1 (sau khi da chay xong va co predictions + report tren val_split_20):
    python scripts/diff_eval_reports.py ^
        --report-a outputs/predictions/embcompare_aiteamvn_final_report.json ^
        --predictions-a outputs/predictions/embcompare_aiteamvn.json ^
        --report-b outputs/predictions/stage1_ft_val_eval_report.json ^
        --predictions-b outputs/predictions/stage1_ft_val.json ^
        --ground-truth data/processed/val_split_20.json ^
        --chunks data/processed/chunks_corpus.jsonl ^
        --top-n 20

(report-a/predictions-a mac dinh o vi du tren la 2 file DA CO SAN trong repo - chinh la ket qua
cua ban PRETRAINED tren val_split_20, Recall 0.9267, dung lai luon khong can chay gi them cho
model A).
"""
from __future__ import annotations

import argparse
import json


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_doc_text_map(chunks_path):
    doc_text = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            if doc_id and doc_id not in doc_text:
                doc_text[doc_id] = obj.get("passage", "")[:150]
    return doc_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-a", default="outputs/predictions/embcompare_aiteamvn_final_report.json",
                     help="Report (--report cua evaluate.py) model A, mac dinh = pretrained co san")
    ap.add_argument("--report-b", required=True, help="Report model B (vd fine-tune Giai doan 1)")
    ap.add_argument("--predictions-a", default="outputs/predictions/embcompare_aiteamvn.json")
    ap.add_argument("--predictions-b", required=True)
    ap.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    ap.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    ap.add_argument("--top-n", type=int, default=20, help="So cau hoi GIAM Recall nhieu nhat can in ra")
    ap.add_argument("--label-a", default="A (pretrained)")
    ap.add_argument("--label-b", default="B (fine-tuned)")
    args = ap.parse_args()

    report_a = load_json(args.report_a)
    report_b = load_json(args.report_b)
    preds_a = load_json(args.predictions_a)
    preds_b = load_json(args.predictions_b)
    gt = load_json(args.ground_truth)
    doc_text = build_doc_text_map(args.chunks)

    pq_a = report_a["per_question"]
    pq_b = report_b["per_question"]

    print(f"Avg Recall {args.label_a}: {report_a['avg_recall']:.4f}")
    print(f"Avg Recall {args.label_b}: {report_b['avg_recall']:.4f}")
    print(f"Chenh lech: {report_b['avg_recall'] - report_a['avg_recall']:+.4f}\n")

    diffs = []
    for qid in pq_a:
        if qid not in pq_b:
            continue
        ra = pq_a[qid]["recall"]
        rb = pq_b[qid]["recall"]
        diffs.append((qid, rb - ra, ra, rb))

    diffs.sort(key=lambda x: x[1])  # tang dan -> giam nhieu nhat len dau

    n_worse = sum(1 for d in diffs if d[1] < 0)
    n_better = sum(1 for d in diffs if d[1] > 0)
    n_same = sum(1 for d in diffs if d[1] == 0)
    print(f"So cau GIAM Recall: {n_worse} | GIU NGUYEN: {n_same} | TANG: {n_better} (tren tong {len(diffs)} cau)\n")

    print(f"=== TOP {args.top_n} cau hoi GIAM RECALL NHIEU NHAT (B so voi A) ===\n")
    shown = 0
    for qid, delta, ra, rb in diffs:
        if delta >= 0 or shown >= args.top_n:
            break
        shown += 1
        question_text = gt.get(qid, {}).get("question", "(khong tim thay trong ground-truth file)")
        relevant = [str(d) for d in (gt.get(qid, {}).get("answer") or [])]
        pred_a = (preds_a.get(qid) or {}).get("answer", [])
        pred_b = (preds_b.get(qid) or {}).get("answer", [])

        print(f"--- qid={qid} | Recall {args.label_a}={ra:.2f} -> {args.label_b}={rb:.2f} (delta={delta:+.2f}) ---")
        print(f"  Cau hoi: {question_text}")
        print(f"  Tai lieu DUNG (ground truth): {relevant}")
        for d in relevant:
            if d in doc_text:
                print(f"    [{d}] {doc_text[d]}...")
        print(f"  {args.label_a} du doan: {pred_a}")
        print(f"  {args.label_b} du doan: {pred_b}")
        for d in pred_b:
            if d not in relevant and d in doc_text:
                print(f"    [SAI, {d}] {doc_text[d]}...")
        print()

    if shown == 0:
        print("(Khong co cau nao giam Recall - B khong te hon A tren tap nay theo per_question.)")


if __name__ == "__main__":
    main()
