"""
Do "drift" cua embedding TRUOC/SAU fine-tune: encode CUNG 1 tap cau hoi + passage bang model
PRETRAINED va model DA FINE-TUNE, roi so sanh cosine similarity giua 2 vector embedding cua
CUNG 1 van ban. Neu fine-tune chi "tinh chinh nhe" thi cosine se rat gan 1.0 (khong doi nhieu).
Neu fine-tune lam "xao tron" ca khong gian bieu dien (dau hieu catastrophic forgetting) thi
cosine se thap han han.

Khong can nhan/ground truth - chay duoc NGAY, KHONG can fine-tune lai gi them.

Cach chay (tu thu muc GOC UIT-data-science-2026):
    python scripts/diagnose_embedding_drift.py ^
        --finetuned-model models/vietnamese-embedding-aiteamvn-ft-full ^
        --n-questions 200 --n-passages 300 ^
        --out outputs/predictions/embedding_drift_report.csv

Doc ket qua:
    - "Trung binh" cang GAN 1.0 nghia la fine-tune thay doi it, cang GAN 0 (hoac am) nghia la
      thay doi nhieu.
    - So van ban co cosine < 0.5 la muc rat dang chu y (gan nhu la 1 embedding khac han so voi
      truoc khi fine-tune).
    - File CSV chi tiet tung dong (id, cosine, text preview) de tu mo xem van ban nao bi drift
      manh nhat - kiem tra xem co lien quan gi den domain phap luat cua train.json khong.

Goi y doc ket qua sau khi chay xong:
    - Neu cosine trung binh RAT CAO (vd > 0.95) ma Score PUBLIC van giam manh -> fine-tune KHONG
      "pha" representation nhieu ve mat con so trung binh -> nhieu kha nang la do OVERFIT/memorize
      hep tren train.json (hoc thuoc long cap cau hoi - tai lieu cu the) chu khong phai
      catastrophic forgetting kieu "quen sach" kien thuc chung.
    - Neu cosine trung binh THAP RO RET (vd < 0.7-0.8) hoac ty le van ban cosine < 0.5 cao ->
      fine-tune da lam xao tron manh khong gian bieu dien goc -> ung ho gia thuyet catastrophic
      forgetting that su.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.full_pipeline_eval import load_chunks, load_questions  # noqa: E402


def encode_texts(model, texts, prefix="", batch_size=64):
    full_texts = [prefix + t for t in texts]
    return model.encode(
        full_texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)


def cosine_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a, b da normalize (norm=1) nen cosine = dot product theo tung hang
    return np.sum(a * b, axis=1)


def summarize(name: str, sims: np.ndarray):
    print(f"\n=== {name} (n={len(sims)}) ===")
    print(f"  Trung binh : {sims.mean():.4f}")
    print(f"  Min        : {sims.min():.4f}")
    print(f"  P10        : {np.percentile(sims, 10):.4f}")
    print(f"  Median     : {np.percentile(sims, 50):.4f}")
    print(f"  P90        : {np.percentile(sims, 90):.4f}")
    print(f"  Max        : {sims.max():.4f}")
    n_low = int((sims < 0.5).sum())
    print(f"  So van ban co cosine < 0.5 (drift RAT MANH): {n_low}/{len(sims)} ({100 * n_low / len(sims):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-model", default="AITeamVN/Vietnamese_Embedding")
    ap.add_argument("--finetuned-model", required=True)
    ap.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    ap.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--n-passages", type=int, default=300)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--passage-prefix", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/predictions/embedding_drift_report.csv")
    args = ap.parse_args()

    random.seed(args.seed)

    from sentence_transformers import SentenceTransformer

    print(f"Dang load model PRETRAINED: {args.pretrained_model}")
    model_pre = SentenceTransformer(args.pretrained_model)
    model_pre.max_seq_length = args.max_seq_length

    print(f"Dang load model FINE-TUNED: {args.finetuned_model}")
    model_ft = SentenceTransformer(args.finetuned_model)
    model_ft.max_seq_length = args.max_seq_length

    chunks = load_chunks(args.chunks)
    questions = load_questions(args.questions)

    random.shuffle(chunks)
    random.shuffle(questions)
    chunks = chunks[: args.n_passages]
    questions = questions[: args.n_questions]

    q_texts = [q["question"] for q in questions]
    p_texts = [c["passage"] for c in chunks]

    print("\nEncoding CAU HOI...")
    q_emb_pre = encode_texts(model_pre, q_texts, prefix=args.query_prefix)
    q_emb_ft = encode_texts(model_ft, q_texts, prefix=args.query_prefix)

    print("\nEncoding PASSAGE...")
    p_emb_pre = encode_texts(model_pre, p_texts, prefix=args.passage_prefix)
    p_emb_ft = encode_texts(model_ft, p_texts, prefix=args.passage_prefix)

    q_sims = cosine_per_row(q_emb_pre, q_emb_ft)
    p_sims = cosine_per_row(p_emb_pre, p_emb_ft)

    summarize("CAU HOI (question)", q_sims)
    summarize("PASSAGE (chunk)", p_sims)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "id", "cosine_pretrained_vs_finetuned", "text_preview"])
        for q, sim in zip(questions, q_sims):
            writer.writerow(["question", q["question_id"], f"{sim:.4f}", q["question"][:120]])
        for c, sim in zip(chunks, p_sims):
            writer.writerow(["passage", c["doc_id"], f"{sim:.4f}", c["passage"][:120]])
    print(f"\nDa luu chi tiet tung dong vao {out_path}")


if __name__ == "__main__":
    main()
