"""
Kiem tra NHANH (vai chuc giay) 1 model embedding co load va encode duoc dung khong, TRUOC khi
cam ket chay encode toan bo corpus (co the mat nhieu tieng). Dung cho model moi/it quen thuoc
(vd can --trust-remote-code, quy uoc prefix rieng) de bat loi tich hop SOM, tranh lang phi ca
dem chay hong giua chung vi 1 loi don gian (thieu trust_remote_code, sai ten prefix...).

Cach chay (tu thu muc GOC UIT-data-science-2026):
    python scripts/sanity_check_embedding_model.py --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --trust-remote-code --max-seq-length 2048 --query-instruct "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question."
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trust-remote-code", action="store_true", default=False)
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--passage-prefix", default="")
    ap.add_argument(
        "--query-instruct", default=None,
        help='Giống --query-instruct của full_pipeline_eval.py - script tự ghép thành '
             '"Instruct: {task}\\nQuery: " (ghi đè --query-prefix nếu cả 2 cùng truyền).',
    )
    args = ap.parse_args()

    query_prefix = args.query_prefix
    if args.query_instruct:
        query_prefix = f"Instruct: {args.query_instruct}\nQuery: "

    from sentence_transformers import SentenceTransformer

    print(f"Dang load model: {args.model} (trust_remote_code={args.trust_remote_code})")
    model = SentenceTransformer(args.model, trust_remote_code=args.trust_remote_code)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
        print(f"Da ghi de max_seq_length = {args.max_seq_length}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nOK - load thanh cong. So tham so: {n_params:,} (~{n_params / 1e9:.3f} ty)")
    if n_params >= 4_000_000_000:
        print("!!! CANH BAO: >= 4 ty tham so, VUOT tran quy dinh cuoc thi - KHONG duoc dung mot minh !!!")

    print(f"\nquery_prefix dang dung: {query_prefix!r}")
    print(f"passage_prefix dang dung: {args.passage_prefix!r}")

    sample_questions = [
        "Nguoi lao dong co quyen don phuong cham dut hop dong lao dong khong?",
        "Dieu kien de thanh lap doanh nghiep tu nhan la gi?",
    ]
    sample_passages = [
        "Theo quy dinh tai Dieu 35 Bo luat Lao dong, nguoi lao dong co quyen don phuong cham dut "
        "hop dong lao dong nhung phai bao truoc theo thoi han luat dinh.",
        "Doanh nghiep tu nhan la doanh nghiep do mot ca nhan lam chu va tu chiu trach nhiem bang "
        "toan bo tai san cua minh ve moi hoat dong cua doanh nghiep.",
    ]

    q_emb = model.encode([query_prefix + t for t in sample_questions], normalize_embeddings=True)
    p_emb = model.encode([args.passage_prefix + t for t in sample_passages], normalize_embeddings=True)
    print(f"\nEncode CAU HOI OK - shape {q_emb.shape}, dtype {q_emb.dtype}")
    print(f"Encode PASSAGE OK - shape {p_emb.shape}, dtype {p_emb.dtype}")

    sim = q_emb @ p_emb.T
    print("\nMa tran cosine similarity (hang=cau hoi, cot=passage) - kiem tra nhanh bang mat:")
    print(sim.round(3))
    print(
        "\nCau hoi 0 (lao dong) nen co similarity CAO HON voi passage 0 (lao dong) hon la passage 1 "
        "(doanh nghiep), va nguoc lai cho cau hoi 1. Neu dung nhu vay (khong loi, gia tri hop ly) "
        "-> model san sang, co the chay full_pipeline_eval.py encode toan bo corpus."
    )


if __name__ == "__main__":
    main()
