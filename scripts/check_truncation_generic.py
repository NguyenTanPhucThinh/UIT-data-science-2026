"""
Kiem tra ti le chunk trong corpus BI CAT CUT (vuot qua max_seq_length) cho 1 model embedding BAT KY -
CHI tokenize (khong can GPU, khong load model day du), chay nhanh (~1-2 phut cho 351K chunk) - LUON
chay script nay TRUOC khi cam ket encode toan bo corpus voi model MOI, dac biet neu max_seq_length cua
model do THAP (vd 512) - rut kinh nghiem tu vu gte-Qwen2 (21.63% chunk bi cat cut o max-seq-length 512,
lam ket qua so sanh khong con sach, phai lam lai/danh gia lai).

Cach dung (tu thu muc goc UIT-data-science-2026):
    python scripts/check_truncation_generic.py --model mainguyen9/vietlegal-harrier-0.6b --max-seq-length 512
    python scripts/check_truncation_generic.py --model AITeamVN/Vietnamese_Embedding --max-seq-length 2048
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Ten model tren HuggingFace (chi can tokenizer, khong can GPU)")
    parser.add_argument("--max-seq-length", type=int, required=True, help="max_seq_length THUC TE se dung khi encode")
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--field", default="passage", help="Ten field text trong chunks_corpus.jsonl can kiem tra do dai")
    parser.add_argument("--sample", type=int, default=0, help="Neu >0, chi kiem tra ngau nhien N chunk dau (test nhanh); 0 = kiem tra HET")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Thieu transformers. Chay: pip install transformers", file=sys.stderr)
        sys.exit(1)

    print(f"Load tokenizer cua {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    texts = []
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            texts.append(d.get(args.field, ""))
            if args.sample and len(texts) >= args.sample:
                break

    print(f"Tokenize {len(texts)} chunk (field='{args.field}')...")
    lengths = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=False, add_special_tokens=True)
        lengths.extend(len(ids) for ids in enc["input_ids"])
        if (i // batch_size) % 50 == 0:
            print(f"  {i}/{len(texts)}...")

    lengths.sort()
    n = len(lengths)

    def pct(p: float) -> int:
        idx = min(n - 1, int(n * p))
        return lengths[idx]

    n_over = sum(1 for l in lengths if l > args.max_seq_length)

    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"max_seq_length dang xet: {args.max_seq_length}")
    print(f"Tong so chunk: {n}")
    print(f"Trung vi (p50)   : {pct(0.50)} token       p90: {pct(0.90)}       p99: {pct(0.99)}       Dai nhat: {lengths[-1]}")
    print(f"> {args.max_seq_length} token : {n_over:,} / {n:,} chunk ({n_over/n*100:.2f}%)")
    print("=" * 70)
    if n_over / n > 0.10:
        print("⚠️  CANH BAO: >10% chunk bi cat cut - RUI RO CONFOUND DANG KE (giong truong hop gte-Qwen2).")
        print("    Nen can nhac: (a) tang max_seq_length neu model ho tro (kiem tra model card), hoac")
        print("    (b) chap nhan ket qua co the KHONG SACH, ghi ro trong log khi bao cao.")
    elif n_over / n > 0.02:
        print("⚠️  Mot ti le nho (2-10%) chunk bi cat cut - CO THE anh huong nhe, nen ghi chu khi bao cao.")
    else:
        print("✅ Ti le cat cut RAT THAP (<2%) - gan nhu sach, khong dang lo.")


if __name__ == "__main__":
    main()
