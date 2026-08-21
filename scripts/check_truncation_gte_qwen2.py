"""
Kiểm tra NHANH (chỉ tokenize, KHÔNG cần GPU, không cần load model embedding) xem corpus chunk có bị
cắt cụt (truncate) nhiều hay không khi dùng --max-seq-length 512 với tokenizer của
Alibaba-NLP/gte-Qwen2-1.5B-instruct.

LÝ DO CẦN CHECK NÀY: lần chạy thực tế cuối cùng (2026-08-16 rạng sáng) dùng --max-seq-length 512
(không phải 1024 như log trước đó ghi nhầm) - THẤP HƠN NHIỀU so với --max-seq-length 2048 dùng cho
AITeamVN. Nếu nhiều chunk pháp luật dài hơn 512 token, model gte-Qwen2 sẽ bị CẮT MẤT thông tin trước
khi encode - kết quả Recall LOCAL 0.8864 thấp hơn AITeamVN CÓ THỂ một phần (hoặc toàn bộ) do bị cắt
cụt, KHÔNG PHẢI do model gte-Qwen2 vốn dĩ yếu hơn trên domain pháp luật. Cần biết tỉ lệ chunk bị ảnh
hưởng trước khi tin chắc vào kết luận "đóng hướng thử nghiệm này".

Cách chạy (từ thư mục gốc UIT-data-science-2026, chỉ mất khoảng 1-3 phút, CPU thôi, không cần GPU):
    python scripts/check_truncation_gte_qwen2.py --chunks data/processed/chunks_corpus.jsonl

Có thể thêm --sample 20000 để chỉ tokenize ngẫu nhiên 1 phần corpus cho nhanh hơn nữa (mặc định
tokenize TOÀN BỘ 351K chunk, vẫn nhanh vì chỉ tokenize, không chạy model).
"""
from __future__ import annotations

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    ap.add_argument("--model", default="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    ap.add_argument("--sample", type=int, default=None, help="Chỉ tokenize ngẫu nhiên N chunk thay vì toàn bộ (nhanh hơn)")
    ap.add_argument("--thresholds", type=int, nargs="+", default=[512, 1024, 2048])
    args = ap.parse_args()

    print(f"Đang đọc chunk từ {args.chunks} ...")
    passages = []
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            passages.append(obj["passage"])
    print(f"Đã đọc {len(passages):,} chunk.")

    if args.sample and args.sample < len(passages):
        random.seed(42)
        passages = random.sample(passages, args.sample)
        print(f"Đã lấy mẫu ngẫu nhiên {len(passages):,} chunk (--sample {args.sample}).")

    print(f"\nĐang load tokenizer: {args.model} (chỉ tokenizer, KHÔNG load model - rất nhanh) ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Đang tokenize {len(passages):,} chunk để đếm độ dài token thật (không cắt) ...")
    lengths = []
    batch_size = 256
    for i in range(0, len(passages), batch_size):
        batch = passages[i : i + batch_size]
        enc = tokenizer(batch, truncation=False, add_special_tokens=True)
        lengths.extend(len(ids) for ids in enc["input_ids"])
        if (i // batch_size) % 50 == 0:
            print(f"  ... {i + len(batch):,}/{len(passages):,}", end="\r")

    lengths.sort()
    n = len(lengths)
    print(f"\n\n=== KẾT QUẢ ĐỘ DÀI TOKEN (tokenizer của {args.model}) ===")
    print(f"Số chunk đã đo   : {n:,}")
    print(f"Ngắn nhất        : {lengths[0]}")
    print(f"Trung vị (p50)   : {lengths[n // 2]}")
    print(f"p90              : {lengths[int(n * 0.90)]}")
    print(f"p99              : {lengths[int(n * 0.99)]}")
    print(f"Dài nhất         : {lengths[-1]}")

    print(f"\n=== TỈ LỆ CHUNK BỊ CẮT CỤT (vượt ngưỡng --max-seq-length) ===")
    for t in sorted(args.thresholds):
        n_over = sum(1 for x in lengths if x > t)
        pct = 100.0 * n_over / n
        print(f"  > {t:5d} token: {n_over:7,} / {n:,} chunk ({pct:.2f}%)")

    print(
        "\nDIỄN GIẢI:\n"
        "- Nếu tỉ lệ > 512 RẤT NHỎ (vd < 1-2%) -> truncation ở --max-seq-length 512 hầu như KHÔNG\n"
        "  ảnh hưởng, kết luận 'gte-Qwen2 thấp hơn AITeamVN' vẫn đáng tin.\n"
        "- Nếu tỉ lệ > 512 ĐÁNG KỂ (vd > 10-20%) -> kết quả Recall 0.8864 CÓ THỂ bị hạ thấp một phần\n"
        "  do mất thông tin khi cắt cụt, KHÔNG PHẢN ÁNH ĐÚNG sức mạnh thật của model - cần cân nhắc\n"
        "  chạy lại với --max-seq-length cao hơn (chấp nhận chậm hơn) trước khi kết luận chắc chắn,\n"
        "  hoặc chấp nhận bỏ qua vì đã có 2 lý do khác (thời gian chạy quá lâu, benchmark tổng quát\n"
        "  không phản ánh domain) khiến hướng này vẫn không đáng ưu tiên."
    )


if __name__ == "__main__":
    main()
