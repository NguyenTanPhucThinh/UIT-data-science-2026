# Kaggle GPU Pipeline — UIT Data Science 2026 Task 1

## Mục đích
Thư mục này chứa Kaggle notebooks để **encode corpus trên GPU miễn phí của Kaggle**,
rồi download cache `.npz` về local để chạy `fuse` sweep (không cần GPU).

## Workflow tổng thể

```
[Local]                        [Kaggle]                    [Local]
Upload dataset          →      Encode corpus         →     Download .npz
val_preprocessed.jsonl         (T4 GPU, ~1-2h)            Chạy fuse sweep
chunks_corpus.jsonl            Save ranks_*.npz            (CPU, vài phút)
```

## Bước 1 — Upload data lên Kaggle

1. Vào https://www.kaggle.com/datasets → **"New Dataset"**
2. Tên dataset: `uit-legalir-corpus` (hoặc tên tùy ý)
3. Upload **2 file** sau từ repo local:
   - `data/processed/chunks_corpus.jsonl` (~1.1 GB)
   - `data/processed/val_preprocessed.jsonl` (~490 KB)
4. Đặt visibility: **Private**
5. Nhấn **Create** → chờ upload xong

## Bước 2 — Chạy notebook trên Kaggle

1. Vào https://www.kaggle.com/code → **"New Notebook"**
2. **Settings** (góc phải):
   - Accelerator: **GPU T4 x2** (hoặc P100)
   - Internet: **On** (để download model từ HuggingFace)
3. **Add Data** → tìm dataset `uit-legalir-corpus` vừa upload
4. Copy nội dung file `notebooks/encode_corpus_qwen3.ipynb` vào notebook
5. **Run All** → đợi ~1-2 giờ

## Bước 3 — Download kết quả về local

Sau khi notebook chạy xong:
1. Tab **Output** trong notebook → download file `ranks_qwen3_emb.npz`
2. Đặt vào `outputs/cache/ranks_qwen3_emb.npz` trong repo local

## Bước 4 — Chạy fuse sweep (local, không cần GPU)

```powershell
# Đảm bảo đã có ranks_v1.npz, ranks_harrier.npz, ranks_bm25.npz
# Nếu chưa có ranks_bm25.npz, chạy trước:
python -X utf8 -m src.evaluation.ensemble_fusion rank-bm25 `
    --name bm25 --out outputs/cache/ranks_bm25.npz

# Sweep ensemble thêm nhánh Qwen3-Embedding-0.6B
python -X utf8 -m src.evaluation.ensemble_fusion fuse `
    --rank v1=outputs/cache/ranks_v1.npz `
    --rank harrier=outputs/cache/ranks_harrier.npz `
    --rank bm25=outputs/cache/ranks_bm25.npz `
    --rank qwen3emb=outputs/cache/ranks_qwen3_emb.npz `
    --weights harrier=0.2 `
    --weights bm25=0.1 `
    --weights qwen3emb=0,0.2,0.4,0.6,0.8,1.0,1.2 `
    --rrf-k 60,30,10 `
    --report outputs/predictions/ensemble_add_qwen3emb_report.json
```

## Đọc kết quả

- Dòng `qwen3emb=0` PHẢI ra đúng **0.9319** (baseline val local hiện tại).
- Nếu không ra 0.9319 → có lỗi gì đó, KHÔNG tin kết quả.
- Nếu có dòng nào > 0.9319 + 0.01 (≥ 0.9419) → đáng nộp public test.

## Lưu ý quan trọng

- **Kaggle session timeout 9 giờ** — notebook nếu chạy >9h sẽ bị dừng (encode corpus ~1-2h, ổn).
- **Kaggle output limit 20GB** — file ranks_*.npz chỉ ~1-2MB, không vấn đề.
- **HuggingFace token** — Qwen/Qwen3-Embedding-0.6B là model công khai, KHÔNG cần token.
  Nếu thử CATI-AI khi access được duyệt: thêm HF token vào Kaggle Secrets.
