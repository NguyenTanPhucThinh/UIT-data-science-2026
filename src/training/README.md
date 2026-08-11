# src/training

Module fine-tune model cho Task 1 - LegalIR (phụ trách: Nguyễn Huy Hoàng, Bùi Diên Gia Bảo).

## Các file

- `data_utils.py` — tiện ích dùng chung: `CorpusReader` (đọc lười từng văn bản trực tiếp từ
  `selected-contexts.zip`, không cần giải nén), `load_labeled_questions` (đọc `train.json`),
  `chunk_passage` (chunk theo cấu trúc Điều/Khoản - bản baseline, nên thay bằng chunker chính
  thức của `src/ingestion + EDA` khi module đó hoàn thiện để đồng bộ với retrieval thật),
  `select_best_chunk` (chọn đoạn khả năng cao nhất chứa câu trả lời trong 1 document, dùng
  word-overlap heuristic vì BTC chỉ gán nhãn ở cấp document chứ không chỉ rõ đoạn/Điều).
- `finetune_embedding.py` — CLI fine-tune embedding model bằng contrastive learning
  (`MultipleNegativesRankingLoss`), có tuỳ chọn `--hard-negatives` để mine hard negative và
  chuyển sang `TripletLoss` (thường cho kết quả tốt hơn nhưng chậm hơn vì phải encode thêm để mine).
- `finetune_reranker.py` — CLI fine-tune cross-encoder reranker (điểm liên quan 0-1 cho cặp
  câu hỏi/đoạn văn), tái sử dụng `build_positive_pairs` từ `finetune_embedding.py`; negative
  mặc định lấy ngẫu nhiên trong corpus, hoặc `--hard-negatives` để mine bằng một embedding model
  riêng (khuyến nghị dùng khi đã có baseline reranker chạy ổn — negative ngẫu nhiên thường quá dễ).

## Cách dùng

Cài dependency (xem `requirements.txt` ở gốc repo):

```bash
pip install sentence-transformers torch
```

Fine-tune baseline (in-batch negatives, nhanh, nên chạy trước để có baseline so sánh):

```bash
python -m src.training.finetune_embedding \
    --base-model bqbbao6/vietnamese-legal-embedding \
    --train-json data/raw/train.json \
    --corpus data/raw/selected-contexts.zip \
    --output-dir models/vietnamese-legal-embedding-ft \
    --epochs 3 --batch-size 16
```

Fine-tune với hard negative mining (thử sau khi có baseline, để xem có cải thiện Recall không):

```bash
python -m src.training.finetune_embedding \
    --base-model bqbbao6/vietnamese-legal-embedding \
    --train-json data/raw/train.json \
    --corpus data/raw/selected-contexts.zip \
    --output-dir models/vietnamese-legal-embedding-ft-hardneg \
    --epochs 3 --batch-size 16 --hard-negatives
```

Fine-tune reranker (sau khi đã có embedding baseline, dùng chính embedding model đó để mine hard negative):

```bash
python -m src.training.finetune_reranker \
    --base-model BAAI/bge-reranker-v2-m3 \
    --train-json data/raw/train.json \
    --corpus data/raw/selected-contexts.zip \
    --output-dir models/reranker-ft \
    --epochs 2 --batch-size 16 \
    --hard-negatives --hard-negative-embedding-model models/vietnamese-legal-embedding-ft
```

Sau khi fine-tune xong, chạy lại retrieval pipeline (`src/retrieval`) với model mới này — hoặc
dùng `src/evaluation/baseline_retrieval.py` để đo nhanh embedding một mình trong lúc chờ
`src/retrieval` hoàn thiện — rồi dùng `src/evaluation/evaluate.py` để so sánh Recall/Precision
với model gốc (pretrained). Đây là cách duy nhất biết chắc việc fine-tune có thực sự cải thiện
hay không. Đừng quên điền kết quả vào Model Card tương ứng (`docs/model-card-template.md`).

## Quan trọng — tuân thủ quy định cuộc thi

- Chỉ dùng `data/raw/train.json` + `data/raw/selected-contexts.zip` (dữ liệu chính thức BTC cấp)
  để tạo cặp training. KHÔNG thêm dữ liệu ngoài, KHÔNG tự gán nhãn thủ công thêm.
- Model base tải về từ HuggingFace Hub rồi chạy hoàn toàn local — hợp lệ (đây là "tải về, vận
  hành và kiểm soát trực tiếp", không phải gọi API dịch vụ ngoài).
- Fine-tune KHÔNG làm thay đổi số tham số của model gốc. Sau khi chọn model cuối cùng, hãy tính
  lại tổng tham số của TOÀN BỘ pipeline (embedding + reranker + LLM nếu có) để chắc chắn vẫn
  dưới 4 tỷ theo quy định.
- Chạy tốt trên GPU RTX 3060 (6-12GB VRAM) — model embedding cỡ vài trăm triệu tham số không đòi
  hỏi VRAM lớn, kể cả khi fine-tune.
