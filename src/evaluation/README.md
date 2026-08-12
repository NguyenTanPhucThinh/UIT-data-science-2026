# src/evaluation

Module đánh giá chất lượng hệ thống Task 1 - LegalIR (phụ trách: Nguyễn Huy Hoàng, Bùi Diên Gia Bảo).

## Các file

- `metrics.py` — cài đặt thuần Python (không phụ thuộc thư viện ML nào) của Recall@5 và
  Precision@5, đúng công thức và ràng buộc chính thức của BTC (xem `docs/DSC2026_Task1_LegalIR_Data_Overview.docx`, mục 4).
- `evaluate.py` — CLI: so sánh một file dự đoán (định dạng `submission.json`) với ground truth
  (`data/raw/train.json`), in ra Recall/Precision trung bình + danh sách câu hỏi tệ nhất để debug.
- `validate_submission.py` — CLI: kiểm tra file `submission.json` có đúng định dạng/ràng buộc
  BTC yêu cầu TRƯỚC KHI đóng gói nộp bài (tránh bị hệ thống từ chối chấm).
- `baseline_retrieval.py` — CLI retrieval "brute-force" (chỉ embedding + cosine similarity trên
  toàn corpus, KHÔNG BM25/hybrid/rerank) để đo Recall/Precision MỐC của một embedding model một
  cách ĐỘC LẬP với tiến độ pipeline production `src/retrieval`. Có cache embedding corpus (.npz)
  vì corpus không đổi giữa các lần chạy. KHÔNG phải pipeline cuối cùng của nhóm.
- `bm25_baseline.py` — CLI retrieval BM25-only (sparse) trên `data/processed/chunks_corpus.jsonl`
  (Phase 4 của Kiên - Data & Utilities), chấm theo ĐÚNG công thức Recall@5/Precision@5 chính
  thức BTC (dùng chung `metrics.py` với các script khác, so sánh 1-1 được với dense/hybrid).
- `full_pipeline_eval.py` — CLI pipeline hybrid ĐẦY ĐỦ (dense + BM25 + RRF fusion + rerank cross-
  encoder), chạy hoàn toàn LOCAL, KHÔNG cần Qdrant Cloud (Storage & Representation) hay code
  retrieval thật của Core RAG (`src/retrieval`). Dùng đúng `data/processed/chunks_corpus.jsonl`
  của Kiên (không tự chunk lại). Có thể dùng ngay để tạo submission chất lượng cao trong lúc chờ
  2 nhóm kia hoàn thiện hạ tầng production — xem docstring đầu file để biết cách dùng đầy đủ.

## Cách dùng

Đánh giá pipeline của nhóm trên tập train (dùng làm dev set nội bộ):

```bash
python -m src.evaluation.evaluate \
    --ground-truth data/raw/train.json \
    --predictions outputs/my_predictions.json \
    --report outputs/eval_report.json
```

Kiểm tra định dạng trước khi nộp bài thật:

```bash
python -m src.evaluation.validate_submission \
    --predictions outputs/submission.json \
    --corpus data/raw/selected-contexts.zip \
    --questions data/raw/public-official.json
```

File dự đoán (`--predictions`) LUÔN phải đúng định dạng `submission.json` chính thức:

```json
{
  "86666": { "answer": ["280282"] }
}
```

Đo baseline retrieval độc lập (không cần chờ `src/retrieval` xong):

```bash
python -m src.evaluation.baseline_retrieval \
    --model bqbbao6/VN_legal_embedding_512 \
    --corpus data/raw/selected-contexts.zip \
    --questions data/raw/train.json \
    --output outputs/baseline_predictions.json \
    --embedding-cache outputs/corpus_embeddings__bqbbao6.npz
```

Chạy pipeline hybrid đầy đủ (dense + BM25 + rerank), sau khi đã có `data/processed/chunks_corpus.jsonl`
(chạy `python "src/ingestion + EDA/build_pipeline.py"` nếu chưa có - cần Java cho VnCoreNLP):

```bash
python -m src.evaluation.full_pipeline_eval \
    --embedding-model models/vietnamese-legal-embedding-ft \
    --reranker-model models/reranker-ft \
    --chunks data/processed/chunks_corpus.jsonl \
    --questions data/processed/val_preprocessed.jsonl \
    --output outputs/full_pipeline_predictions.json \
    --embedding-cache outputs/chunks_corpus_embeddings__ft.npz

python -m src.evaluation.evaluate \
    --ground-truth data/processed/val_split_20.json \
    --predictions outputs/full_pipeline_predictions.json
```

## Lưu ý cho các thành viên khác

- Nếu bạn phát triển `src/retrieval`, sau khi chạy pipeline trên `data/raw/train.json`, hãy xuất
  kết quả ra đúng định dạng trên rồi chạy `evaluate.py` để biết Recall/Precision thực tế trước
  khi nộp lên hệ thống chính thức (chỉ được submit tối đa 3 lần/ngày ở vòng private test).
- `metrics.py` không phụ thuộc torch/sentence-transformers nên có thể import/dùng độc lập ở bất
  kỳ đâu trong pipeline (kể cả trong notebook EDA) mà không cần cài thêm gì.
