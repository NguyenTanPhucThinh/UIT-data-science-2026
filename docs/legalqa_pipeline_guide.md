# Hướng dẫn pipeline Task 2 — LegalQA

Tài liệu này mô tả pipeline chính thức hiện tại sau khi nối LegalIR tốt nhất của Task 1 vào
`AITeamVN/Vi-Qwen2-1.5B-RAG` để sinh submission Task 2. Điểm BTC tăng từ `0.2807` lên `0.3450`: tăng `0.0643`, tương đương khoảng `22,9%` so với baseline cũ.

## 1. Pipeline đang sử dụng

```text
public_test_2_preprocessed.jsonl
              |
              +--> AITeamVN/Vietnamese_Embedding (weight 1.0, top 100)
              +--> mainguyen9/vietlegal-harrier-0.6b (weight 0.2, top 100)
              +--> BM25 (weight 0.1, top 100)
                              |
                        RRF, rrf_k=60
                              |
                   max score chunk / document
                              |
             retrieval_context_ensemble.json
                              |
         mở rộng chunk lân cận + giới hạn token
                              |
                 AITeamVN/Vi-Qwen2-1.5B-RAG
                              |
              postprocess -> validate -> ZIP
```

Cấu hình retrieval này đạt public score Task 1 `0.9208`. Không dùng reranker: các thử nghiệm trong
`docs/submission_log_task1.md` cho thấy reranker làm giảm kết quả. Tổng model của retrieval khoảng
1,16B tham số; cộng LLM 1,5B là khoảng 2,66B.

## 2. Các module và artifact

| File/module | Input | Output | Vai trò |
|---|---|---|---|
| `src/evaluation/ensemble_fusion.py rank` | model embedding, chunks, questions | `ranks_*.npz` | Encode câu hỏi và lưu top chunk của từng dense model |
| `src/evaluation/ensemble_fusion.py rank-bm25` | chunks, questions | `ranks_task2_bm25.npz` | Sinh thứ hạng BM25 |
| `src/evaluation/ensemble_context_for_task2.py` | ba file rank, chunks, questions | `retrieval_context_ensemble.json` | Fuse RRF, gộp chunk thành document và giữ passage đại diện |
| `src/evaluation/run_legalqa.py` | retrieval context, chunks, LLM | `submission_raw.json`, checkpoint, diagnostics | Xây context theo token budget và sinh câu trả lời |
| `src/qa/context_builder.py` | evidence được xếp hạng | prompt context | Mở rộng chunk lân cận, giới hạn document/chunk/token |
| `src/qa/generator.py` | prompt messages | raw answer | Chạy LLM local, greedy decoding |
| `src/qa/postprocess.py` | raw answer | cleaned answer | Làm sạch wrapper/preamble trong từng câu trả lời |
| `src/evaluation/postprocess_legalqa_submission.py` | raw submission | final submission | Bỏ các tiền tố như “Dựa vào...” sau toàn pipeline |
| `src/evaluation/validate_legalqa_submission.py` | submission và file test gốc | exit code/báo lỗi | Kiểm tra ID, schema và answer rỗng/sai wrapper |
| `scripts/run_pipeline_task2.ps1` | toàn bộ artifact trên | JSON cuối và ZIP | Wrapper end-to-end khuyến nghị |

### Contract retrieval context

Mỗi câu hỏi trong `retrieval_context_ensemble.json` có dạng:

```json
{
  "18781": {
    "question": "Nội dung câu hỏi",
    "document_ids": ["123", "456"],
    "evidence": [
      {
        "doc_id": "123",
        "corpus_chunk_index": 9123,
        "chunk_id": "123_chunk_004",
        "document_chunk_index": 4,
        "legal_section": "Điều 10",
        "document_name": "Tên văn bản",
        "passage": "Nội dung bằng chứng",
        "retrieval_rank": 1,
        "rrf_score": 0.018,
        "reranker_score": null
      }
    ]
  }
}
```

File rank mới chứa cả hash nội dung câu hỏi. Module context sẽ dừng nếu dùng rank sinh từ phiên bản
câu hỏi cũ, kể cả khi `question_id` không thay đổi.

## 3. Yêu cầu môi trường

- Windows PowerShell, chạy từ thư mục gốc repo.
- Python 3.10–3.12.
- Java 8+ nếu cần chạy lại VnCoreNLP.
- GPU NVIDIA được khuyến nghị; pipeline mặc định dùng `cuda`.
- Nên còn ít nhất khoảng 10 GB dung lượng trống cho model và hai corpus embedding cache.

Tạo và kích hoạt môi trường:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài PyTorch CUDA phù hợp trước, theo lệnh do trang PyTorch cung cấp. Sau đó:

```powershell
python -m pip install -r requirements.txt
```

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Kết quả cần có `True` và tên GPU. Nếu `False`, không nên chạy encode corpus hoặc generation full.

## 4. Chuẩn bị dữ liệu đúng Task 2

Các file raw chuẩn:

```text
data/raw/train-2.json
data/raw/public-official-2.json
data/raw/selected-contexts/context_*.json
```

Chạy:

```powershell
python "src/ingestion + EDA/build_pipeline.py" --task 2
```

Các artifact bắt buộc sau bước này:

```text
data/processed/chunks_corpus.jsonl
data/processed/public_test_2_preprocessed.jsonl
data/processed/train_2_preprocessed.jsonl
```

Nếu cả dữ liệu Task 1 và Task 2 cùng thay đổi, dùng:

```powershell
python "src/ingestion + EDA/build_pipeline.py" --task all
```

Pipeline ghi đè có kiểm soát các file `data/processed`; không cần xóa chúng trước khi chạy lại.

## 5. Tải và khóa model

### Cách đơn giản

Không cần tải thủ công. Lần chạy online đầu tiên, `sentence-transformers`/`transformers` sẽ tải:

- `AITeamVN/Vietnamese_Embedding`
- `mainguyen9/vietlegal-harrier-0.6b`
- `AITeamVN/Vi-Qwen2-1.5B-RAG`

Không truyền `-LocalFilesOnly` trong lần đầu.

### Cách khóa model để tái lập/offline

Lấy commit SHA hiện tại:

```powershell
python -c "from huggingface_hub import HfApi; api=HfApi(); [print(x, api.model_info(x).sha) for x in ['AITeamVN/Vietnamese_Embedding','mainguyen9/vietlegal-harrier-0.6b','AITeamVN/Vi-Qwen2-1.5B-RAG']]"
```

Thay `<SHA_...>` bằng SHA vừa nhận rồi tải snapshot vào repo local:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('AITeamVN/Vietnamese_Embedding', revision='<SHA_AITEAMVN>', local_dir='models/Vietnamese_Embedding')"

python -c "from huggingface_hub import snapshot_download; snapshot_download('mainguyen9/vietlegal-harrier-0.6b', revision='<SHA_HARRIER>', local_dir='models/vietlegal-harrier-0.6b')"

python -c "from huggingface_hub import snapshot_download; snapshot_download('AITeamVN/Vi-Qwen2-1.5B-RAG', revision='<SHA_LLM>', local_dir='models/Vi-Qwen2-1.5B-RAG')"
```

Không commit các thư mục model lên Git. Ghi ba SHA vào model card/submission log của nhóm.

## 6. Chạy full pipeline lần đầu

### Dùng model ID Hugging Face

```powershell
powershell -ExecutionPolicy Bypass `
  -File scripts\run_pipeline_task2.ps1 `
  -RebuildRetrieval `
  -RebuildGeneration `
  -Package
```

### Dùng model đã khóa trong `models/`

```powershell
powershell -ExecutionPolicy Bypass `
  -File scripts\run_pipeline_task2.ps1 `
  -AnchorModel "models/Vietnamese_Embedding" `
  -HarrierModel "models/vietlegal-harrier-0.6b" `
  -LlmModel "models/Vi-Qwen2-1.5B-RAG" `
  -LocalFilesOnly `
  -RebuildRetrieval `
  -RebuildGeneration `
  -Package
```

Wrapper thực hiện tuần tự để không giữ hai embedding model trên GPU cùng lúc:

1. Encode/cache corpus và rank bằng AITeamVN.
2. Encode/cache corpus và rank bằng Harrier.
3. Rank BM25.
4. Fuse RRF và tạo evidence.
5. Sinh câu trả lời bằng LLM; ghi checkpoint sau từng câu.
6. Hậu xử lý, validate và đóng ZIP.

Không chạy ba nhánh retrieval song song: RAM/VRAM đỉnh sẽ tăng mạnh.

Nếu thiếu VRAM khi encode, giảm batch:

```powershell
.\scripts\run_pipeline_task2.ps1 `
  -EmbeddingBatchSize 16 `
  -QuestionBatchSize 16 `
  -RebuildRetrieval `
  -RebuildGeneration `
  -Package
```

## 7. Chạy lại và resume

### Tiến trình LLM bị dừng giữa chừng

Chạy lại cùng lệnh, không thêm cờ rebuild:

```powershell
.\scripts\run_pipeline_task2.ps1 -LocalFilesOnly -Package
```

Wrapper tái sử dụng rank/context và đọc:

```text
outputs/task2/generation_checkpoint_ensemble.jsonl
```

để chỉ sinh các câu còn thiếu.

### Cập nhật `train-2.json` hoặc `public-official-2.json`

```powershell
python "src/ingestion + EDA/build_pipeline.py" --task 2

.\scripts\run_pipeline_task2.ps1 `
  -RebuildRetrieval `
  -RebuildGeneration `
  -Package
```

`-RebuildRetrieval` ghi lại ba rank và retrieval context. Wrapper tự không resume checkpoint cũ.

### Corpus chunks hoặc model retrieval thay đổi

Nếu `selected-contexts`, logic chunking, model dense, prefix hoặc max sequence length thay đổi, cache
embedding cũ không còn hợp lệ. Chạy:

```powershell
.\scripts\run_pipeline_task2.ps1 `
  -RebuildEmbeddingCaches `
  -RebuildGeneration `
  -Package
```

Cờ này encode lại cả hai corpus cache, đồng thời tự rebuild rank/context. Đây là bước rất tốn thời
gian; không cần dùng khi chỉ thay file câu hỏi.

### Chỉ thay prompt/generation

Không cần encode/rank lại:

```powershell
.\scripts\run_pipeline_task2.ps1 -RebuildGeneration -LocalFilesOnly -Package
```

### Chỉ muốn đóng gói lại file đã có

```powershell
python -m src.evaluation.validate_legalqa_submission `
  --predictions outputs/task2/submission.json `
  --questions data/raw/public-official-2.json

Compress-Archive `
  -Path outputs/task2/package/submission.json `
  -DestinationPath outputs/task2/submission.zip `
  -Force
```

## 8. Kiểm tra submission

Lệnh validate phải in:

```text
Task 2 submission is valid
```

Submission JSON đúng có dạng:

```json
{
  "question_id": {
    "answer": "Câu trả lời pháp luật"
  }
}
```

Mỗi item chỉ được có trường `answer`. Validator còn kiểm tra thiếu/thừa question ID, answer rỗng,
code fence và các wrapper như `<think>`.

Kiểm tra nội dung ZIP:

```powershell
tar -tf outputs/task2/submission.zip
```

ZIP phải chứa đúng file `submission.json`, không chứa thêm thư mục cha.

## 9. Xử lý lỗi thường gặp

### `Rank source ... is stale`

Question ID giống nhưng nội dung câu hỏi đã đổi. Chạy lại với:

```powershell
.\scripts\run_pipeline_task2.ps1 -RebuildRetrieval -RebuildGeneration -Package
```

### `RepositoryNotFoundError`, lỗi mạng hoặc model chưa có cache

Không dùng `-LocalFilesOnly` ở lần đầu; kiểm tra kết nối Hugging Face. Hoặc tải snapshot theo mục 5
và truyền ba đường dẫn local.

### CUDA out of memory khi retrieval

Giảm `-EmbeddingBatchSize` và `-QuestionBatchSize` xuống 16 hoặc 8. Không chạy hai rank dense cùng
lúc. Nếu lỗi xảy ra ở LLM, giữ `-AnswerBatchSize 1`.

### Warning `unsupported legal markers`

Đây là cảnh báo chống hallucination: câu trả lời nhắc một điều luật không xuất hiện trong evidence.
Pipeline vẫn tiếp tục và lưu checkpoint. Xem `diagnostics_ensemble.json`; không tự động xóa điều luật
vì có thể nội dung đúng nhưng evidence bị cắt.

### Câu trả lời thường mở đầu bằng “Dựa vào...”

Prompt hiện đã yêu cầu trả lời trực tiếp. Bước postprocess vẫn chạy như lớp bảo hiểm và báo số câu đã
sửa trong `postprocess_report_ensemble.json`.

### Wrapper đang resume nhưng muốn sinh lại

Dùng `-RebuildGeneration`. Không cần xóa checkpoint thủ công.

## 10. Mốc kết quả Task 2

| Phiên bản | Retrieval | Generation | Điểm BTC | Chênh lệch |
|---|---|---|---:|---:|
| Baseline cũ | Embedding/reranker cũ | `AITeamVN/Vi-Qwen2-1.5B-RAG` | `0.2807` | — |
| Pipeline hiện tại | AITeamVN `1.0` + Harrier `0.2` + BM25 `0.1`, RRF `k=60`, không reranker | `AITeamVN/Vi-Qwen2-1.5B-RAG` | **`0.3450`** | **`+0.0643`** |

Kết quả này xác nhận chất lượng retrieval/evidence ảnh hưởng trực tiếp và đáng kể đến Task 2. Tuy
nhiên, không nên mặc định rằng cấu hình tối ưu Recall@5 của Task 1 cũng là cấu hình tối ưu cho metric
sinh câu trả lời. Các thử nghiệm tiếp theo phải được đánh giá end-to-end trên dữ liệu Task 2.

## 11. Hướng thử nghiệm tiếp theo cho Task 2


### 11.1. Nhiều evidence chunk trong cùng document

Pipeline hiện tại giữ chunk có RRF score cao nhất của mỗi document, sau đó lấy chunk lân cận. Cách
này có thể bỏ sót điều kiện, ngoại lệ hoặc mức xử phạt nằm ở một điều khác nhưng không nằm sát chunk
đại diện.

Nên so sánh:

1. `1 chunk tốt nhất + neighbor_window=1` — baseline hiện tại.
2. Top 2 chunk RRF độc lập trong mỗi document, không yêu cầu liền kề.
3. Top 3 chunk độc lập, có loại trùng nội dung.
4. MMR/diversity: ưu tiên chunk liên quan nhưng không lặp lại cùng một đoạn luật.

Giữ nguyên top document và LLM trong thí nghiệm này để đo riêng tác động của evidence selection. Đây
là hướng ưu tiên cao nhất vì không phải encode lại corpus hoặc đổi model.

### 11.2. Sweep context budget và độ dài câu trả lời

Từ cùng một `retrieval_context_ensemble.json`, thử lưới nhỏ:

| Tham số | Giá trị nên thử |
|---|---|
| `max_documents` | `3`, `5`, `8` |
| `max_chunks_per_document` | `1`, `2`, `3` |
| `neighbor_window` | `0`, `1` |
| `max_context_tokens` | `2048`, `3072`, `4096`, `6000` |
| `max_new_tokens` | `192`, `256`, `384`, `512` |

Không sweep toàn bộ tích Descartes. Bắt đầu từ baseline và đổi từng trục. Metric overlap thường có
thể giảm khi answer chứa quá nhiều thông tin đúng nhưng không xuất hiện trong đáp án tham chiếu, nên
độ dài lớn hơn không mặc định tốt hơn.

Các thử nghiệm này chỉ cần sinh lại answer; không dùng `-RebuildRetrieval`:

```powershell
.\scripts\run_pipeline_task2.ps1 -RebuildGeneration -LocalFilesOnly -Package
```

Muốn sweep thuận tiện hơn cần đưa các giá trị context/generation hiện đang cố định trong wrapper
thành tham số PowerShell và dùng output/checkpoint riêng cho từng cấu hình.

### 11.3. Query-adaptive fusion

Trọng số hiện tại cố định cho mọi câu hỏi. Có thể route theo đặc điểm câu hỏi mà không thêm model:

- Có số hiệu văn bản, `Điều`, `Khoản`, `Điểm`: tăng BM25.
- Câu hỏi tình huống dài: giữ AITeamVN làm anchor mạnh.
- Câu hỏi ngắn, thuần thuật ngữ pháp lý: tăng nhẹ Harrier.
- Câu hỏi chứa trích dẫn chính xác: thêm exact-match citation retrieval trước RRF.

Bắt đầu bằng rule-based router trên validation. Nếu có lợi ổn định, mới huấn luyện classifier nhỏ để
chọn profile trọng số. Không chèn cưỡng bức document vào top kết quả như các thử nghiệm citation/entity
injection trước đây đã thất bại ở Task 1.

Hai corpus embedding cache và ba file rank hiện có vẫn tái sử dụng được; đổi trọng số RRF không cần
encode lại Harrier. Vì Harrier mất khoảng 10 giờ cho lần tạo cache đầu, tuyệt đối không xóa cache khi
chỉ sweep fusion.

### 11.4. Cải thiện prompt và answer style

Thử các prompt yêu cầu:

- Trả lời trực tiếp, không nhắc “ngữ cảnh” hoặc “tài liệu được cung cấp”.
- Nêu căn cứ pháp lý chỉ khi evidence có căn cứ đó.
- Ưu tiên câu trả lời ngắn gần phong cách đáp án train.
- Không suy luận thêm khi evidence thiếu; trả lời phần có thể xác nhận.
- Tách rõ kết luận và điều kiện/ngoại lệ nhưng không lặp lại câu hỏi.

Đánh giá riêng prompt trên cùng context và cùng decode greedy. Chỉ sau khi chốt prompt mới thử thay
decoding; thay prompt và decoding cùng lúc sẽ không xác định được nguồn cải thiện.

### Fine-tune LLM bằng LoRA/QLoRA

Sau khi có validation đáng tin cậy, có thể tạo cặp huấn luyện:

```text
question + retrieval evidence -> answer chuẩn trong train-2
```

Ưu tiên QLoRA 4-bit hoặc LoRA vì GPU laptop 8 GB không phù hợp full fine-tune 1.5B với context dài.
Bắt đầu 1 epoch, learning rate nhỏ và early stopping. Phải sinh retrieval context cho train/validation
bằng cùng cấu hình production để tránh lệch phân phối khi inference.

Rủi ro chính là overfit và làm answer dài hơn đáp án tham chiếu. Không fine-tune trước khi baseline
prompt/context được đo đầy đủ, nếu không sẽ khó biết cải thiện đến từ retrieval hay LLM.

### 11.6. Multi-resolution retrieval

Tạo thêm biểu diễn ở nhiều mức:

- Chunk cấp Điều ngắn.
- Chunk hiện tại.
- Window gồm nhiều Điều liên tiếp.
- Tiêu đề văn bản + chương/mục + nội dung điều.

Fuse rank của các mức rồi chỉ đưa evidence không trùng vào LLM. Chunk ngắn giúp tìm đúng quy định;
window dài giữ điều kiện và ngoại lệ. Hướng này có tiềm năng nhưng phải xây và encode index mới nên
chi phí cao hơn các ưu tiên 1–5.

### 11.7. Peak supervision cho retrieval từ `train-2`

Task 2 không có document label trực tiếp nhưng answer chuẩn thường chứa số hiệu, tên văn bản hoặc
`Điều/Khoản`. Có thể trích các marker này để tạo positive document/chunk yếu, sau đó:

- Đánh giá recall evidence trên tập weak-label.
- Học query router hoặc trọng số fusion.
- Fine-tune nhẹ retriever với learning rate thấp và validation nghiêm ngặt.

Không fine-tune lại AITeamVN theo cơ chế cũ: thí nghiệm Task 1 đã cho thấy catastrophic forgetting.
Weak supervision chỉ nên dùng sau khi xác nhận label trích xuất có độ chính xác cao.

### Hướng ưu tiên thấp hoặc chưa nên lặp lại

- Reranker thay thế toàn bộ thứ hạng RRF.
- Fine-tune AITeamVN theo công thức Task 1 cũ.
- Tăng mạnh trọng số Harrier chỉ vì local score.
- Dedup/backfill, entity boost hoặc citation injection toàn cục.
- Đổi LLM trước khi đo xong prompt/context trên model hiện tại.
- Sampling/self-consistency nhiều lần cho toàn bộ test khi chưa chứng minh lợi ích trên validation.

### Thứ tự triển khai đề xuất

1. Tạo split và báo cáo validation Task 2 cố định.
2. Thử top 2–3 evidence chunk độc lập trong mỗi document.
3. Sweep nhỏ context budget và `max_new_tokens`.
4. Thử query-adaptive fusion bằng rank cache hiện có.
5. Chốt prompt trả lời ngắn, grounded.
6. Fine-tune QLoRA nếu các bước không-train đã bão hòa.
7. Chỉ sau đó mới xây multi-resolution index hoặc đổi model.

Mục tiêu gần nhất hợp lý là cải thiện `0.3450` bằng evidence selection và prompt/context trước. Hai
hướng này rẻ hơn nhiều so với encode lại Harrier hoặc fine-tune LLM, đồng thời giúp xác định rõ nút
thắt hiện nằm ở retrieval, context construction hay generation.
