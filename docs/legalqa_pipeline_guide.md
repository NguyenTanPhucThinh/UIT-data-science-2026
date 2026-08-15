# Guidance for pipeline Task 2 — LegalQA

## 1. Pipeline hoàn chỉnh

```text
File câu hỏi Task 2
  -> load + chuẩn hóa câu hỏi
  -> dense retrieval (embedding fine-tune)
  -> BM25 retrieval
  -> RRF ở cấp chunk
  -> gom chunk thành candidate document
  -> reranker fine-tune
  -> evidence chunk có provenance/score
  -> thêm chunk lân cận cùng document
  -> cắt context bằng tokenizer của Vi-Qwen2
  -> Vi-Qwen2-1.5B-RAG sinh answer local, greedy
  -> bỏ reasoning/XML/Markdown
  -> cảnh báo claim pháp lý không có trong evidence
  -> submission.json
  -> validator
  -> submission.zip
```

## 2. Các file/module đã tạo

### 2.1. Package `src/qa`

| File | Input | Output | Vai trò |
|---|---|---|---|
| `src/qa/types.py` | mapping evidence/QA | `EvidenceChunk`, `QAInput`, `QAResult` | Hợp đồng dữ liệu có type và nguồn gốc rõ ràng |
| `src/qa/config.py` | tham số CLI/config | `LLMConfig`, `ContextConfig`, `LegalQAGenerationConfig` | Cấu hình model, context budget và decode |
| `src/qa/prompts.py` | question + context | chat messages | Prompt tiếng Việt cố định, cấm suy đoán/XML/Markdown và tiền tố meta |
| `src/qa/context_builder.py` | ranked evidence + toàn bộ chunk + tokenizer | context đã cắt token, danh sách evidence dùng thực tế | Mở rộng chunk lân cận, dedupe và bảo vệ context window |
| `src/qa/generator.py` | chat messages | raw answer | Load `AITeamVN/Vi-Qwen2-1.5B-RAG` local và batch generation |
| `src/qa/postprocess.py` | raw answer + context | clean answer + unsupported markers | Bỏ `<think>`, XML, prompt wrapper, tiền tố meta; kiểm tra số liệu/Điều luật |
| `src/qa/pipeline.py` | `QAInput[]` + corpus chunks | `QAResult[]`, submission dict | Điều phối context → prompt → generation → cleanup |
| `src/qa/__init__.py` | — | public package exports | API gọn cho module khác import |

### 2.2. Retrieval/evaluation

| File | Thay đổi/đầu vào | Output | Vai trò |
|---|---|---|---|
| `src/evaluation/full_pipeline_eval.py` | thêm `evidence_output` tùy chọn | giữ nguyên Task 1 output; thêm ranked evidence nếu caller yêu cầu | Tái sử dụng đúng retrieval đã kiểm chứng, không phá caller cũ |
| `src/evaluation/retrieve_for_task2.py` | questions + retrieval artifacts | JSON `question + evidence` | Xuất chunk liên quan thay vì toàn văn document |
| `src/evaluation/run_legalqa.py` | questions + chunks + 3 model | `submission.json`, context/diagnostics tùy chọn | CLI end-to-end chính thức cho Task 2 |
| `src/evaluation/postprocess_legalqa_submission.py` | raw submission JSON | cleaned submission + report | Xóa tiền tố nói về context/tài liệu nhưng giữ căn cứ pháp lý thực chất |
| `src/evaluation/legalqa_metrics.py` | reference/prediction strings | METEOR, ROUGE-L, per-question | Metric local với tokenizer Unicode tiếng Việt |
| `src/evaluation/evaluate_legalqa.py` | reference JSON + prediction JSON | báo cáo console/JSON | CLI đánh giá local |
| `src/evaluation/validate_legalqa_submission.py` | prediction + question reference | danh sách lỗi / exit code | Chặn schema sai, thiếu ID, answer rỗng, reasoning wrapper |
| `src/evaluation/task_selection.py` | đường dẫn questions | Task 1/2 hoặc unknown | Chặn LegalQA nhận nhầm artifact Task 1 |

### 2.3. Test đã tạo

| File | Phạm vi |
|---|---|
| `tests/test_qa_context_builder.py` | neighbor expansion, khác document, token budget |
| `tests/test_qa_postprocess.py` | strip reasoning/XML và marker grounding |
| `tests/test_qa_pipeline.py` | orchestration với fake generator, không tải LLM |
| `tests/test_legalqa_metrics.py` | METEOR/ROUGE-L exact, disjoint, missing prediction |
| `tests/test_legalqa_submission.py` | schema hợp lệ và wrapper không hợp lệ |
| `tests/test_postprocess_legalqa_submission.py` | cleanup submission, bảo toàn schema và thống kê thay đổi |
| `tests/test_retrieval_evidence.py` | reranker giữ đúng chunk provenance/score |
| `tests/test_task_selection.py` | mapping file Task 1/2, schema Task 2 và chặn chọn nhầm |

### 2.4. Script vận hành

| File | Vai trò |
|---|---|
| `scripts/run_pipeline_task2.ps1` | Wrapper PowerShell chạy end-to-end, validate và tùy chọn đóng ZIP |

## 3. Các file hiện có đã sửa

| File | Nội dung sửa |
|---|---|
| `src/ingestion + EDA/run_phase2.py` | dùng VnCoreNLP thật; PyVi chỉ fallback theo từng dòng |
| `src/ingestion + EDA/build_pipeline.py` | thêm `--task 1`, `--task 2`, `--task all` |
| `src/config/task_data.py` | mapping tập raw/clean/preprocessed riêng cho hai task |
| `src/retrieval/reranker.py` | default base reranker đổi sang `BAAI/bge-reranker-v2-m3` |
| `requirements.txt` | thêm direct dependencies cho LLM và METEOR |
| `docs/allowed-models.md` | đánh dấu Vi-Qwen2-1.5B-RAG là LLM Task 2 đã chọn |
| `docs/retrieval.md` | đồng bộ reranker chính thức |


## 5. Chuẩn bị môi trường

### 5.1. Phần mềm

- Python 3.10+.
- Java 8+ cho VnCoreNLP.
- Khuyến nghị GPU NVIDIA; CPU chạy được nhưng LLM/retrieval rất chậm.
- Dung lượng đủ cho embedding, reranker, LLM và embedding cache.

### 5.2. Cài dependency

Từ thư mục gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Máy NVIDIA/Windows phải kiểm tra PyTorch CUDA trước khi train:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Nếu kết quả có hậu tố `+cpu`, `torch.version.cuda` là `None` hoặc giá trị cuối là `False`, cài
wheel CUDA chính thức phù hợp. Ví dụ với driver mới hỗ trợ CUDA 13.x:

```powershell
python -m pip install --force-reinstall --no-deps torch==2.13.0 `
  --index-url https://download.pytorch.org/whl/cu130
```

Sau đó lệnh kiểm tra phải trả về `torch.cuda.is_available() == True`. Wheel PyTorch đã kèm CUDA
runtime cần thiết; thông thường không cần cài riêng CUDA Toolkit để chạy wheel.

Nếu PowerShell chặn activation, có thể gọi trực tiếp `.venv\Scripts\python.exe` cho các lệnh dưới đây.

### 5.3. Sinh processed artifacts

Đặt các file câu hỏi tương ứng trong `data/raw/` và chọn rõ task cần xử lý; chỉ `--task all`
yêu cầu đủ cả bốn file:

```powershell
# Task 1: train.json + public-official.json
python "src/ingestion + EDA/build_pipeline.py" --task 1

# Task 2: train-2.json + public-official-2.json
python "src/ingestion + EDA/build_pipeline.py" --task 2

# Sinh artifact câu hỏi cho cả hai task
python "src/ingestion + EDA/build_pipeline.py" --task all
```


Mapping output câu hỏi:

| Task | Raw public | Processed public |
|---|---|---|
| Task 1 | `data/raw/public-official.json` | `data/processed/public_test_preprocessed.jsonl` |
| Task 2 | `data/raw/public-official-2.json` | `data/processed/public_test_2_preprocessed.jsonl` |

`corpus.jsonl`, `corpus_preprocessed.jsonl` và `chunks_corpus.jsonl` dùng chung cho hai task.

Không cần xóa trước `corpus*.jsonl`, `chunks_corpus.jsonl` hoặc các file question JSONL; pipeline
sẽ ghi đè chúng. Nhưng nếu `chunks_corpus.jsonl`, tokenizer, embedding model hoặc prefix thay đổi,
phải bỏ/tạo lại mọi vector artifact phụ thuộc corpus:

```text
outputs/cache/*.npz
embeddings/corpus_embeddings.npy
embeddings/metadata.jsonl
vectorstore/corpus.index
```

Không được trộn artifact PyVi cũ với query VnCoreNLP mới. `run_legalqa.py` hiện dùng cache NPZ
truyền qua `--embedding-cache`; nó không tự đọc `corpus_embeddings.npy` hoặc `corpus.index`.
Do đó hai file này không cần thiết cho lệnh Task 2 ở mục 8, nhưng phải được tái tạo trước khi dùng
pipeline module trong `src/retrieval/`.

## 6. Chuẩn bị model retrieval

Đây là bước bắt buộc trước mục 8. `run_pipeline_task2.ps1` chỉ điều phối inference; script không
tự fine-tune hai model retrieval. Nếu dùng đường dẫn local mặc định mà checkpoint chưa tồn tại,
wrapper sẽ dừng sớm và chỉ rõ model còn thiếu.

CLI mặc định dùng hai checkpoint local tốt nhất:

```text
models/vietnamese-legal-embedding-ft-full
models/reranker-ft-full
```

Nếu hai thư mục này chưa tồn tại, có hai lựa chọn.

### Lựa chọn A — train lại checkpoint chính thức

```powershell
python -m src.training.finetune_embedding `
  --base-model bqbbao6/VN_legal_embedding_512 `
  --train-json data/raw/train.json `
  --corpus data/raw/selected-contexts `
  --output-dir models/vietnamese-legal-embedding-ft-full `
  --device cuda `
  --use-amp `
  --batch-size 8 `
  --checkpoint-path outputs/checkpoints/embedding-ft-full `
  --checkpoint-save-steps 100

python -m src.training.finetune_reranker `
  --base-model BAAI/bge-reranker-v2-m3 `
  --train-json data/raw/train.json `
  --corpus data/raw/selected-contexts `
  --output-dir models/reranker-ft-full `
  --hard-negatives `
  --hard-negative-embedding-model models/vietnamese-legal-embedding-ft-full `
  --hard-negative-top-k 1 `
  --epochs 1 `
  --batch-size 4 `
  --device cuda
```

Nếu embedding training bị gián đoạn sau khi đã có checkpoint, chạy lại nguyên lệnh embedding
ở trên và thêm `--resume-from-checkpoint`. Không resume từ lần chạy CPU cũ nếu chưa tồn tại
`outputs/checkpoints/embedding-ft-full`. Với GPU 8 GB, cấu hình khởi đầu an toàn là
`--batch-size 8 --use-amp`; chỉ tăng batch sau khi smoke test không OOM.

### Lựa chọn B — smoke test với pretrained model

Thêm vào lệnh chạy LegalQA:

```text
--embedding-model bqbbao6/VN_legal_embedding_512
--reranker-model BAAI/bge-reranker-v2-m3
--embedding-cache outputs/cache/chunks_corpus_embeddings__pretrained.npz
```

Lựa chọn B chạy được end-to-end nhưng chất lượng dự kiến thấp hơn checkpoint fine-tune.

Có thể smoke test trực tiếp bằng wrapper và cache riêng, không làm ảnh hưởng cache fine-tune:

```powershell
.\scripts\run_pipeline_task2.ps1 `
  -EmbeddingModel "bqbbao6/VN_legal_embedding_512" `
  -RerankerModel "BAAI/bge-reranker-v2-m3" `
  -EmbeddingCache "outputs/cache/chunks_corpus_embeddings__pretrained.npz" `
  -Package
```

Trước khi chạy cấu hình chính thức, kiểm tra hai checkpoint local:

```powershell
Test-Path models/vietnamese-legal-embedding-ft-full
Test-Path models/reranker-ft-full
```

Cả hai phải trả về `True`. `corpus_embeddings.npy` và `corpus.index` không thay thế hai thư mục
checkpoint: chúng chỉ chứa vector/index corpus, không chứa encoder để nhúng câu hỏi hoặc model
cross-encoder để rerank.

## 7. Tải LLM Task 2

Model đã chọn:

```text
AITeamVN/Vi-Qwen2-1.5B-RAG
```

Lần chạy đầu, `transformers` tải model về Hugging Face cache. Sau khi tải xong, chạy lại với `--local-files-only` để bảo đảm không có network call.

Khuyến nghị trước khi submit:

1. Ghi lại model revision/commit hash.
2. Đo số tham số từ checkpoint thực tế.
3. Nếu cần môi trường hoàn toàn portable, copy snapshot checkpoint vào `models/Vi-Qwen2-1.5B-RAG` và truyền đường dẫn local cho `--llm-model`.

## 8. Chạy pipeline hoàn chỉnh

### 8.0. Wrapper PowerShell

Chỉ chạy mục này sau khi hai lệnh `Test-Path` ở mục 6 đều trả về `True` (hoặc đã chủ động chọn
cặp pretrained theo cách B, mục 6).

```powershell
.\scripts\run_pipeline_task2.ps1 -Package
```

Nếu PowerShell chặn script bởi Execution Policy, dùng:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline_task2.ps1 -Package
```

Wrapper mặc định dùng `data/processed/public_test_2_preprocessed.jsonl` và đối chiếu ID với
`data/raw/public-official-2.json`. Với private test hoặc tên file khác, truyền lại `-Questions` và
`-QuestionReferenceJson` tương ứng. Câu trả lời được ghi sau từng câu vào
`outputs/task2/generation_checkpoint.jsonl`; chạy lại cùng wrapper sẽ tự resume các câu còn thiếu.
Sau generation, wrapper giữ bản gốc tại `submission_raw.json`, xóa tiền tố meta thành
`submission.json`, validate bản đã làm sạch rồi mới tạo ZIP. 

Nếu chạy trực tiếp CLI thay vì wrapper, hậu xử lý submission bằng:

```powershell
python -m src.evaluation.postprocess_legalqa_submission `
  --input outputs/task2/submission_raw.json `
  --output outputs/task2/submission.json `
  --report outputs/task2/postprocess_report.json
```

Luôn validate và đóng ZIP từ
`submission.json`, không dùng `submission_raw.json` để nộp.


## 9. Debug retrieval

### 9.1 Evidence

```powershell
python -m src.evaluation.retrieve_for_task2 `
  --questions data/processed/public_test_2_preprocessed.jsonl `
  --output outputs/task2/evidence.json `
  --top-k 5
```

Output này dùng để kiểm tra document/chunk/section trước khi LLM sinh. CLI end-to-end vẫn là cách chính để tạo submission; script evidence không tự sinh answer.

### 9.2 Generation

Hiện generation được tích hợp trong `run_legalqa.py` để bảo đảm cùng contract và tokenizer budget. Nếu cần vận hành tách process hoàn toàn, nên thêm một CLI generation-only đọc `evidence.json`; không nên tự ghép full document bằng script ngoài.
