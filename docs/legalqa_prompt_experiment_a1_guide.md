# A1 — Thử nghiệm prompt LegalQA có chọn tự động và sinh submission

Tài liệu này hướng dẫn chạy trọn vẹn mục A1 trong
[`phan_cong_thu_nghiem_task2.md`](phan_cong_thu_nghiem_task2.md): so sánh các prompt trên cùng
retrieval/context/LLM/decode, chọn biến thể tốt nhất theo metric local, rồi sinh submission Task 2
mới bằng đúng biến thể đã chọn.

Script điều phối là:

```text
scripts/run_prompt_experiment_a1.ps1
```

Không cần sửa tay `src/qa/prompts.py` khi chạy thử. Mỗi prompt có tên cố định và được truyền vào
pipeline bằng `--prompt-variant`; nhờ đó checkpoint, output và report của các prompt không ghi đè
lẫn nhau.

## 1. Mục tiêu và nguyên tắc đo

Mục tiêu A1 là đo riêng ảnh hưởng của cách yêu cầu LLM trả lời. Các yếu tố dưới đây bị khóa cho tất
cả biến thể:

- Retrieval: AITeamVN `1.0` + Harrier `0.2` + BM25 `0.1`, RRF `k=60`, top 5 document.
- Evidence/context: `max_documents=5`, `max_chunks_per_document=3`, `neighbor_window=1`,
  `max_context_tokens=4096`.
- LLM: `AITeamVN/Vi-Qwen2-1.5B-RAG`.
- Decode greedy: `do_sample=false`, `max_new_tokens=384`, batch generation 1.
- Hậu xử lý tiền tố và metric local: giữ nguyên cho mọi lần chạy.

Việc khóa các yếu tố này rất quan trọng: nếu đổi context hoặc sampling cùng lúc, không thể kết luận
điểm thay đổi là do prompt.

## 2. Pipeline end-to-end

```text
train_2_preprocessed.jsonl (có answer)
       |
       | deterministic shuffle: seed=20260828, lấy 200 câu
       v
val_questions.jsonl + val_references.json + val_split_manifest.json
       |
       +--> AITeamVN cache (.npz) ----+
       +--> Harrier cache (.npz) ------+--> ranks_*.npz --> RRF context cố định
       +--> BM25 -----------------------+                         |
                                                         retrieval_context_ensemble.json
                                                                      |
 baseline / answer_first / legal_structured / grounded_concise ------+
                                                                      v
                                      raw answer -> postprocess -> submission_<variant>.json
                                                                      |
                                                        METEOR + ROUGE-L từng biến thể
                                                                      |
                                     chọn METEOR cao nhất, ROUGE-L không giảm so với baseline
                                                                      |
                                                chạy public generation bằng prompt thắng
                                                                      v
                                    submission_a1_best_prompt.json + ZIP + best_pipeline_run.json
```

Lần tạo rank validation chỉ encode **câu hỏi**. Script bắt buộc hai cache corpus hiện hữu:
`outputs/cache/embcompare_aiteamvn.npz` và
`outputs/cache/embcompare_vietlegal_harrier.npz`; nó không truyền cờ cho phép encode lại corpus.
Vì vậy A1 không lặp lại bước Harrier hàng giờ.

## 3. Các prompt được thử

Định nghĩa nằm trong [`src/qa/prompts.py`](../src/qa/prompts.py).

| Tên | Ý đồ |
|---|---|
| `baseline` | Prompt production hiện tại: trả lời trực tiếp, grounded, không preamble. |
| `answer_first` | Ép câu đầu nêu ngay kết luận, điều kiện/căn cứ theo sau. |
| `legal_structured` | Kết luận trước, sau đó chỉ đưa căn cứ và điều kiện có thật trong evidence. |
| `grounded_concise` | Câu trả lời ngắn, sát đáp án; chỉ giữ chi tiết có thể xác nhận. |

Mọi biến thể cùng cấm các tiền tố kiểu “Dựa vào ngữ cảnh…”, bịa điều/khoản, XML/Markdown và suy
luận ngoài evidence. Bước postprocess vẫn được giữ làm lớp bảo hiểm, và lưu số câu bị sửa trong
`postprocess_<variant>.json`.

## 4. Input bắt buộc

Chạy từ thư mục gốc repo, sau khi kích hoạt `.venv`.

| Input | Mặc định | Vai trò |
|---|---|---|
| Training Task 2 đã preprocess | `data/processed/train_2_preprocessed.jsonl` | Nguồn tạo validation có đáp án. |
| Corpus chunks | `data/processed/chunks_corpus.jsonl` | Xây evidence context. |
| AITeamVN cache | `outputs/cache/embcompare_aiteamvn.npz` | Tránh encode lại corpus dense. |
| Harrier cache | `outputs/cache/embcompare_vietlegal_harrier.npz` | Tránh encode lại corpus Harrier. |
| Public Task 2 | `data/processed/public_test_2_preprocessed.jsonl` | Tạo submission cuối. |
| Public ID reference | `data/raw/public-official-2.json` | Validator kiểm tra đủ/sai question ID. |
| Model local hoặc Hugging Face cache | theo `-LlmModel` | LLM sinh từng câu trả lời. |

Nếu file preprocess chưa có, chuẩn bị dữ liệu trước:

```powershell
python "src/ingestion + EDA/build_pipeline.py" --task 2
```

Nếu corpus hoặc cách chunk thay đổi, cache embedding không còn đáng tin cậy. Khi đó chạy lại
pipeline production với `-RebuildEmbeddingCaches` trước, rồi mới chạy A1.

## 5. Chạy A1

Lần đầu đã tải model xong, chạy đầy đủ bằng:

```powershell
powershell -ExecutionPolicy Bypass `
  -File scripts\run_prompt_experiment_a1.ps1 `
  -LocalFilesOnly
```

Không thêm `-LocalFilesOnly` trong lần đầu nếu model chưa có trong Hugging Face cache/local folder.
Script sẽ:

1. Tạo một split validation cố định 200 câu với seed `20260828`.
2. Dùng hai embedding cache hiện có để rank 200 câu validation, rank BM25, rồi tạo một retrieval
   context ensemble duy nhất.
3. Sinh câu trả lời cho bốn prompt, hậu xử lý và tính METEOR/ROUGE-L từng prompt.
4. Chọn biến thể METEOR cao nhất trong các biến thể có ROUGE-L không giảm so với `baseline`.
5. Gọi `scripts/run_pipeline_task2.ps1 -RebuildGeneration -PromptVariant <prompt_thắng>` để sinh,
   validate và đóng ZIP submission public mới.
6. Ghi log tái lập cho pipeline thắng.

Để chỉ thử một tập nhỏ hơn trong lúc kiểm tra kỹ thuật:

```powershell
.\scripts\run_prompt_experiment_a1.ps1 `
  -ValidationSize 20 `
  -PromptVariants baseline,answer_first `
  -PublicOutput outputs/task2/submission_a1_smoketest.json
```

Không dùng submission từ smoke test để nộp bài. Lần chạy chính thức nên giữ 200 câu, bốn prompt
và output mặc định.

## 6. Quy tắc chọn biến thể

Module [`src/evaluation/select_legalqa_prompt_variant.py`](../src/evaluation/select_legalqa_prompt_variant.py)
thực hiện chọn tự động với quy tắc xác định:

1. `baseline` là mốc.
2. Loại biến thể có `ROUGE-L < ROUGE-L baseline - MaxRougeDrop`.
3. Trong phần còn lại, lấy METEOR cao nhất.
4. Nếu hòa, lấy ROUGE-L cao hơn; nếu vẫn hòa thì ưu tiên baseline, sau đó tên prompt theo thứ tự
   chữ cái.

Mặc định `MaxRougeDrop=0.0`, tức không chấp nhận giảm ROUGE-L. Nếu đội muốn chấp nhận nhiễu metric
nhỏ, cần nêu rõ ngưỡng, ví dụ `-MaxRougeDrop 0.002`; ngưỡng này sẽ được lưu trong log.

Không coi metric local là score BTC. Sau khi chọn prompt, nên đọc ngẫu nhiên 20–30 câu trong output
và chỉ nộp public khi câu trả lời không lặp tiền tố, không dài bất thường, không bịa căn cứ pháp lý.

## 7. Output và log

Tất cả artifact thử nghiệm nằm ở `outputs/task2/a1/` theo mặc định.

| Artifact | Nội dung |
|---|---|
| `val_questions.jsonl` | 200 câu validation, có cả answer để module nhận diện đúng Task 2. |
| `val_references.json` | Đáp án chuẩn, schema `{question_id: {"answer": ...}}` cho evaluator. |
| `val_split_manifest.json` | Source, seed, size và danh sách question ID — bằng chứng split cố định. |
| `ranks_aiteamvn.npz`, `ranks_harrier.npz`, `ranks_bm25.npz` | Top-100 rank validation, có hash nội dung câu hỏi. |
| `retrieval_context_ensemble.json` | Evidence cố định dùng cho toàn bộ prompt. |
| `submission_<variant>_raw.json` | Answer LLM trước hậu xử lý. |
| `submission_<variant>.json` | Answer sạch dùng để chấm local. |
| `generation_checkpoint_<variant>.jsonl` | Checkpoint sau từng câu của đúng prompt đó. |
| `diagnostics_<variant>.json` | Token prompt, độ dài answer, evidence đã dùng cho những câu sinh trong lần chạy. |
| `postprocess_<variant>.json` | Số/lý do câu bị bỏ meta prefix. |
| `metrics_<variant>.json` | METEOR, ROUGE-L và chi tiết từng câu. |
| `prompt_selection.json` | Toàn bộ metric, eligibility và biến thể được chọn. |
| `best_pipeline_run.json` | Log cuối: prompt thắng, rule chọn, metric, setting bị khóa, đường dẫn artifact và SHA-256 file prompt/wrapper. |

Submission public mới mặc định là:

```text
outputs/task2/submission_a1_best_prompt.json
outputs/task2/submission.zip
```

`submission.zip` chỉ chứa `submission.json` theo quy định đóng gói của wrapper production. File
JSON có tên `submission_a1_best_prompt.json` là bản dễ phân biệt trong workspace; trước khi nộp,
mở `outputs/task2/package/submission.json` để xác nhận đó chính là JSON đã đóng ZIP.

## 8. Resume, chạy lại và thay đổi dữ liệu

Nếu generation validation bị dừng, chạy lại cùng tham số và thêm:

```powershell
.\scripts\run_prompt_experiment_a1.ps1 -LocalFilesOnly -ResumeValidationGeneration
```

Script chỉ resume checkpoint của từng biến thể riêng. Không resume nếu bạn vừa thay prompt source;
khi đó checkpoint cũ không còn cùng điều kiện thí nghiệm.

Nếu chỉ muốn sinh lại câu trả lời của mọi prompt trên split/context cũ, chạy mặc định lần nữa
(không có `-ResumeValidationGeneration`). Các checkpoint được làm mới trước khi generation.

Nếu muốn tạo split mới có chủ ý (không dùng để so trực tiếp với report cũ):

```powershell
.\scripts\run_prompt_experiment_a1.ps1 `
  -RebuildValidationSplit `
  -Seed 20260829
```

`-RebuildValidationSplit` tự tạo lại rank/context validation vì hash câu hỏi sẽ đổi. Nếu chỉ thiếu
hoặc muốn tạo lại rank validation, dùng `-RebuildValidationRetrieval`; cache corpus vẫn được tái sử
dụng. Không xóa cache Harrier chỉ để chạy A1.

Nếu `train-2` hoặc corpus thay đổi, chạy preprocessing tương ứng, tạo split A1 mới và kiểm tra lại
cache corpus trước khi so sánh kết quả.

## 9. Thời gian ước tính và lỗi thường gặp

Trên GPU laptop đã dùng cho pipeline hiện tại:

- Rank/context validation 200 câu: thường vài phút đến vài chục phút; không có encode corpus.
- Generation validation: khoảng 1–1,5 giờ/biến thể theo lần đo trước; bốn biến thể khoảng 4–6 giờ.
- Public generation 1.000 câu: thêm khoảng 1–1,5 giờ.

Tổng lần chính thức khoảng 5–8 giờ, phụ thuộc độ dài context/answer và VRAM.

Nếu báo thiếu cache embedding, đừng thêm cờ encode một cách vô tình. Kiểm tra đường dẫn cache hoặc
xây lại cache đúng corpus qua `scripts/run_pipeline_task2.ps1 -RebuildEmbeddingCaches`. Nếu LLM
chưa được tải mà có `-LocalFilesOnly`, bỏ cờ đó một lần để tải model, sau đó chạy offline.

## 10. Các file/module được thêm hoặc cập nhật

| File | Vai trò |
|---|---|
| `scripts/run_prompt_experiment_a1.ps1` | Entry point A1 end-to-end, gọi các bước theo thứ tự và sinh submission cuối. |
| `src/evaluation/prepare_legalqa_prompt_validation.py` | Tạo split answered, deterministic và manifest. |
| `src/evaluation/select_legalqa_prompt_variant.py` | Chọn prompt bằng quy tắc metric tái lập, lưu `prompt_selection.json`. |
| `src/qa/prompts.py` | Khai báo bốn biến thể prompt có tên. |
| `src/qa/pipeline.py` | Truyền prompt variant vào message builder. |
| `src/evaluation/run_legalqa.py` | Nhận CLI `--prompt-variant`. |
| `scripts/run_pipeline_task2.ps1` | Nhận `-PromptVariant` để public generation dùng chính prompt thắng. |

Pipeline production chung được mô tả ở
[`legalqa_pipeline_guide.md`](legalqa_pipeline_guide.md). A1 chỉ bổ sung lớp validation/chọn prompt;
không thay đổi retrieval ensemble đang đạt public Task 2 `0.3450` trước thử nghiệm này.
