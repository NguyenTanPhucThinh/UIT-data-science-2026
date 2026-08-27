# Phân công thử nghiệm Task 2 — Vòng tiếp theo (Bảo / Thịnh)



**Quy trình đồng bộ 2 file (để không lệch nhau):** mỗi người tự điền kết quả vào đúng dòng của mình
trong bảng dưới đây ngay khi chạy xong. Khi một hướng đã có **Score PUBLIC xác nhận** (hoặc METEOR
LOCAL sạch đủ để đóng hướng mà không cần public — xem quy tắc bên dưới), thêm 1 dòng mới vào bảng kết
quả mục 10 của `docs/legalqa_pipeline_guide.md` (hoặc `docs/submission_log_task2.md` nếu đã tạo) rồi
đánh dấu dòng tương ứng ở đây là `✅ đã chuyển vào legalqa_pipeline_guide.md`. File hướng dẫn pipeline
luôn là bản ghi lịch sử đầy đủ nhất; file này chỉ là bảng điều phối công việc đang chạy dở.

## Bối cảnh nhanh (đọc `legalqa_pipeline_guide.md` nếu cần chi tiết đầy đủ)

**Cấu hình tốt nhất hiện tại — Score PUBLIC 0.3450:** retrieval ensemble RRF 3 nhánh giữ nguyên từ
Task 1 (`AITeamVN/Vietnamese_Embedding` weight 1.0 + `mainguyen9/vietlegal-harrier-0.6b` weight 0.2 +
BM25 weight 0.1, `rrf_k=60`, gộp max, KHÔNG rerank) → `retrieval_context_ensemble.json` → context
builder (1 chunk tốt nhất/doc + `neighbor_window=1`) → `AITeamVN/Vi-Qwen2-1.5B-RAG` → postprocess.
Tăng từ baseline cũ `0.2807` (+0.0643, ~22,9%). Tổng ~2,66 tỷ tham số (dưới hạn mức 4 tỷ).

**Các hướng đã đóng hoặc chưa nên lặp lại (xem mục 11 cuối `legalqa_pipeline_guide.md`):
Hiện tại chỉ đóng các hướng xử lý retrieval mà task 1 đã đóng.


**Hướng chờ :** QLoRA fine-tune LLM: Nên xử lý trên google colab trước khi đăng lên github

Thử các cách prompt engineering khác nhau để xem nó có tăng hiệu suất lên hay không

Thử các hướng LLM khác như:
 thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2,
ntphuc149/ViLegalQwen3-1.7B-Base,
ntphuc149/ViLegalQwen2.5-1.5B-Base.


## Bảng phân công

| # | Hướng thử nghiệm | Thời gian ước tính | Người phụ trách | Kết quả / Insights | METEOR LOCAL | Score PUBLIC |
|---|---|---|---|---|:---:|:---:|
| A1 | Thử biến thể prompt trong `src/qa/prompts.py`: trả lời trực tiếp, không preamble "Dựa vào...", ưu tiên ngắn gọn gần `train-2`, không suy luận khi evidence thiếu. Giữ nguyên context + LLM + greedy decode | ~4-6 giờ (~1 giờ chỉnh 3-4 biến thể + ~1-1,5 giờ/generation trên val, mỗi biến thể chỉ `-RebuildGeneration`) | **Bảo** | *(chưa chạy)* — đo riêng prompt, không đổi decoding cùng lúc | — | — |
| B1 | Thử finetune LLM Baseline mình đã dùng cho task 2 cho phần QA sử dụng QLora  | ~1-2 ngày (~2 giờ setup Colab/dataset + ~6-8 giờ GPU QLoRA trên ~7000 mẫu + ~2 giờ tích hợp adapter và generation eval) | **Thịnh** | *(chưa chạy)* — ưu tiên cao nhất, không cần encode lại corpus | — | — |
| B2 | Thử dùng các LLM thay thế đã đề cập ở hướng mở đã cho| ~2-3 giờ/model (~30-45 phút download/tích hợp + ~1,5-2 giờ generation full 1000 câu public); ~1 ngày cho cả 3 model | **Bảo** | *(chưa chạy)* — chỉ mở sau A1 có prompt baseline ổn định | — | — |




## Định nghĩa "xong" cho mỗi dòng

Một dòng được coi là hoàn tất khi có: (1) METEOR và ROUGE-L PUBLIC có kết quả chính thức, lúc này mới nên quyết định là giữ hay đóng
