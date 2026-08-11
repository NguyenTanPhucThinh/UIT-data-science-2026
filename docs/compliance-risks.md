# Rủi ro vi phạm quy định cuộc thi — theo dõi liên tục

File này ghi lại các điểm trong quá trình phát triển có khả năng vi phạm thể lệ
UIT Data Science Challenge 2026, để cả nhóm cùng theo dõi và xử lý trước khi nộp bài.
Cập nhật file này mỗi khi phát hiện rủi ro mới hoặc xử lý xong một rủi ro đã ghi nhận.

Quy định gốc tham chiếu: `docs/DSC2026_Task1_LegalIR_Data_Overview.docx` và thể lệ chung
cuộc thi (mục "Quy định về mô hình" / "Quy định về API và giấy phép sử dụng" / Điều 4 Điều lệ).

---

## 🔴 Đang mở (cần xử lý)

### 1. Qdrant Cloud được ghi làm output của bước Storage & Representation
- **Phát hiện tại:** `TASK_DISTRIBUTION.md`, mục "Thành viên 2 + 5: Storage & Representation" — Output: "Qdrant Cloud".
- **Vi phạm:** mục "Quy định về API và giấy phép sử dụng" — 2 gạch đầu dòng:
  > Không được phép sử dụng bất kỳ API nào... bao gồm cả API thương mại và API phi thương mại.
  > BTC chỉ chấp nhận các mô hình/hệ thống mã nguồn mở mà đội thi có thể tải về, vận hành và
  > kiểm soát trực tiếp, không thông qua dịch vụ của bên thứ ba.
- **Vì sao vi phạm:** Qdrant Cloud là dịch vụ vector DB hosted bởi Qdrant Inc. — dù bản thân
  phần mềm Qdrant mã nguồn mở, dùng bản Cloud nghĩa là gọi API qua mạng tới server bên thứ ba,
  đúng thứ bị cấm ở cả 2 gạch đầu dòng trên. Kể cả dùng gói miễn phí cũng không được miễn trừ,
  vì quy định ghi rõ "bao gồm cả API phi thương mại".
- **Hướng khắc phục đề xuất:** đổi sang Qdrant chạy local qua Docker (tự host, tự kiểm soát),
  hoặc đơn giản hơn là dùng FAISS (thư viện chạy hoàn toàn local, không cần server riêng).
- **Trạng thái:** người phụ trách (@Nguyễn Huy Hoàng) đã xác nhận sẽ thông báo cho nhóm, đặc
  biệt bạn Nguyễn Tấn Phúc Thịnh (phụ trách `src/embeddings` + `src/vectorstore`). *Chưa có xác
  nhận đã đổi sang giải pháp local.*
- **Ngày phát hiện:** 2026-08-11

### 2. GPT-4o / Gemini xuất hiện trong sơ đồ pipeline Task 2
- **Phát hiện tại:** `docs/rag_pipeline_with_anti_hallucination.svg`, node "LLM trả lời (RAG)"
  ghi ví dụ: "GPT-4o, Gemini, Llama...".
- **Vi phạm:** vi phạm ĐỒNG THỜI 3 quy định — cấm dùng bất kỳ API nào, chỉ được dùng mô hình
  mã nguồn mở tự kiểm soát (không qua bên thứ ba), và giới hạn tổng tham số dưới 4 tỷ (GPT-4o/
  Gemini không mở, không rõ số tham số nhưng chắc chắn vượt xa 4B).
- **Hướng khắc phục đề xuất:** chỉ dùng LLM mã nguồn mở tự host cho bước "LLM trả lời (RAG)" và
  bước "LLM chống hallucination" của Task 2 — ví dụ model đã liệt kê trong
  `docs/allowed-models.md` (thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2), hoặc model mở
  khác miễn tổng tham số toàn pipeline vẫn dưới 4 tỷ.
- **Trạng thái:** CHƯA XÁC NHẬN với người phụ trách Task 2 (`src/qa`, bạn Nguyễn Trường Giang)
  — có thể chỉ là text minh hoạ khi vẽ sơ đồ (không phải quyết định thật của nhóm), nhưng cần
  làm rõ và sửa lại sơ đồ nếu đúng vậy để tránh hiểu nhầm về sau.
- **Ngày phát hiện:** 2026-08-11

---

## 🟡 Cần theo dõi / rà soát định kỳ (chưa xác nhận vi phạm)

- **Tổng tham số toàn pipeline:** model đề xuất hiện tại (`docs/allowed-models.md`) gồm
  bqbbao6/vietnamese-legal-embedding + AITeamVN/Vietnamese_Reranker +
  thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2 (~1.7B riêng phần LLM). Cần cộng chính xác
  tổng tham số CẢ 3 model (+ bất kỳ model nào thêm sau này) trước khi chốt, đảm bảo dưới 4 tỷ.
  Chưa có số liệu chính thức verify từng model.
- **`.env.example.txt` đang trống:** sự tồn tại của file này gợi ý nhóm định dùng biến môi
  trường (thường để chứa API key). Cần rà soát khi file này được điền nội dung — nếu là API key
  cho dịch vụ ngoài (kể cả Qdrant Cloud, hay bất kỳ LLM API nào) thì vi phạm.
- **Cấu hình load model qua HuggingFace:** một số thư viện (transformers, sentence-transformers)
  có thể mặc định gọi HuggingFace Inference API (endpoint online) nếu cấu hình sai, thay vì tải
  checkpoint về chạy local. Cần đảm bảo code luôn tải model về đĩa và chạy local (không bật
  `use_remote` / không gọi qua `huggingface_hub.InferenceClient`).
- **License của các model cộng đồng ít tên tuổi:** bqbbao6/vietnamese-legal-embedding,
  mainguyen9/vietlegal-harrier-0.6b, và các model tương tự do cá nhân đăng tải trên HuggingFace —
  cần kiểm tra license cụ thể (research/education/non-commercial OK theo quy định) trước khi
  dùng chính thức, đặc biệt nếu lọt Top 7 phải công khai mã nguồn.
- **Dữ liệu training:** `src/training/finetune_embedding.py` hiện chỉ dùng
  `data/raw/train.json` + `data/raw/selected-contexts.zip` (đúng dữ liệu BTC cấp, không gán
  nhãn thủ công thêm) — đã tuân thủ, ghi lại đây để nhóm biết và không vô tình thêm dữ liệu
  ngoài vào script training sau này.

---

## ✅ Đã xử lý

*(chưa có mục nào — cập nhật khi một rủi ro ở trên được xác nhận khắc phục xong)*

---

## Cách dùng file này

- Khi phát hiện rủi ro mới (dù chỉ là nghi ngờ), thêm vào mục 🔴 hoặc 🟡 kèm: mô tả, vị trí phát
  hiện, quy định liên quan (trích dẫn nếu có), hướng khắc phục đề xuất, trạng thái, ngày phát hiện.
- Khi xử lý xong, chuyển mục đó xuống ✅ **Đã xử lý** kèm ghi chú cách đã khắc phục và ngày xử lý.
- Trước khi nộp bài (cả public test lẫn private test), rà lại toàn bộ mục 🔴 — không được còn
  mục nào ở trạng thái mở khi nộp bài chính thức.
