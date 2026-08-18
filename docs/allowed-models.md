# Các model sử dụng trong cuộc thi (UIT-DSC 2026, Task 1 LegalIR + Task 2 LegalQA)

> Cập nhật 2026-08-14. Ràng buộc theo `quy-dinh-cuoc-thi-task1.md` mục 5-6: **tổng tham số toàn bộ
> pipeline phải dưới 4 tỷ (4B)**, chỉ mã nguồn mở tự vận hành (không API), giấy phép nghiên
> cứu/giáo dục/phi thương mại được chấp nhận. **Muốn bổ sung model mới ngoài danh sách này (kể cả
> mã nguồn mở): phải gửi đề xuất cho BTC trước 10 ngày so với hạn chót vòng private test** (theo
> mục 7 quy định) — nên cập nhật file này ngay khi quyết định đổi/thêm model để không quên gửi đề
> xuất kịp hạn.

## ✅ Đang dùng trong pipeline chính thức (cấu hình dòng #4/#6 trong `submission-log-task1.md`, đã nộp, Score PUBLIC 0.8387)

| Vai trò | Model gốc | Đã fine-tune riêng? | Tham số (~) | Giấy phép |
|---|---|---|---|---|
| Embedding (bi-encoder, retrieval dense) | [`bqbbao6/VN_legal_embedding_512`](https://huggingface.co/bqbbao6/VN_legal_embedding_512) (fine-tune từ `intfloat/multilingual-e5-base`) | Có — `models/vietnamese-legal-embedding-ft-full` (fine-tune thêm trên train.json cuộc thi, in-batch negatives, prefix E5-style) | ~278M | Cần xác nhận lại trên model card trước khi chốt nộp cuối |
| Reranker (cross-encoder, tinh chỉnh top-k) | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Có — `models/reranker-ft-full` (fine-tune thêm trên train.json cuộc thi, hard negatives) | ~568M | Apache 2.0 |
| BM25 (sparse, không phải model học máy) | `rank_bm25` (thuật toán, không phải model) | — | 0 | — |

**Tổng tham số ước tính: ~846M — cách xa trần 4B, còn dư nhiều ngân sách** (kể cả khi sau này ghép
thêm model sinh (LLM) cho Task 2 vào cùng 1 pipeline nếu 2 task tính chung — cần xác nhận với BTC).

⚠️ *Sửa lỗi (2026-08-14): file này trước đó ghi sai repo ID của bqbbao6 là
`bqbbao6/vietnamese-legal-embedding` — ID đúng là `bqbbao6/VN_legal_embedding_512` (dùng ID sai gây
lỗi 401 "Repository Not Found" khi load qua sentence-transformers). File cũng từng liệt kê
`AITeamVN/Vietnamese_Reranker` như đang dùng chính thức — thực ra mới chỉ ở giai đoạn thử nghiệm và
đã bị loại (xem bảng candidate bên dưới).*

## 🔎 Candidate đã thử nghiệm nhưng KHÔNG áp dụng (embedding/reranker)

| Model | Vai trò thử | Ngày test | Kết quả | Kết luận |
|---|---|---|---|---|
| [`AITeamVN/Vietnamese_Reranker`](https://huggingface.co/AITeamVN/Vietnamese_Reranker) (fine-tune từ `bge-reranker-v2-m3` trên Zalo Legal 2021, ~568M) | Thay reranker | 2026-08-14 | Recall 0.8179 (dòng #7) — THUA cả `reranker-ft` (0.8624) VÀ bản `bge-reranker-v2-m3` pretrained chưa tune gì (0.8324, dòng #8) | Dấu hiệu negative transfer từ Zalo Legal 2021, không tổng quát tốt sang corpus cuộc thi. Đã đóng hướng thử nghiệm này. |

## 🔲 Candidate chưa test — đáng thử tiếp cho embedding/reranker Task 1

| Model | Vai trò | Lý do đáng thử | Nguồn gợi ý |
|---|---|---|---|
| [`AITeamVN/Vietnamese_Embedding`](https://huggingface.co/AITeamVN/Vietnamese_Embedding) (fine-tune từ `BAAI/bge-m3`, ~600M, 1024 chiều) | Thay embedding | Cùng team AITeamVN huấn luyện cặp với reranker đã thử — nhưng LÀ EMBEDDING (khác reranker), kết quả xấu của reranker AITeamVN không nhất thiết lặp lại | Claude gợi ý 2026-08-14 khi tra cứu HuggingFace |
| [`CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3`](https://huggingface.co/CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3) | Thay embedding | Embedding chuyên pháp lý tiếng Việt khác nguồn, đáng so sánh chéo | Claude gợi ý 2026-08-14 |
| `truro7/vn-law-embedding` | Thay embedding | Chuyên pháp luật VN, chưa xác minh kỹ tên gọi/model card | `danh-sach-model-tham-khao.md` (2026-08-10) |
| `mainguyen9/vietlegal-harrier-0.6b` | Thay embedding | Chuyên pháp luật VN (~0.6B) | `danh-sach-model-tham-khao.md` (2026-08-10) |

## ✅ Đã chọn cho pipeline Task 2 (LegalQA)

| Vai trò | Model | Tham số (~) | Giấy phép | Trạng thái |
|---|---|---:|---|---|
| LLM sinh câu trả lời RAG | [`AITeamVN/Vi-Qwen2-1.5B-RAG`](https://huggingface.co/AITeamVN/Vi-Qwen2-1.5B-RAG) | 1.5B | Apache 2.0 | Pipeline local tại `src/qa/` và CLI `src/evaluation/run_legalqa.py` |

**Tổng pipeline ước tính:** retrieval ~846M + LLM ~1.5B = **~2.346B**, dưới trần 4B. Trước
khi nộp chính thức vẫn phải đo lại số tham số từ đúng checkpoint/revision đã đóng gói.

## 🔲 Candidate dự phòng cho Task 2

| Model | Tham số (~) | Ghi chú |
|---|---|---|
| `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2` | 1.7B | Qwen3 fine-tune GRPO chuyên pháp luật VN — candidate hàng đầu cho Task 2 |
| `ntphuc149/ViLegalQwen2.5-1.5B-Base` | 1.5B | Chuyên pháp luật VN |
| `ntphuc149/ViLegalQwen3-1.7B-Base` | 1.7B | Chuyên pháp luật VN |

⚠️ Nếu Task 1 + Task 2 tính CHUNG 1 trần 4B tham số (cần hỏi lại BTC để chắc chắn), pipeline Task 1
hiện tại (~846M) + 1 model Task 2 (1.5-1.7B) vẫn nằm dưới trần thoải mái (~2.3-2.6B).

Xem danh sách đầy đủ hơn (đã tổng hợp từ các đội khác + tham khảo) tại `danh-sach-model-tham-khao.md`
trong project Claude — file đó liệt kê nhiều lựa chọn hơn nhưng CHƯA phân loại theo trạng thái
đang-dùng/đã-thử/chưa-thử như file này.
