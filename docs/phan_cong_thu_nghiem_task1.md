# Phân công thử nghiệm Task 1 — Vòng tiếp theo (Hoàng / Kiên / Giang)

Nối tiếp `docs/submission_log_task1.md` (xem đầy đủ lịch sử 20+ thử nghiệm đã chạy, các hướng đã đóng,
và phương pháp luận đã đúc kết tại đó — đặc biệt là quy tắc thẩm định local→public). File này CHỈ theo
dõi các hướng CÒN LẠI, chia việc cho 3 người.

**Quy trình đồng bộ 2 file (để không lệch nhau):** mỗi người tự điền kết quả vào đúng dòng của mình
trong bảng dưới đây ngay khi chạy xong. Khi một hướng đã có **Score PUBLIC xác nhận** (hoặc LOCAL sạch
đủ để đóng hướng mà không cần public — xem quy tắc bên dưới), thêm 1 dòng mới vào bảng chính của
`docs/submission_log_task1.md` (đúng format cột đang dùng ở đó) rồi đánh dấu dòng tương ứng ở đây là
`✅ đã chuyển vào submission_log_task1.md`. `submission_log_task1.md` luôn là bản ghi lịch sử đầy đủ,
đầy đủ nhất; file này chỉ là bảng điều phối công việc đang chạy dở.

## Bối cảnh nhanh (đọc `submission_log_task1.md` nếu cần chi tiết đầy đủ)

**Cấu hình tốt nhất hiện tại — Score PUBLIC 0.9208 (ID 896875):** ensemble RRF 3 nhánh —
`AITeamVN/Vietnamese_Embedding` (weight 1.0) + `mainguyen9/vietlegal-harrier-0.6b` (weight 0.2) + BM25
(weight 0.1), `rrf_k=60`, gộp max, KHÔNG rerank, KHÔNG dedup hậu-xử-lý. Tổng ~1,16 tỷ tham số (dưới
hạn mức 4 tỷ, còn dư ~2,84 tỷ nếu muốn thêm nhánh).

**12+ hướng đã đóng, KHÔNG thử lại:** fine-tune embedding (4 lần, 4 cơ chế thất bại khác nhau) ·
reranker mọi hình thức (4 phép thử độc lập) · đổi hẳn embedding model (`gte-Qwen2-1.5B-instruct`
— cả đứng riêng lẫn làm nhánh ensemble, `AITeamVN v2`, `bge-m3` gốc, `vietlegal-harrier` đứng riêng) ·
entity-boost · citation-graph · dedup+backfill · sweep cách gộp chunk→document · sweep lưới
dense-top-n/bm25-top-n/rrf-k.

**⚠️ Quy tắc thẩm định bắt buộc áp dụng cho MỌI kết quả bên dưới:** chênh lệch Recall LOCAL dưới
~0.01 KHÔNG đủ tin cậy để dự đoán PUBLIC (đã kiểm chứng nhiều lần, xem chi tiết trong
`submission_log_task1.md`). Nếu LOCAL tăng nhưng dưới ngưỡng đó, ghi rõ "chưa đủ bằng chứng", KHÔNG
kết luận "cải thiện" và KHÔNG vội tốn lượt nộp public để xác nhận — trừ khi thay đổi ảnh hưởng ĐỀU lên
mọi câu hỏi (như bm25-weight, ensemble weight) chứ không phải chỉ vài case cụ thể (như entity-boost).

## Bảng phân công

| # | Hướng thử nghiệm | Thời gian ước tính | Người phụ trách | Kết quả / Insights | Recall LOCAL | Score PUBLIC |
|---|---|---|---|---|:---:|:---:|
| A1 | Fine-tune lại v1 với hard-negative đã LỌC theo ngưỡng cosine (loại bỏ candidate có cos > 0.85 với chunk đáp án đúng trước khi đưa vào `mine_hard_negatives_from_ranks()`), dùng MNRL (đã sửa bug margin/collapse) | ~3-4 giờ GPU (fine-tune) + ~20 phút (`quick_eval_finetune.py` kiểm tra sớm trước khi encode full corpus) | **Hoàng** | ✅ **XONG (2026-08-27) — ĐÓNG hướng fine-tune.** Filter chạy đúng (412/6519 candidate bị lọc, ~6,3%, đúng bậc độ lớn ước tính trước) nhưng Δ = -0.1189, không cải thiện so với Lần 3 (-0.1080) — thậm chí hơi tệ hơn. Kết luận: false-negative contamination KHÔNG PHẢI nguyên nhân chính gây fine-tune thất bại. 4/4 lần fine-tune AITeamVN đã thất bại với 4 cơ chế khác nhau — đóng hẳn hướng này, không thử lại trừ khi có ý tưởng thực sự mới. Không cần encode full corpus (tốn 1.5-2h GPU) vì tín hiệu sớm đã đủ rõ ràng âm. **✅ đã chuyển vào submission_log_task1.md** | -0.1189 (quick_eval subset, cận dưới) | — |
| A2 | Cập nhật `scripts/allowed-models.md` cho `mainguyen9/vietlegal-harrier-0.6b` (đã dùng trong ensemble chính thức nhưng chưa khai báo) + soạn đề xuất gửi BTC (quy định mục 7, cần trước 10 ngày hạn chót private test) | ~30-60 phút (admin, không chạy code) | **Hoàng** | *(chưa làm)* | — | — |
| B1 | Reranker chuyên luật `itdainb/PhoRanker` và `Qwen/Qwen3-Reranker-0.6B` làm nguồn phụ RRF (dùng lại `rank-reranker` trong `ensemble_fusion.py`, đã có sẵn từ 4 lần thử reranker trước — chỉ đổi tên model) | ~1-2 giờ/model (tái dùng candidate pool top-100 của v1 có sẵn, không cần encode lại corpus) | **Giang** | *(chưa chạy)* — kỳ vọng THẤP, 4/4 phép thử reranker trước đều âm | — | — |
| B2 | Query expansion (PRF/RM3) cho nhánh BM25 — lấy top-N tài liệu ban đầu, trích term trọng số cao (Rocchio/RM3) mở rộng query gốc, chạy lại BM25. Chỉ dùng corpus chính thức — hợp lệ theo quy định mục 7 | ~3-5 giờ dev (code mới, chưa có sẵn) + eval nhanh (không cần GPU) | **Giang** | *(chưa làm)* — hướng cổ điển, độc lập hoàn toàn với embedding/cache người khác | — | — |
| C1 | Thử chunking khác — structure-aware theo "Điều X" của văn bản luật, hoặc đổi kích thước/overlap so với `--max-chars 2000` hiện tại, re-encode corpus + đo lại LOCAL | ~1 ngày (viết lại chunker + re-encode corpus ~1.5-2h GPU + eval) | **Kiên** | *(chưa làm)* — case [CHUNK] chỉ ~2% trong 53 case đọc tay trước đây, ưu tiên thấp nhưng không xung đột hướng khác | — | — |
| C2 | `CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3` — xác minh prefix qua model card (gated, cần đăng nhập HuggingFace chấp nhận điều khoản hoặc tải về đọc `config_sentence_transformers.json`), encode corpus, đo LOCAL đứng riêng, rồi thử làm nhánh ensemble (dùng `run_ensemble_add_gteqwen2.ps1` làm mẫu) | ~2-3 giờ (encode corpus model 0.6B ~1-1.5h + eval + sweep ensemble) | **Kiên** *(đang chạy sẵn)* | *(đang chạy)* — cùng quy mô tham số harrier (0.6B) nhưng kiến trúc Qwen3 mới hơn, đáng thử độc lập dù pattern "domain-specific thua" đã lặp 2 lần | — | — |
| C3 | Xem lại/sửa ground truth cho 13 câu hỏi lương theo chức danh nghi ngờ GT thiếu văn bản chuyên biệt (danh sách trong `bao-cao-data-quality-cho-Kien.md`) — cần domain judgment, KHÔNG tự động hoá được | ~nửa ngày | **Kiên** | *(chưa làm)* | — | — |

**❌ Đã loại khỏi bảng phân công (2026-08-23): re-crawl 20 tài liệu `passage` rỗng.** Xác nhận việc này
KHÔNG được làm dù Kiên có tài khoản Pro riêng — vẫn tính là "thu thập dữ liệu ngoài bộ dữ liệu chính
thức BTC phát hành" (quy định mục 7), không phân biệt theo ai crawl. 20 tài liệu này đã được xử lý đủ
bằng fallback passage an toàn (xem `submission_log_task1.md`, mục data quality) — không còn việc gì
cần làm thêm cho vấn đề này.

## Ghi chú phân công

- **Hoàng** giữ 2 việc gắn với hạ tầng đã có sẵn (cache, script fine-tune, hiểu rõ 3 lần thất bại
  trước) — không cần onboard người khác vào codebase ensemble/fine-tune phức tạp.
- **Giang** nhận 2 hướng ĐỘC LẬP, tự chứa (không phụ thuộc cache riêng của Hoàng, có sẵn script mẫu để
  copy — `rank-reranker`) — phù hợp nếu Giang mới tiếp cận codebase.
- **Kiên** giữ các việc gắn với dữ liệu gốc (chunking/ingestion, GT) cộng CATI-AI (đã nhận sẵn, đang
  chạy) — đúng vai trò đã đảm nhiệm từ đầu dự án (báo cáo data-quality 2026-08-17).
- Đây là giả định phân công dựa trên vai trò đã biết từ trước — nếu Kiên/Giang có ràng buộc GPU/thời
  gian khác, đổi lại các hướng cho hợp lý, KHÔNG cần giữ đúng bảng này.

## Định nghĩa "xong" cho mỗi dòng

Một dòng được coi là hoàn tất khi có: (1) Recall LOCAL đo trên `val_split_20.json` bằng đúng quy trình
trong `submission_log_task1.md` (mục "Cách lấy số liệu LOCAL"), (2) kết luận rõ ràng ĐÓNG/GIỮ theo
đúng quy tắc thẩm định ở trên, (3) nếu đủ điều kiện nộp public (LOCAL tăng ≥0.01 và ảnh hưởng đều lên
nhiều câu hỏi, hoặc không đo được LOCAL sạch) thì có thêm Score PUBLIC. Không cần đợi tất cả 8 dòng
xong mới cập nhật — cập nhật ngay từng dòng khi có kết quả để cả 3 người thấy tiến độ chung.
