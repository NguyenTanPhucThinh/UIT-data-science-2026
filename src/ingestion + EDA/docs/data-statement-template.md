# Data Statement — [Tên bộ dữ liệu/tập huấn luyện đã dùng]

> **Bắt buộc theo quy định cuộc thi** (mục 11 - Task 1 / mục 7 - Task 2 trong
> `claude/quy-dinh-cuoc-thi-task1.md` và `claude/quy-dinh-cuoc-thi-task2.md`):
> *"Mỗi bài nộp phải kèm 'Data Statement' và 'Model Card' (nguồn dữ liệu, tiền xử lý,
> phương pháp huấn luyện, chỉ số đánh giá, biện pháp giảm thiểu rủi ro)."*
>
> File này là **khung mẫu** — điền vào từng mục bên dưới trước khi nộp bài. Nếu nhóm dùng
> nhiều biến thể dữ liệu (vd. có/không hard negative mining) cho các model khác nhau, nên
> nhân bản file này thành `data-statement-<tên-model>.md` cho từng model, hoặc ghi rõ trong
> mục "Các biến thể" bên dưới. Xoá dòng hướng dẫn in nghiêng khi điền xong.

## 1. Nguồn dữ liệu (Curation rationale & provenance)

- **Nguồn gốc:** dữ liệu do Ban Tổ chức UIT Data Science Challenge 2026 cung cấp chính thức
  cho Task 1 - LegalIR, gồm:
  - `data/raw/selected-contexts.zip` — 8.532 văn bản hành chính/pháp luật (`context_<id>.json`,
    trường `id`, `name`, `link`, `passage`).
  - `data/raw/train.json` — 7.000 câu hỏi có nhãn (`question` + `answer` = danh sách
    `document_id` đúng), dùng làm dữ liệu train + dev set nội bộ.
  - `data/raw/public-official.json` — 1.000 câu hỏi vòng public test (`answer: null`).
  - *(Điền thêm nếu có `warmup.json` / `private-official.json` được BTC cấp sau, và ngày nhận.)*
- **Không dùng dữ liệu ngoài:** theo đúng quy định "chỉ được dùng bộ dữ liệu chính thức do BTC
  phát hành, không gán nhãn thủ công, không thu thập dữ liệu ngoài, không data augmentation từ
  nguồn ngoài". Nhóm xác nhận [ ] có / [ ] không thêm bất kỳ nguồn dữ liệu nào khác — nếu có,
  liệt kê cụ thể và đối chiếu lại với BTC (quy định cho phép đề xuất bổ sung mô hình mã nguồn mở
  nhưng KHÔNG đề cập cho phép bổ sung dữ liệu train/eval ngoài danh sách BTC cấp).
- **Ngôn ngữ:** tiếng Việt (văn bản hành chính/pháp luật + câu hỏi tiếng Việt).
- **Thời điểm lấy dữ liệu:** [điền ngày BTC phát hành / ngày nhóm tải về].

## 2. Thống kê mô tả dữ liệu

*(Xem chi tiết đầy đủ tại `claude/tong-quan-du-lieu-task1.md`; tóm tắt lại các số liệu chính
đã dùng để huấn luyện/đánh giá model cụ thể trong file này.)*

- Số câu hỏi dùng để train: [điền — vd. toàn bộ 7.000, hay một tập con `--limit-questions`?]
- Số văn bản trong corpus: 8.532 (toàn bộ `selected-contexts.zip`).
- Số cặp (question, positive_passage) tạo được sau xử lý: [điền — output của
  `build_positive_pairs()` trong `src/training/finetune_embedding.py`, in ra log khi chạy].
- Phân bố số document đúng/câu hỏi: 92% câu hỏi chỉ có 1 document đúng (xem
  `claude/tong-quan-du-lieu-task1.md` để biết đầy đủ phân bố 1–5 document/câu hỏi).
- Độ dài văn bản (passage): rất phân tán, từ ~2.262 đến ~670.448 ký tự → cần chunking.

## 3. Tiền xử lý (Preprocessing)

- **Chunking:** văn bản dài được chia nhỏ theo cấu trúc "Điều X" (ưu tiên ranh giới pháp lý tự
  nhiên), fallback cắt theo cửa sổ ký tự (`max_chars`, mặc định 2000) có overlap (mặc định 200)
  khi một "Điều" vẫn quá dài — cài đặt tại `chunk_passage()` trong `src/training/data_utils.py`.
  *(Lưu ý: đây là chunker baseline dùng cho training/evaluation nội bộ — nếu nhóm đã thay bằng
  chunker chính thức của `src/ingestion`, cập nhật lại mục này cho khớp.)*
- **Chọn đoạn positive:** vì BTC chỉ gán nhãn ở cấp document (không chỉ rõ đoạn/Điều nào chứa
  câu trả lời), nhóm dùng heuristic word-overlap (`select_best_chunk()`) để chọn chunk có độ
  trùng từ vựng với câu hỏi cao nhất trong document đúng, thay vì lấy đại chunk đầu tiên (thường
  chỉ là quốc hiệu/tiêu đề văn bản, không chứa nội dung trả lời).
- **Chuẩn hoá văn bản:** [điền — có strip/normalize Unicode, lowercase, v.v. thêm bước nào không?]
- **Negative sampling (nếu train reranker):** [điền — ngẫu nhiên (`build_random_negative_pairs`)
  hay hard negative mining (`mine_hard_negatives`, dùng embedding model nào để mine)?]

## 4. Các biến thể dữ liệu đã thử (nếu có)

*(Vd. có/không hard negative, `max_chars` khác nhau, có/không giới hạn số câu hỏi để debug
nhanh — ghi lại để phần Model Card tương ứng có thể trỏ đúng biến thể đã dùng.)*

| Biến thể | Mô tả | Dùng cho model nào |
|---|---|---|
| [vd. baseline-inbatch] | in-batch negatives, không hard negative | [tên model/thư mục output] |
| [vd. hardneg-v1] | hard negative mining, top_k=5 | [tên model/thư mục output] |

## 5. Rủi ro & biện pháp giảm thiểu (Risk mitigation)

- **Thông tin cá nhân nhạy cảm:** văn bản là văn bản hành chính/pháp luật công khai (Nghị định,
  Thông tư, v.v.) từ nguồn công khai (`thuvienphapluat.vn` theo trường `link`), không phát hiện
  thông tin cá nhân nhạy cảm chưa ẩn danh trong quá trình sử dụng. [Cập nhật nếu phát hiện khác.]
- **Nhiễu nhãn (label noise):** heuristic `select_best_chunk` là suy luận tự động (word-overlap),
  không phải nhãn thủ công của BTC — có thể chọn sai đoạn trong một số trường hợp câu hỏi dùng từ
  đồng nghĩa/diễn giải khác với văn bản gốc. Giảm thiểu bằng cách [điền — có kiểm tra thủ công một
  mẫu ngẫu nhiên không? bao nhiêu %?].
- **Mất cân bằng độ dài văn bản:** văn bản quá dài (>500K ký tự) sau khi chunk có thể tạo nhiều
  chunk "nhiễu" không liên quan cho cùng 1 document — cân nhắc ảnh hưởng tới in-batch negatives
  (một batch có thể chứa 2 chunk cùng document mà không phải hard negative thật).
- **Tuân thủ quy định cuộc thi:** toàn bộ dữ liệu dùng đúng nguồn BTC cấp, không augment/label
  thêm — xem thêm `docs/compliance-risks.md` để theo dõi các rủi ro tuân thủ khác của dự án.

---
*Cập nhật lần cuối: [điền ngày] — người điền: [điền tên thành viên].*
