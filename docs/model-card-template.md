# Model Card — [Tên model, vd. vietnamese-legal-embedding-ft-hardneg]

> **Bắt buộc theo quy định cuộc thi** (mục 11 - Task 1 / mục 7 - Task 2 trong
> `claude/quy-dinh-cuoc-thi-task1.md` và `claude/quy-dinh-cuoc-thi-task2.md`):
> *"Mỗi bài nộp phải kèm 'Data Statement' và 'Model Card' (nguồn dữ liệu, tiền xử lý,
> phương pháp huấn luyện, chỉ số đánh giá, biện pháp giảm thiểu rủi ro)."*
>
> File này là **khung mẫu** — nhân bản thành `docs/model-cards/<tên-model>.md` cho MỖI model
> thành phần thật sự dùng trong pipeline nộp bài (embedding, reranker, LLM sinh câu trả lời nếu
> có), vì tổng tham số + compliance phải verify riêng cho từng model. Xoá hướng dẫn in nghiêng
> khi điền xong.

## 1. Thông tin cơ bản (Model details)

| | |
|---|---|
| Tên model | [vd. vietnamese-legal-embedding-ft-hardneg] |
| Loại | [ ] Embedding  [ ] Reranker (cross-encoder)  [ ] Generative/LLM |
| Model base (pretrained gốc) | [vd. bqbbao6/vietnamese-legal-embedding, từ HuggingFace Hub] |
| Có fine-tune không | [ ] Có — script: `src/training/finetune_embedding.py` hoặc `finetune_reranker.py` [ ] Không (dùng nguyên bản pretrained) |
| Số tham số | [điền số chính xác, vd. từ `model.num_parameters()` hoặc HuggingFace model card của base] |
| Ngày huấn luyện/hoàn thiện | [điền ngày] |
| Người phụ trách | [điền tên thành viên] |
| Vị trí lưu checkpoint | [điền đường dẫn, vd. `models/vietnamese-legal-embedding-ft-hardneg/`] |

## 2. Mục đích sử dụng (Intended use)

- Dùng trong bước [điền — vd. "dense retrieval" trong pipeline hybrid Task 1, hoặc "rerank top-20
  ứng viên sau khi fusion BM25+dense"].
- Phạm vi: chỉ dùng cho bài toán LegalIR/LegalQA tiếng Việt trong khuôn khổ UIT Data Science
  Challenge 2026 — không đảm bảo hiệu năng ngoài phạm vi văn bản hành chính/pháp luật VN.

## 3. Dữ liệu huấn luyện (Training data)

- Xem chi tiết đầy đủ tại Data Statement tương ứng: `docs/data-statement-template.md` (hoặc bản
  đã điền cụ thể, vd. `docs/data-statements/vietnamese-legal-embedding-ft-hardneg.md`).
- Tóm tắt nhanh: [điền — vd. "7.000 câu hỏi train.json, 109 cặp positive/câu hỏi debug mẫu 100
  câu → điền số thật khi chạy full 7.000 câu", "hard negative mining top_k=5 dùng chính base
  model trước fine-tune"].

## 4. Phương pháp huấn luyện (Training procedure)

- **Loss function:** [ ] MultipleNegativesRankingLoss (in-batch negatives)  [ ] TripletLoss
  (hard negatives)  [ ] Binary cross-entropy (reranker, `CrossEncoder` với `label=0/1`)
- **Hyperparameters:**

| Tham số | Giá trị |
|---|---|
| epochs | [điền] |
| batch_size | [điền] |
| max_chars (chunk) | [điền] |
| learning rate | [điền — mặc định sentence-transformers nếu không set riêng] |
| warmup_steps | [điền — tính tự động = 10% tổng step, xem log khi chạy] |
| seed | [điền] |
| hard negatives | [ ] Có (top_k=[điền]) [ ] Không |

- **Phần cứng:** [điền — vd. GPU RTX 3060, thời gian huấn luyện thực tế].
- **Lệnh chạy đầy đủ đã dùng:** (copy nguyên dòng lệnh thực tế đã chạy để tái lập được)

```bash
[điền lệnh, vd:
python -m src.training.finetune_embedding \
    --base-model bqbbao6/vietnamese-legal-embedding \
    --train-json data/raw/train.json \
    --corpus data/raw/selected-contexts.zip \
    --output-dir models/vietnamese-legal-embedding-ft-hardneg \
    --epochs 3 --batch-size 16 --hard-negatives]
```

## 5. Kết quả đánh giá (Evaluation)

*(Chạy `src/evaluation/baseline_retrieval.py` rồi `src/evaluation/evaluate.py` — hoặc pipeline
retrieval thật của nhóm nếu đã sẵn sàng — trên `data/raw/train.json` làm dev set nội bộ, rồi
điền kết quả vào đây trước khi so sánh các model.)*

| Model | Recall@5 (dev) | Precision@5 (dev) | Ghi chú |
|---|---|---|---|
| [Base — chưa fine-tune] | [điền] | [điền] | baseline so sánh |
| [Model này] | [điền] | [điền] | [vd. "+X điểm Recall so với base"] |

- Tập đánh giá: [ ] `train.json` (dev nội bộ, 7.000 câu — lưu ý model đã "thấy" tập này khi
  train nên số liệu có thể lạc quan hơn thực tế)  [ ] tập con giữ lại riêng không dùng để train
  (khuyến nghị nếu có, để đánh giá khách quan hơn) [ ] `public-official.json` (qua submit thật
  lên hệ thống BTC — điểm chính thức).
- Điểm public test chính thức (nếu đã nộp): [điền].

## 6. Rủi ro & biện pháp giảm thiểu (Ethical considerations / risk mitigation)

- **Thiên lệch dữ liệu (data bias):** model học từ 7.000 câu hỏi + 8.532 văn bản do BTC cung cấp,
  có thể không đại diện đầy đủ mọi lĩnh vực pháp luật (vd. phân bố lĩnh vực văn bản không đều) —
  [điền nếu có phát hiện cụ thể].
- **Giới hạn ngữ cảnh:** văn bản dài nhất ~670K ký tự phải chunk nhỏ, có thể mất ngữ cảnh liên
  Điều/Khoản khi câu trả lời cần tham chiếu chéo nhiều Điều trong cùng văn bản.
- **Giấy phép model base:** [điền — license của model base, xác nhận cho phép dùng nghiên cứu/phi
  thương mại theo đúng quy định cuộc thi; xem `claude/danh-sach-model-tham-khao.md`].

## 7. Checklist tuân thủ quy định cuộc thi (bắt buộc điền trước khi nộp bài)

- [ ] Model mã nguồn mở, tải về và chạy hoàn toàn local (không gọi API/dịch vụ bên thứ ba nào).
- [ ] Số tham số của RIÊNG model này: [điền số]. Tham số CỘNG DỒN toàn pipeline (embedding +
      reranker + LLM nếu có): [điền tổng] — xác nhận **< 4.000.000.000**.
- [ ] Chỉ dùng dữ liệu BTC cấp để fine-tune (không data ngoài, không augmentation ngoài).
- [ ] Nếu vào Top 7: đã chuẩn bị sẵn Docker image + mã nguồn giấy phép MIT để BTC tái lập.
- [ ] Đã đối chiếu với `docs/compliance-risks.md` — không còn rủi ro 🔴 nào liên quan tới model này.

---
*Cập nhật lần cuối: [điền ngày] — người điền: [điền tên thành viên].*
