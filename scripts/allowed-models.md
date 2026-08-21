# Các model sử dụng trong cuộc thi (UIT-DSC 2026, Task 1 LegalIR)

> Cập nhật 2026-08-15. Ràng buộc theo `quy-dinh-cuoc-thi-task1.md` mục 5-6: **tổng tham số toàn bộ
> pipeline phải dưới 4 tỷ (4B)**, chỉ mã nguồn mở tự vận hành (không API), giấy phép nghiên
> cứu/giáo dục/phi thương mại được chấp nhận. **Muốn bổ sung model mới ngoài danh sách này (kể cả
> mã nguồn mở): phải gửi đề xuất cho BTC trước 10 ngày so với hạn chót vòng private test** (theo
> mục 7 quy định) — nên cập nhật file này ngay khi quyết định đổi/thêm model để không quên gửi đề
> xuất kịp hạn.

## ✅ Cấu hình LOCAL tốt nhất ĐÃ XÁC NHẬN PUBLIC hiện tại (cập nhật 2026-08-16 chiều) — `AITeamVN/Vietnamese_Embedding` (PRETRAINED), `bm25-weight=0.1`, Score PUBLIC **0.9173** — chính thức sau khi thử fine-tune, đổi model lớn hơn, VÀ reranker đều thất bại

**Đã xác nhận trên public test thật** (xem `submission-log-task1.md` dòng #11b): nộp
`submission_v6_aiteamvn_bm25p1.zip` (ID **890769**, 2026-08-16 15:07) — đạt **Score PUBLIC 0.9173**,
tăng **+0.0115** so với bản `bm25-weight=0.4` trước đó (dòng #11, ID 889673, 0.9058). `bm25-weight=0.1`
chính thức thay thế `0.4` làm mặc định. `AITeamVN/Vietnamese_Embedding` PRETRAINED (dựa trên
`BAAI/bge-m3`, ~568M tham số, không cần prefix, `--max-seq-length 2048`) vẫn là embedding chính thức.

| Vai trò | Model gốc | Đã fine-tune riêng? | Tham số (~) | Giấy phép |
|---|---|---|---|---|
| Embedding (bi-encoder, retrieval dense) | [`AITeamVN/Vietnamese_Embedding`](https://huggingface.co/AITeamVN/Vietnamese_Embedding) (base `BAAI/bge-m3`) | **Chưa** — dùng thẳng bản pretrained, KHÔNG cần prefix (`--query-prefix=""` `--passage-prefix=""`), `--max-seq-length 2048` | 567.754.752 (~0,568 tỷ, đã xác minh qua API HuggingFace) | Cần xác nhận lại trên model card trước khi chốt nộp cuối |
| BM25 (sparse, không phải model học máy) | `rank_bm25` (thuật toán, không phải model), trọng số trong RRF **`bm25-weight=0.1`** (đổi từ 0.4, xem trên) | — | 0 | — |

**Tổng tham số hiện dùng: ~568M — CHỈ dùng ~14% ngân sách 4B, còn dư gần như TOÀN BỘ ngân sách.**
Reranker (`BAAI/bge-reranker-v2-m3` fine-tune, `models/public_train/reranker-ft-full`, ~568M) **đã
bị BỎ khỏi cấu hình đang dùng** (phát hiện làm giảm Recall — xem `submission-log-task1.md` mục
"stage-eval xác nhận RERANKER làm giảm điểm") — code vẫn giữ nguyên trong repo để tham khảo/Model
Card, nhưng KHÔNG chạy trong pipeline nộp bài hiện tại.

**Việc tiếp theo (cập nhật 2026-08-16 rạng sáng, xem `submission-log-task1.md` mục "🏆 QUYẾT ĐỊNH
HIỆN TẠI"):**
1. ✅ Nộp public xác nhận bản PRETRAINED — HOÀN TẤT (0.9058, dòng #11).
2. ❌ Fine-tune `AITeamVN/Vietnamese_Embedding` trên TOÀN BỘ train.json (bỏ qua Giai đoạn 1 80%, đi
   thẳng vào full-train theo yêu cầu tiết kiệm thời gian) — **ĐÃ CHẠY XONG VÀ ĐÃ NỘP**
   (`submission_v5_aiteamvn_ft.zip`, ID 889985, 2026-08-15 18:22) — **Score PUBLIC 0.8922, THẤP HƠN
   bản pretrained tới -0.0136**. Nghi ngờ catastrophic forgetting (model đã train trên 1.1M triplet
   nội bộ, fine-tune thêm trên ~7-8 nghìn cặp hẹp của cuộc thi làm hại thay vì giúp) — xem
   `submission-log-task1.md` dòng #12 và mục "Bài học rút ra". **ĐÃ ĐÓNG hướng thử nghiệm fine-tune
   AITeamVN lại — quay về dùng bản PRETRAINED (dòng #11) làm cấu hình chính thức.**
3. ✅ Sweep lại `--bm25-weight` riêng cho bản **PRETRAINED** — **HOÀN TẤT (2026-08-15 tối, dòng
   #11b)**. Đỉnh mới `weight=0.1` (Recall LOCAL **0.9303**, +0.0036 so với `0.4` đang dùng) — cải
   thiện nhỏ hơn ngưỡng "+0.01" nên CHƯA nộp public riêng để xác nhận, có thể gộp vào lần nộp sau.
4. ⚠️ Đổi sang embedding lớn hơn `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (PRETRAINED, KHÔNG fine-tune)
   — **ĐÃ CHẠY XONG (2026-08-16 rạng sáng)** — Recall LOCAL **0.8864**, THẤP HƠN cả 2 mốc AITeamVN
   (0.9267/0.9303). Check truncation (2026-08-16, `scripts/check_truncation_gte_qwen2.py`) cho thấy
   **21,63% chunk bị cắt cụt** ở `--max-seq-length 512` đã dùng — kết quả này **KHÔNG THỂ KẾT LUẬN
   CHẮC CHẮN** (một phần chênh lệch có thể do truncation, không hẳn model yếu hơn). Đo công bằng hơn
   (max-seq-length 1024, chỉ 0,5% cắt cụt) cần ~21h GPU — **quyết định (2026-08-16, Hoàng): TẠM đóng
   lại, không chạy lại ngay**, KHÔNG nộp public. Ý tưởng dự phòng: thuê server/cloud GPU để thử lại sau
   nếu có ngân sách — chưa lên kế hoạch. Xem bảng "Candidate đã thử nghiệm nhưng KHÔNG áp dụng" bên
   dưới và chi tiết đầy đủ trong `submission-log-task1.md` dòng #13.
5. ❌ Thử lại reranker (`BAAI/bge-reranker-v2-m3` pretrained) trên chính baseline AITeamVN hiện tại —
   **ĐÃ CHẠY XONG (2026-08-16)**. Recall LOCAL giảm từ 0.9267 xuống **0.8660** (-0.0607) — XÁC NHẬN
   LẠI (lần 2, trên baseline mạnh hơn hẳn `bqbbao6`) kết luận cũ rằng reranker làm giảm điểm, mức giảm
   tương đối còn LỚN HƠN (-6,5% so với -2,1% trên `bqbbao6`). **ĐÃ ĐÓNG HẲN hướng dùng reranker cho
   Task 1** — không cần thử lại trừ khi có hướng khác hẳn (fine-tune reranker riêng cho domain, hoặc
   kết hợp điểm thay vì thay thế thứ hạng).
6. ✅ Nộp public xác nhận `bm25-weight=0.1` — **ĐÃ CHẠY XONG (2026-08-16 15:07)**. Score PUBLIC
   **0.9173** (ID 890769), +0.0115 so với `bm25-weight=0.4` (0.9058). **`bm25-weight=0.1` chính thức
   thay thế `0.4` làm cấu hình mặc định — xem bảng ở đầu file.**
7. ✅ Chạy `scripts/check_truncation_gte_qwen2.py` — **HOÀN TẤT (2026-08-16)**. 21,63% chunk bị cắt cụt
   — xác nhận dòng #13 confounded, đã tạm đóng lại (xem mục 4 ở trên).
8. ❌ Kiểm tra complementarity gte-Qwen2 vs AITeamVN (`scripts/check_complementarity.py`, để xem có
   đáng làm fusion 3 chiều không) — **ĐÃ CHẠY XONG (2026-08-16)**. Tỉ lệ gte-Qwen2 "cứu" được tài liệu
   AITeamVN bỏ lỡ ở đúng nhóm cần quan tâm: n_relevant=2 chỉ 12,8%, n=3/n=4 là 0% — **KHÔNG có tín hiệu
   bổ sung ở nhóm yếu nhất. ĐÃ BỎ ý tưởng fusion 3 chiều với gte-Qwen2** (chi tiết xem
   `submission-log-task1.md`). **Không còn việc nào trong danh sách ưu tiên gần đây** — các hướng tối
   ưu tiếp theo (tăng `--dense-top-n`/`--bm25-top-n`, tinh chỉnh BM25, phân tích lỗi định tính câu hỏi
   n_relevant≥2, chất lượng data với Kiên) chưa được lên kế hoạch cụ thể.

⚠️ **Cấu hình `bqbbao6` cũ (đã bị thay thế, chỉ giữ lại tham khảo lịch sử):** embedding
`bqbbao6/VN_legal_embedding_512` (fine-tune từ `intfloat/multilingual-e5-base`, ~278M, ĐÃ fine-tune
trên train.json cuộc thi, prefix E5-style bắt buộc `"query: "`/`"passage: "`), `bm25-weight=0.4`,
KHÔNG rerank — best public đã xác nhận trước đó: **0.8472** (dòng #9, `bm25-weight=1.0`); dòng #10
(`bm25-weight=0.4`, Recall LOCAL 0.8919) chưa từng nộp public riêng.

⚠️ *Sửa lỗi (2026-08-14): file này trước đó ghi sai repo ID của bqbbao6 là
`bqbbao6/vietnamese-legal-embedding` — ID đúng là `bqbbao6/VN_legal_embedding_512` (dùng ID sai gây
lỗi 401 "Repository Not Found" khi load qua sentence-transformers). File cũng từng liệt kê
`AITeamVN/Vietnamese_Reranker` như đang dùng chính thức — thực ra mới chỉ ở giai đoạn thử nghiệm và
đã bị loại (xem bảng candidate bên dưới).*

## 🎯 Ngân sách còn dư rất lớn (2026-08-14 tối) — vì sao đáng thử model embedding LỚN hơn hẳn

Từ khi bỏ reranker khỏi pipeline, model embedding hiện tại (~568M) chỉ dùng ~14% trần 4B — nghĩa là
có thể thử thẳng các model embedding **lớn hơn nhiều lần** (tới gần 4B tham số cho riêng 1 model
embedding) mà vẫn hợp lệ, thay vì chỉ so các model cỡ nhỏ (300M-600M) như trước đây. Đã research kỹ
(2026-08-14 tối) các ứng viên mạnh trong tầm 1-4B, xác minh tham số THẬT (không tin theo tên gọi —
xem cảnh báo Qwen3-Embedding-4B bên dưới) qua HuggingFace API/`config.json`.

### ⚠️ CẢNH BÁO QUAN TRỌNG: `Qwen/Qwen3-Embedding-4B` — tham số THẬT vượt trần 4B, KHÔNG dùng được

Tên gọi "4B" chỉ là số làm tròn. **Tham số THẬT đã xác minh qua HuggingFace API (`safetensors.total`),
đối chiếu chéo bằng cách tính tay từ `config.json`: 4.021.774.336 (~4,022 tỷ)** — vượt trần "dưới 4
tỷ" khoảng 21,8 triệu tham số. Đây là kiểu bẫy tên gọi giống hệt các lần trước (`PhoGPT-4B-Chat`
thực ra 3.7B, `Nemotron-3-Embed-1B` thực ra 1.14B) — chỉ khác lần này lệch theo hướng NGUY HIỂM hơn
(tên làm tròn XUỐNG, thực tế lại VƯỢT trần, không phải dưới trần). **KHÔNG dùng model này một mình
làm embedding** trừ khi hỏi lại BTC xác nhận có cho phép sai số làm tròn hay không (rủi ro cao, nên
tránh). Model 8B cùng dòng (`Qwen/Qwen3-Embedding-8B`, thật ~7,57 tỷ) càng vượt xa hơn, loại hẳn.

### 🔲 Candidate mạnh, CHƯA test — trong ngân sách 4B, tham số đã xác minh THẬT (research 2026-08-14)

| Model | Tham số THẬT (đã xác minh) | Chiều embedding | Max seq len | Prefix/prompt query-passage | Giấy phép | Ghi chú |
|---|---|---|---|---|---|---|
| **`Qwen/Qwen3-VL-Embedding-2B`** | **2.127.532.032 (~2,13 tỷ)** — an toàn dưới trần, dùng nhiều ngân sách hơn hẳn các model đang thử | tới 2048 (Matryoshka 64-2048) | 32K token | Kiểu instruction/system-prompt tùy biến (mặc định `"Represent the user's input."`) | Apache 2.0 | Kiến trúc RoPE — nghiên cứu VN-MTEB (arXiv 2507.21500) cho thấy model RoPE thắng model APE trên tiếng Việt. Rất mới, CHƯA có benchmark tiếng Việt công khai — cần tự đo. **Ứng viên ưu tiên hàng đầu để thử SAU KHI fine-tune AITeamVN xong** (xem thứ tự ưu tiên ở mục trên). |
| **`Alibaba-NLP/gte-Qwen2-1.5B-instruct`** | **~1.776 tỷ (tính tay từ config.json — tên gọi "1.5B" cũng là làm tròn, thực tế cao hơn)** | 1536 | 32K token | Query: `"Instruct: {task}\nQuery: {query}"`; Passage: không cần prefix | Apache 2.0 | Kiến trúc RoPE (họ Qwen2). VN-MTEB tìm thấy bản 7B cùng họ (`gte-Qwen2-7B-instruct`) là model MẠNH NHẤT trên tiếng Việt trong 18 model được test (46.05% retrieval) — dấu hiệu cả họ model này hợp tiếng Việt, bản 1.5B là lựa chọn vừa ngân sách. |
| **`jinaai/jina-embeddings-v4`** | **3.754.885.248 (~3,75 tỷ)** — dưới trần nhưng CHỈ CÒN ~245 triệu dư, gần sát 4B | 2048 (Matryoshka tới 128) | 32.768 token | `prompt_name="query"`/`prompt_name="passage"` qua sentence-transformers | **Qwen Research License** (phi thương mại — model card có sửa lại từ nhãn cc-by-nc-4.0 trước đó; CẦN đọc kỹ văn bản license gốc, đối chiếu với quy định BTC trước khi dùng) | Thiết kế chính cho multimodal/tài liệu ảnh (kiểu ColPali), chế độ text vẫn dùng được nhưng **hỗ trợ tiếng Việt CHƯA XÁC NHẬN** (chỉ ghi chung chung "30+ ngôn ngữ", không nêu tên tiếng Việt cụ thể). Rủi ro cao hơn 2 model trên — cân nhắc sau. |
| **`intfloat/multilingual-e5-large-instruct`** | ~560M (suy ra từ kiến trúc XLM-R-large, chưa xác minh trực tiếp qua API) | 1024 | 512 token (ngắn hơn hẳn — cần lưu ý nếu chunk pháp luật dài) | Query: `"Instruct: {task}\nQuery: {query}"`; Passage: không cần | MIT | Cùng họ E5 với model cũ đã bị thay thế (`multilingual-e5-base`), bản lớn hơn. VN-MTEB: retrieval 40.88% — cao hơn AITeamVN v1 (34.18%) trong cùng bộ test đó (⚠️ điểm benchmark này KHÔNG khớp với kết quả thực tế đo được trên data cuộc thi — AITeamVN v1 pretrained cho Recall LOCAL 0.9267 và Score PUBLIC 0.9058, rất mạnh — benchmark VN-MTEB tổng quát không nhất thiết phản ánh đúng domain pháp luật cụ thể này). Vẫn chỉ dùng ~14% ngân sách — chưa tận dụng hết dư địa 4B. |
| **`mainguyen9/vietlegal-e5`** | 560M (base `multilingual-e5-large`) | 1024 (Matryoshka tới 128) | 512 token | **Bắt buộc `"query: "` và `"passage: "` prefix** (ghi rõ trong model card — đúng kiểu lỗi từng gặp 12/8, cẩn thận) | Apache 2.0 | Fine-tune CHUYÊN pháp luật VN, tự công bố NDCG@10 0.7229 trên Zalo AI Legal Text Retrieval (+5.69 so với e5-large gốc) — nhưng lượt tải thấp (~118/tháng), cần tự kiểm chứng lại, không tin số liệu tự công bố 100%. |
| `truro7/vn-law-embedding` | ~0.3B (base `multilingual-e5-base`) | 128 (Matryoshka) | chưa rõ | **CHƯA ghi rõ trong model card — chưa xác minh, cẩn thận trước khi dùng** | Apache 2.0 | Chuyên pháp luật VN nhưng lượt dùng rất thấp (~237/tháng), số liệu khiêm tốn (Acc@1 62.3%) — không rõ ràng tốt hơn việc tự fine-tune base lớn hơn. |
| `dangvantuan/vietnamese-document-embedding` | ~0.3B (base `gte-multilingual-base`) | 768 | 8192 | **CHƯA ghi rõ — chưa xác minh** | Apache 2.0 | Benchmark công bố chỉ có STS/semantic-similarity, KHÔNG có số liệu retrieval — chưa rõ có hợp bài toán này không, cần tự đo trước khi tin. |
| `AITeamVN/Vietnamese_Embedding_v2` | **567.754.752 (~0,568 tỷ, đã xác minh qua API)**, base `BAAI/bge-m3` | 1024 | 2048 token (query train ở 256, passage ở 2048) | **KHÔNG cần prefix** — encode thẳng text, xác nhận qua code mẫu trong model card | Apache 2.0 | So với v1 (`AITeamVN/Vietnamese_Embedding`, model ĐANG DÙNG chính thức, xem mục trên): cải thiện chính là quy mô dữ liệu train (1,1M so với 300K triplet) để tổng quát rộng hơn, **KHÔNG phải cải thiện riêng cho domain pháp luật** — trên benchmark pháp luật tự công bố trong chính model card, v1 nhỉnh hơn nhẹ v2 (Acc@1 72.74% so với 72.62%). Cả 2 vẫn chỉ dùng ~14% ngân sách. |

**Gợi ý thứ tự thử tiếp theo (cập nhật 2026-08-16 rạng sáng):** cả 2 hướng "học thêm" (fine-tune
AITeamVN, dòng #12) VÀ "dùng model tổng quát mạnh hơn" (`gte-Qwen2-1.5B-instruct`, dòng #13) đều đã
thất bại — `AITeamVN/Vietnamese_Embedding` PRETRAINED (Score PUBLIC 0.9058) vẫn là cấu hình chính
thức. **Ưu tiên hàng đầu bây giờ chuyển sang hướng hybrid retrieval** (rẻ hơn, không cần train/encode
lại): (1) thử lại reranker (`BAAI/bge-reranker-v2-m3` pretrained) trên baseline AITeamVN hiện tại —
CHƯA test lại kể từ khi đổi baseline, (2) chốt dùng `bm25-weight=0.1` (LOCAL tốt nhất đã sweep, dòng
#11b) cho lần nộp tiếp theo. Việc thử thêm model embedding khác (`Qwen/Qwen3-VL-Embedding-2B`,
`mainguyen9/vietlegal-e5`...) hạ độ ưu tiên xuống — 2/2 lần thử "model lớn hơn/mạnh hơn benchmark
tổng quát" trong dự án này đều KHÔNG cải thiện trên domain hẹp của cuộc thi, nên hướng này còn nhiều
rủi ro tốn thời gian mà chưa chắc cải thiện. **KHÔNG dùng `Qwen/Qwen3-Embedding-4B`** (xem cảnh báo
vượt trần ở trên). `CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3` vẫn đang chờ Hoàng tự xác minh
quy ước prefix (model card gated) nếu muốn thử sau này, nhưng cũng không còn là ưu tiên cao.

## ✅ Candidate đã test — kết quả MẠNH, ĐÃ XÁC NHẬN PUBLIC (2026-08-15)

| Model | Tham số | Prefix/prompt | Kết quả LOCAL (`val_split_20`) | Kết quả PUBLIC (BTC) | Trạng thái |
|---|---|---|---|---|---|
| [`AITeamVN/Vietnamese_Embedding`](https://huggingface.co/AITeamVN/Vietnamese_Embedding) (base `BAAI/bge-m3`, ~568M, 1024 chiều) | 567.754.752 (~0,568 tỷ, đã xác minh qua API) | **KHÔNG cần prefix** (`--query-prefix=""` `--passage-prefix=""`), `--max-seq-length 2048` — đã chạy thực tế thành công 2026-08-14 khuya | Dense-only 0.9200/0.1961; Hybrid (bm25-weight=0.4) = Final **0.9267/0.1973** — PRETRAINED, CHƯA fine-tune | **0.9058** (`submission_v4_aiteamvn_pretrained.zip`, ID 889673, 2026-08-15 12:03) — +0.0586 so với best public trước đó | **ĐANG DÙNG chính thức** (thay thế `bqbbao6`). CHƯA fine-tune. Xem `submission-log-task1.md` dòng #11 để biết việc cần làm tiếp (fine-tune + sweep lại bm25-weight riêng). |

## 🔎 Candidate đã thử nghiệm nhưng KHÔNG áp dụng (embedding/reranker)

| Model | Vai trò thử | Ngày test | Kết quả | Kết luận |
|---|---|---|---|---|
| [`AITeamVN/Vietnamese_Reranker`](https://huggingface.co/AITeamVN/Vietnamese_Reranker) (fine-tune từ `bge-reranker-v2-m3` trên Zalo Legal 2021, ~568M) | Thay reranker | 2026-08-14 | Recall 0.8179 (dòng #7) — THUA cả `reranker-ft` (0.8624) VÀ bản `bge-reranker-v2-m3` pretrained chưa tune gì (0.8324, dòng #8) | Dấu hiệu negative transfer từ Zalo Legal 2021, không tổng quát tốt sang corpus cuộc thi. Đã đóng hướng thử nghiệm này. |
| `BAAI/bge-reranker-v2-m3` (PRETRAINED, chưa fine-tune) — thử LẠI trên baseline `AITeamVN/Vietnamese_Embedding` mạnh (khác dòng #8, vốn test trên baseline `bqbbao6` yếu hơn) | Rerank sau hybrid | 2026-08-16 | Recall LOCAL giảm từ 0.9267 (không rerank, dòng #11) xuống **0.8660** (dòng #14) — giảm **-0.0607**, tương đối LỚN HƠN mức giảm đã thấy trên `bqbbao6` (-2,1%) | **XÁC NHẬN LẠI (lần 2, trên baseline khác hẳn)** kết luận reranker làm giảm điểm — không phải hiện tượng riêng của baseline yếu. **ĐÃ ĐÓNG HẲN hướng dùng reranker cho Task 1** — pipeline chính thức tiếp tục KHÔNG rerank. |
| Toàn bộ reranker (kể cả `reranker-ft-full` đã fine-tune riêng) | Bước rerank sau hybrid | 2026-08-14 | `--stage-eval` cho thấy hybrid_fused (TRƯỚC rerank) 0.8791/0.1871 CAO HƠN final (SAU rerank) 0.8610/0.1835 | Reranker đang đẩy nhầm document đúng ra khỏi top-5 — đã BỎ reranker khỏi cấu hình chính thức (xem mục "Cấu hình LOCAL tốt nhất" ở trên), code vẫn giữ lại trong repo. |
| `bqbbao6/VN_legal_embedding_512` (fine-tune từ `intfloat/multilingual-e5-base`, ~278M) | Embedding chính | 2026-08-12 → 2026-08-14 | Best public từng xác nhận: 0.8472 (dòng #9); dòng #10 (bm25-weight=0.4) Recall LOCAL 0.8919, chưa nộp public riêng | Bị `AITeamVN/Vietnamese_Embedding` (pretrained, CHƯA fine-tune) vượt qua cả LOCAL (+0.0348) lẫn PUBLIC (+0.0586) — đã chính thức thay thế, xem mục trên. |
| `AITeamVN/Vietnamese_Embedding` **ĐÃ fine-tune** trên TOÀN BỘ `train.json` cuộc thi (fp16, `--max-seq-length 2048`, 3 epoch) — `models/vietnamese-embedding-aiteamvn-ft-full` | Embedding chính (thử thay bản pretrained) | 2026-08-15 | Không đo được LOCAL sạch (train dùng cả val_split_20 nên không còn tập giữ lại) | **0.8922** (`submission_v5_aiteamvn_ft.zip`, ID 889985, 2026-08-15 18:22) — **THẤP HƠN bản pretrained -0.0136** | Catastrophic forgetting/overfitting nghi ngờ — model đã train sẵn trên 1.1M triplet nội bộ, fine-tune thêm trên ~7-8K cặp hẹp của cuộc thi làm HẠI thay vì giúp (khác hẳn `bqbbao6`, nơi fine-tune chắc chắn giúp). **ĐÃ ĐÓNG hướng thử nghiệm này** — quay lại dùng bản PRETRAINED (dòng #11) làm chính thức. Song song với phát hiện tương tự ở `AITeamVN/Vietnamese_Reranker` (cũng negative transfer khi tune thêm trên data hẹp) — có thể là đặc điểm chung của họ model AITeamVN đã train kỹ từ đầu. |
| [`Alibaba-NLP/gte-Qwen2-1.5B-instruct`](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct) (PRETRAINED, KHÔNG fine-tune, ~1,776 tỷ tham số ≈ 3x AITeamVN) | Thay embedding chính bằng model lớn hơn (ngân sách 4B còn dư nhiều) | 2026-08-16 rạng sáng | Đo LOCAL sạch trên `val_split_20` (model pretrained, không train trên data cuộc thi nên không rò rỉ): Recall **0.8864**, Precision 0.1879 — ⚠️ đo với `--max-seq-length 512` (phải giảm dần từ 2048→1024→512 do chạy quá chậm, xem chi tiết) | Không nộp public (kết quả không kết luận được, không đáng tốn lượt nộp) | ⚠️ **KHÔNG THỂ KẾT LUẬN CHẮC CHẮN (confounded bởi truncation)** — check truncation (2026-08-16, `scripts/check_truncation_gte_qwen2.py`) xác nhận **21,63% chunk (76.002/351.394) vượt quá 512 token**, bị cắt cụt mất thông tin (ở max-seq-length 1024 chỉ 0,5%, gần sạch). Chênh lệch -0.0403/-0.0439 so với AITeamVN CÓ THỂ một phần đến từ truncation, không hẳn model yếu hơn. Đo công bằng hơn cần ~21h GPU (đo trước đó, batch=32/max-seq=1024) — **quyết định (2026-08-16, Hoàng): TẠM đóng lại, không chạy lại ngay**, quay về dùng AITeamVN PRETRAINED + `bm25-weight=0.1` (dòng #11b) làm chính thức. **Ý tưởng dự phòng: thuê server/cloud GPU để thử lại ở max-seq-length cao hơn sau này nếu có ngân sách/nhu cầu** — chưa lên kế hoạch cụ thể. Gặp nhiều trục trặc tích hợp trước khi đo được (transformers v5 phá `rope_theta`, xoá `get_usable_length` — bản dùng được: `transformers==4.52.0`; không có cách né `--trust-remote-code` vì model cần attention 2 chiều). Xem chi tiết đầy đủ trong `submission-log-task1.md` dòng #13. |

## 🔲 Candidate đang chờ test — embedding cụ thể cho pha so sánh hiện tại

| Model | Vai trò | Lý do đáng thử | Trạng thái |
|---|---|---|---|
| [`CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3`](https://huggingface.co/CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3) | Thay embedding | Embedding chuyên pháp lý tiếng Việt khác nguồn, đáng so sánh chéo | ⚠️ Model card GATED trên HuggingFace — Hoàng cần tự xác minh quy ước prefix/prompt trước khi test |
| `Qwen/Qwen3-VL-Embedding-2B` | Thay embedding | Xác minh tham số thật 2,13 tỷ (an toàn dưới trần), kiến trúc RoPE — có cơ sở nghiên cứu ủng hộ cho tiếng Việt. ⚠️ Sau kết quả THẤP HƠN của `gte-Qwen2-1.5B-instruct` (cùng lý luận "benchmark tổng quát mạnh + RoPE"), độ ưu tiên của hướng "đổi sang model lớn hơn" nói chung đã giảm — chỉ đáng thử nếu còn dư thời gian, không phải ưu tiên hàng đầu nữa. | Mới thêm 2026-08-14, chưa chuẩn bị lệnh chạy |
| `mainguyen9/vietlegal-e5` | Thay embedding | Chuyên pháp luật VN, tự công bố NDCG@10 tốt trên Zalo Legal — cần bắt đúng prefix `"query: "`/`"passage: "` | Mới thêm 2026-08-14, chưa chuẩn bị lệnh chạy |
| `truro7/vn-law-embedding` | Thay embedding | Chuyên pháp luật VN, chưa xác minh kỹ tên gọi/model card, chưa rõ prefix | `danh-sach-model-tham-khao.md` (2026-08-10) |
| `mainguyen9/vietlegal-harrier-0.6b` | Thay embedding | Chuyên pháp luật VN (~0.6B) | `danh-sach-model-tham-khao.md` (2026-08-10) |

❌ **KHÔNG dùng** `Qwen/Qwen3-Embedding-4B` (tham số thật ~4,022 tỷ, VƯỢT trần), `Qwen/Qwen3-Embedding-8B`
(~7,57 tỷ), `BAAI/bge-multilingual-gemma2` (~9 tỷ) — đều đã xác minh vượt xa trần 4B, không cần
cân nhắc thêm.

## 🔲 Candidate cho Task 2 (LegalQA — chưa bắt đầu, cần model sinh/LLM)

| Model | Tham số (~) | Ghi chú |
|---|---|---|
| `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2` | 1.7B | Qwen3 fine-tune GRPO chuyên pháp luật VN — candidate hàng đầu cho Task 2 |
| `ntphuc149/ViLegalQwen2.5-1.5B-Base` | 1.5B | Chuyên pháp luật VN |
| `ntphuc149/ViLegalQwen3-1.7B-Base` | 1.7B | Chuyên pháp luật VN |
| `AITeamVN/Vi-Qwen2-1.5B-RAG` | 1.5B | Qwen2 fine-tune cho RAG tiếng Việt |

⚠️ Nếu Task 1 + Task 2 tính CHUNG 1 trần 4B tham số (cần hỏi lại BTC để chắc chắn), pipeline Task 1
hiện tại (~568M, đã bỏ reranker, dùng AITeamVN) + 1 model Task 2 (1.5-1.7B) vẫn nằm dưới trần rất
thoải mái (~2.1-2.3B) — thậm chí còn dư nhiều hơn trước (lúc trước tính cả reranker ~846M).

⚠️ **Lưu ý số liệu chưa xác minh 100% trong bảng "Candidate mạnh" ở trên:** một số tham số (đặc biệt
`gte-Qwen2-1.5B-instruct`, `multilingual-e5-large-instruct`) được tính suy ra từ `config.json`/kiến
trúc tương đương, KHÔNG lấy trực tiếp từ trường `safetensors.total` của HuggingFace API như các model
khác — nên chạy lại `sum(p.numel() for p in model.parameters())` sau khi tải model về để chốt số liệu
cuối cùng trước khi báo cáo BTC hoặc viết Model Card chính thức.

Xem danh sách đầy đủ hơn (đã tổng hợp từ các đội khác + tham khảo) tại `danh-sach-model-tham-khao.md`
trong project Claude — file đó liệt kê nhiều lựa chọn hơn nhưng CHƯA phân loại theo trạng thái
đang-dùng/đã-thử/chưa-thử như file này.
