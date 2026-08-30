# Log thử nghiệm & submission — Task 1 (LegalIR)

Bản rút gọn (2026-08-23) — chỉ giữ các thử nghiệm chính, quyết định cuối và bài học có giá trị tái sử
dụng. Toàn bộ nhật ký chi tiết (log debug, breakdown từng bước, các bug đã sửa) được giữ trong lịch sử
chỉnh sửa của doc này trên claude.ai nếu cần tra lại.

**Các hướng CÒN LẠI đang chạy dở, chia việc cho Hoàng/Kiên/Giang, xem `docs/phan_cong_thu_nghiem_task1.md`**
— khi 1 hướng ở đó có kết luận, kết quả sẽ được chuyển về bảng log chính bên dưới để giữ lịch sử liền mạch.

## 🏆 Cấu hình tốt nhất hiện tại

**Score PUBLIC 0.9208** (ID 896875, nộp 2026-08-22) — ensemble RRF 3 nhánh: `AITeamVN/Vietnamese_Embedding`
(weight 1.0) + `mainguyen9/vietlegal-harrier-0.6b` (weight 0.2) + BM25 (weight 0.1), `rrf_k=60`, gộp
chunk→document bằng max-score, top-k=5, KHÔNG rerank, KHÔNG hậu-xử-lý dedup. Tổng ~1,16 tỷ tham số
(dưới hạn mức 4 tỷ). Chi tiết đầy đủ + lệnh tái lập ở phần trả lời tiếp theo.

## Bảng log các lần thử nghiệm chính

| # | Ngày | Thay đổi so với dòng trước | Recall LOCAL | Score PUBLIC | Kết luận |
|---|------|------|:---:|:---:|---|
| 4 | 08-12 | Fine-tune embedding (80%) + reranker hard-negative, hybrid BM25+dense | 0.8624 | — | Mốc khởi đầu pipeline hybrid |
| 9 | 08-14 | Bỏ reranker (RRF hybrid thuần) | 0.8791 | 0.8472 | Reranker đang làm giảm điểm — bỏ reranker tốt hơn |
| 10 | 08-14 | Sweep `bm25-weight` → 0.4 | 0.8919 | — | Cải thiện nhỏ, không nộp riêng |
| **11** | 08-15 | Đổi embedding sang `AITeamVN/Vietnamese_Embedding` pretrained (chưa fine-tune) | 0.9267 | **0.9058** | **Cú nhảy lớn nhất dự án** (+0.0586 public) |
| **11b** | 08-16 | Sweep lại `bm25-weight` → 0.1 | 0.9303 | **0.9173** | Đỉnh mới, là baseline chính thức trong thời gian dài |
| 12 | 08-15 | Fine-tune AITeamVN trên full `train.json` | — | 0.8922 | THẤP HƠN pretrained — catastrophic forgetting. ĐÓNG |
| 13 | 08-16 | Đổi sang `gte-Qwen2-1.5B-instruct` (lớn hơn ~3x) | 0.8864 | — | Thấp hơn, nhưng confound bởi truncation (max-seq-length 512) — không kết luận chắc. TẠM đóng |
| 14 | 08-16 | Thêm lại reranker `BAAI/bge-reranker-v2-m3` trên baseline AITeamVN | 0.8660 | — | Giảm mạnh (-0.0607) — xác nhận lần 2 reranker có hại. ĐÓNG HẲN |
| 15/17 | 08-16/17 | Entity-boost injection (bơm văn bản trùng tên thực thể) | 0.9310 (+0.0007) | 0.9163 (-0.0010) | Local dương nhưng public NGƯỢC CHIỀU — tín hiệu dựa trên quá ít case (2/1398). ĐÓNG |
| 16 | 08-17 | Đổi embedding sang `vietlegal-harrier-0.6b` (chuyên luật) đứng RIÊNG | 0.9040 | 0.8775 | Thua AITeamVN cả 2 độ đo khi đứng riêng — nhưng sau này thành nguồn bổ trợ tốt cho ensemble (xem #21) |
| 18 | 08-17 | Đổi embedding sang `AITeamVN/Vietnamese_Embedding_v2` | 0.9131 | — | Thấp hơn v1 ở mọi giai đoạn, khớp benchmark chính thức của model. ĐÓNG |
| 19 | 08-17 | Đổi embedding sang `BAAI/bge-m3` gốc | 0.8847 | — | Thấp hơn cả v1 lẫn v2. ĐÓNG |
| 20 | 08-21 | **Ensemble v1+harrier(0.8)+bm25, rrf_k=8** | 0.9360 (+0.0065, p=0.033) | 0.9160 | Local RẤT chắc (kể cả xác nhận trên holdout 5602 câu độc lập) nhưng vẫn THUA public — harrier tổng quát hoá kém hơn v1 2 lần, trọng số nặng nhập luôn điểm yếu đó |
| **21** | 08-22 | **Ensemble v1+harrier(0.2, nhẹ tay)+bm25, rrf_k=60** | — | **0.9208** | 🏆 **BEST HIỆN TẠI.** Chọn trọng số nhẹ theo suy luận từ khoảng cách local→public của từng model, KHÔNG theo local argmax |
| — | 08-22 | Reranker làm nguồn RRF phụ (không thay thế) — 4 biến thể reranker khác nhau | Giảm đơn điệu mọi trọng số | — | ĐÓNG DỨT ĐIỂM hướng reranker cho Task 1 (4/4 phép thử độc lập đều âm) |
| 37 | 08-23 | Dedup+backfill hậu-xử-lý 27 nhóm tài liệu trùng lặp | -0.0025 đến -0.0039 (3 phép đo độc lập, đều âm) | — | ĐÓNG |
| 38 | 08-30 | **Thêm nhánh `Qwen/Qwen3-Embedding-0.6B` vào ensemble (v1+harrier+bm25)** | 0.9294 (không đổi ở w=0, giảm đơn điệu ở w>0: 0.9289 tại w=0.2, 0.9111 tại w=1.2) | — | Giảm đơn điệu giống gte-Qwen2 — kiến trúc decoder pooling không bổ trợ được cho ensemble hiện tại. ĐÓNG |
| 39 | 08-30 | **Tái kiểm tra nộp Public Test (v1=1.0 + harrier=0.2 + bm25=0.1, rrf_k=60)** | — | **0.9188** (Precision 0.1976, Recall 0.91875) | Xác nhận pipeline nộp bài độc lập hoạt động chuẩn xác, điểm số ổn định (~0.919 - 0.921). |


## Các phát hiện phương pháp luận quan trọng

**Ngưỡng tin cậy local→public:** chênh lệch LOCAL dưới ~0.01 không mang thông tin đáng tin cậy về
PUBLIC — đã kiểm chứng trực tiếp bằng dòng #20 (local +0.0065 có ý nghĩa thống kê, tái lập trên holdout
5602 câu độc lập, vẫn thua public). Từ dòng #20, quy tắc áp dụng: chỉ tin tưởng thay đổi local khi độ
lớn ≥0.02 (mức mà 3 vị trí biên của bảng so sánh local/public khớp thứ hạng); với thay đổi nhỏ hơn,
quyết định phải dựa trên khoảng cách local→public riêng của TỪNG thành phần (model có gap lớn hơn thì
chỉ nên dùng trọng số nhỏ) — đây là cách #21 chọn đúng trọng số harrier=0.2 sau khi #20 (harrier=0.8)
thất bại.

**Cơ chế RRF multi-branch ensemble:** `score(d) = Σ w_i / (rrf_k + rank_i(d))`. Cửa sổ trọng số hữu
ích RẤT HẸP và phụ thuộc `rrf_k` — với `rrf_k=60`, trọng số dưới ~0.8 gần như vô tác dụng, ~1.0 mới đủ
để cứu tài liệu bị model chính bỏ sót, ~1.5 gây "thảm hoạ" (nguồn phụ chiếm toàn bộ top-5). Hạ `rrf_k`
làm đường cong dốc hơn, giảm trọng số cần thiết.

**Model "domain-specific" không tự động thắng model tổng quát:** lặp lại 2 lần trong dự án — reranker
`AITeamVN/Vietnamese_Reranker` (đã tune cho pháp lý) thua `BAAI/bge-reranker-v2-m3` vanilla; embedding
`vietlegal-harrier` (chuyên luật) thua `AITeamVN/Vietnamese_Embedding` (tổng quát tiếng Việt) khi đứng
riêng. Giả thuyết: quy mô + chất lượng pretrain gốc quan trọng hơn việc fine-tune hẹp trên 1 domain cụ
thể. Tuy nhiên model thua khi đứng riêng vẫn có thể là nguồn bổ trợ tốt cho ensemble nếu đủ khác biệt
kiến trúc (harrier: Qwen decoder/last-token pooling, khác hẳn v1: XLM-R encoder) — đo bằng
"complementarity" (% tài liệu model chính bỏ sót mà model phụ cứu được), ngưỡng >30-40% coi là đáng thử.

**Fine-tune thêm AITeamVN v1 nhất quán phản tác dụng** — 3 lần thử, 3 nguyên nhân kỹ thuật khác nhau,
đều thất bại:
1. Full `train.json`, LR mặc định 2e-5, 3 epoch → catastrophic forgetting (PUBLIC 0.8922 < 0.9058).
2. TripletLoss margin=5 trên embedding đã chuẩn hoá L2 (khoảng cách Euclid tối đa chỉ 2) → gradient
   không bao giờ tắt → sụp đổ biểu diễn hoàn toàn (Recall 0.92→0.0046, 9 giờ GPU mất trắng).
3. Đổi sang MultipleNegativesRankingLoss (không margin, không sụp đổ) nhưng vẫn Recall -0.1080 — do
   hard negative đào từ retriever mạnh (top-ranked của v1) dính ~12% false negative (tài liệu gần
   trùng nội dung với đáp án đúng, do 27 nhóm tài liệu trùng lặp + GT thiếu sót trong data gốc).

Kết luận: model AITeamVN gốc (1,1 triệu triplet pretrain) đã ở gần điểm tối ưu cục bộ tốt cho domain
này — cả fine-tune thêm lẫn đổi sang model tổng quát mạnh hơn (`gte-Qwen2-1.5B-instruct`) đều không
vượt qua được nó trong mọi thử nghiệm đã làm.

**Reranker đóng dứt điểm sau 4 phép thử độc lập:** thay thế thứ hạng RRF hoàn toàn (2 biến thể model)
và làm nguồn đóng góp phụ trong RRF thay vì thay thế (2 biến thể khác) — cả 4 đều làm giảm điểm, biến
thể "đóng góp phụ" (giả thuyết mạnh nhất còn lại) vẫn giảm đơn điệu ngay từ trọng số nhỏ nhất.

**Data quality:** phát hiện + định lượng được 3 vấn đề trong dữ liệu gốc BTC qua điều tra trực tiếp
(`device_bash`, không cần GPU): 20/8.532 tài liệu (~0,23%) có `passage` rỗng (nguyên nhân: paywall khi
crawl, đã xác nhận qua `WebFetch`); 27 nhóm/54 tài liệu trùng lặp nội dung (crawl 2 lần từ 2 URL khác
nhau của cùng 1 văn bản); 13/47 câu hỏi về lương theo chức danh nghi ngờ ground truth chỉ gắn văn bản
khung chung thay vì văn bản chuyên biệt. Đã báo cáo đầy đủ cho Kiên (`bao-cao-data-quality-cho-Kien.md`)
— team KHÔNG tự sửa GT/crawl thêm vì vi phạm quy định mục 7 (cấm gán nhãn thủ công/thu thập dữ liệu
ngoài). 20 tài liệu rỗng đã được vá bằng fallback passage (số hiệu + lĩnh vực trích từ URL, không cần
dữ liệu ngoài) — an toàn nhưng tác động đo được ~0 (chỉ chạm 3/1398 câu hỏi val).

**⚠️ Đã xác nhận (2026-08-23): re-crawl lại 20 tài liệu rỗng KHÔNG được làm**, kể cả bởi Kiên với tài
khoản Pro riêng — vẫn tính là "thu thập dữ liệu ngoài bộ dữ liệu chính thức BTC phát hành" (quy định
mục 7), không phân biệt theo AI làm hay ai có quyền truy cập nguồn gốc. Hướng xử lý DUY NHẤT hợp lệ cho
20 tài liệu này là fallback passage đã vá ở trên — không còn việc gì cần làm thêm cho vấn đề này.

## Hướng đã đóng (không cần thử lại trừ khi có ý tưởng mới)

Fine-tune embedding (3 lần, 3 cơ chế thất bại khác nhau) · reranker mọi hình thức (4 phép thử độc lập)
· đổi sang embedding model khác (`gte-Qwen2-1.5B-instruct`, `AITeamVN v2`, `bge-m3` gốc, `vietlegal-harrier`
đứng riêng — đều thua v1) · entity-boost injection · citation-graph injection · dedup+backfill hậu-xử-lý
· sweep cách gộp chunk→document (max vẫn tốt nhất) · sweep lưới `dense-top-n`/`bm25-top-n`/`rrf-k`
(mặc định 100/100/60 đã gần tối ưu, chênh lệch tốt nhất chỉ +0.0004).

**`gte-Qwen2-1.5B-instruct` (~1,543 tỷ tham số, ~3x v1) làm nhánh thứ 4 trong ensemble (2026-08-23,
`scripts/run_ensemble_add_gteqwen2.ps1`, dùng lại cache corpus có sẵn từ dòng #13 nên không tốn GPU
encode lại):** giảm ĐƠN ĐIỆU trên toàn bộ lưới trọng số đã thử (0.05 → 1.2), không có vùng ngọt nào —
khác hẳn harrier (có cửa sổ hẹp mang lại lợi ích). Baseline `gteqwen2=0` khớp đúng 0.9319 (bảo hiểm
đã xác nhận không có lỗi đo). **Model "nhiều tham số hơn" không tự động là hướng còn bỏ ngỏ** — dù
standalone complementarity đo được 25,4% (gần ngưỡng 30-40%), RRF không khai thác được lợi ích đó
thành cải thiện thật, kể cả ở trọng số rất nhỏ. **ĐÓNG, không đáng đầu tư ~21h GPU** để encode lại
corpus ở max-seq-length công bằng hơn (512→1024, tránh confound truncation) — xu hướng âm bắt đầu
ngay từ trọng số nhỏ nhất, không phải dấu hiệu "chưa dò đúng cửa sổ" mà nhiều khả năng là bản thân
tín hiệu (dù bị nhiễu bởi truncation hay không) không bổ trợ được cho v1+harrier+bm25 đã có.

## Hướng còn lại (chưa thử hoặc ưu tiên thấp)

Lọc hard-negative theo ngưỡng cosine trước khi fine-tune lại (kỳ vọng thấp — cần +0.0165 để vào Top 10
trong khi lần fine-tune gần nhất đang ở -0.108) · `itdainb/PhoRanker` hoặc `Qwen/Qwen3-Reranker-0.6B`
(ưu tiên thấp, pattern reranker domain-specific đã thua 2 lần) · `CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3`
(gated, cần Hoàng tự xác minh prefix — không to hơn v1 nhưng là kiến trúc Qwen3 + domain luật, chưa
thử) · query expansion (PRF/RM3) cho nhánh BM25 — cổ điển, chưa thử, không cần GPU, chỉ dùng corpus
chính thức nên hợp lệ · thử nghiệm chunking khác (kích thước/overlap) · cập nhật
`scripts/allowed-models.md` cho harrier trong ensemble chính thức · chờ Kiên xem lại GT cho 13 câu hỏi
lương (re-crawl 20 tài liệu rỗng KHÔNG còn là việc cần làm — xem cảnh báo tuân thủ ở mục data quality).

## Ghi chú tuân thủ

Tổng tham số ensemble hiện tại ~1,16 tỷ (v1 568M + harrier 596M), dưới hạn mức 4 tỷ (quy định mục 5).
Thêm model mới vào pipeline chính thức cần đề xuất cho BTC trước 10 ngày so với hạn chót private test
(quy định mục 7) — `allowed-models.md` cần cập nhật khi chốt dùng ensemble cho vòng private.
