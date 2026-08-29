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
| — | 08-22/23 | Fine-tune lại v1 (2 lần: TripletLoss margin=5, rồi MultipleNegativesRankingLoss) | Sụp đổ (0.0046), rồi -0.1080 | — | 3/3 lần fine-tune AITeamVN đều thất bại (xem mục "Fine-tune" bên dưới) |
| — | 08-27 | **Lần 4:** MNRL + lọc hard-negative theo ngưỡng cosine 0,85 (loại candidate gần trùng đáp án trước khi mining, khắc phục đúng nguyên nhân đã đo của Lần 3) | -0.1189 (quick_eval subset, cùng phép đo với Lần 3) | — | Lọc CHẠY ĐÚNG (412/6519 candidate bị loại, ~6,3%, cùng bậc độ lớn với ước tính 12,3% trước đó) nhưng Δ KHÔNG cải thiện — còn hơi tệ hơn Lần 3. Kết luận: false-negative contamination KHÔNG PHẢI nguyên nhân chính của thất bại fine-tune. **ĐÓNG HẲN hướng fine-tune (4/4 lần thất bại, 4 cơ chế khác nhau đã loại trừ)** |
| — | 08-23 | Dedup+backfill hậu-xử-lý 27 nhóm tài liệu trùng lặp | -0.0025 đến -0.0039 (3 phép đo độc lập, đều âm) | — | ĐÓNG |
| — | 08-28 | Thêm nhánh `BAAI/bge-m3` SPARSE (lexical học được, tự cài đắp lại sparse head từ `sparse_linear.pt`, KHÔNG dùng thư viện FlagEmbedding) làm nguồn thứ 4 RRF | Đỉnh 0.9346 tại `m3sparse=0.04-0.05` (Δ+0.0027), lưới mịn (0→0.08, bước 0.01) cho đường cong TRƠN MỘT ĐỈNH — xác nhận tín hiệu THẬT, không phải nhiễu | — | Tín hiệu thật nhưng QUÁ NHỎ để tin theo quy tắc đã chốt (cần ≥0.01, lý tưởng ≥0.02); tiền lệ entity-boost (local +0.0007 → public -0.0010) cho thấy rủi ro đổi dấu ở mức Δ nhỏ này là có thật. KHÔNG đóng hẳn (khác gte-Qwen2) — GIỮ làm ứng viên kết hợp: thử cộng `m3sparse≈0.04` vào bất kỳ cấu hình mới nào từ B1/B2/hướng #1 (score-based fusion) trước khi nộp, xem cải thiện nhỏ có cộng dồn vượt ngưỡng không. **[CẬP NHẬT 08-28 tối: đã nộp public (m3sparse=0.04 trong RRF, xác minh đúng tham số qua SHA-256/tái tạo độc lập) → 0.9183, Δ−0.0025. ĐỔI DẤU. ĐÓNG HẲN — xem mục "Danh mục đã nộp" ở phần chẩn đoán pipeline. (Riêng bản kết hợp CombSUM+m3sparse KHÔNG có dữ liệu public thật — xem đính chính "sai lệch #2" cùng mục.)]** |
| — | 08-28 | **Hướng #1: fuse theo SCORE (CombSUM)** thay RRF, giữ v1=1.0 neo, sweep harrier+bm25 (minmax norm) | Coarse (0→2 / 0→1): đỉnh 0.9368 tại `harrier=0.75,bm25=0.25`. Fine (0.4→1 / 0.1→0.4): **CÙNG đỉnh 0.9368, đạt ở 3 tổ hợp lân cận khác nhau** (0.6/0.35 · 0.75/0.25 · 0.8/0.2) + nhiều lân cận khác trong dải 0.933-0.937 → CAO-NGUYÊN RỘNG, tín hiệu THẬT còn chắc hơn 1 đỉnh đơn lẻ. Xác minh `harrier=0,bm25=0` (CombSUM 1 nguồn v1) = 0.9200, khớp ĐÚNG với `fuse --mode rrf` cùng cấu hình (đối chiếu tay, không lệch) → code không có bug | 0.9368 (đứng riêng CombSUM, so RRF chính thức 0.9319: **Δ+0.0049**) | — | Δ **DƯỚI ngưỡng +0.01** đã chốt dù cao-nguyên rộng cho thấy đây là tín hiệu thật (không phải nhiễu) — theo đúng quy tắc, CHƯA đủ bằng chứng để nộp public. KHÔNG đóng — GIỮ làm ứng viên kết hợp giống bge-m3 sparse (Δ+0.0027): 2 cải thiện nhỏ CỘNG lại có thể vượt ngưỡng nếu chạy `--mode score` với cả m3sparse là 1 nguồn (cần thêm scores vào `bgem3_sparse.py` trước — CHƯA làm, để sau). **[CẬP NHẬT 08-28 tối: đã thêm scores cho m3sparse và nộp public 2 biến thể chuẩn hoá — minmax harrier=0.75/bm25=0.25 → 0.9198 (Δ−0.0010); zscore CÙNG điểm trọng số 0.75/0.25 (đã xác minh lại qua SHA-256/tái tạo độc lập — KHÔNG PHẢI 0.75/0.5 như ghi ban đầu, đỉnh thật của zscore chưa từng nộp) → 0.9128 (Δ−0.0080, tệ nhất mọi bài nộp). CẢ 2 ĐỀU ĐỔI DẤU. (Bản "gộp thêm m3sparse" hoá ra trùng byte-for-byte với bản minmax thuần — sweep local cho thấy m3sparse=0 chính là đỉnh, nên không phải phép thử độc lập, không tính là bằng chứng.) ĐÓNG HẲN toàn bộ hướng score-based fusion dựa trên 2 phép thử thật (minmax, zscore cùng 1 điểm) — xem mục "Danh mục đã nộp" ở phần chẩn đoán pipeline.]** ⚠️ Nhân tiện phát hiện + sửa 1 bug pre-existing thật trong `cmd_fuse` (không phải do đổi mới lần này): baseline tự động dùng điều kiện `all(v==0 for ... if 0.0 in weight_grids[name])` — khi lưới sweep KHÔNG chứa 0.0 cho bất kỳ nguồn phụ nào (đúng ca fine-sweep này), điều kiện `all([])` vô nghĩa = True cho MỌI dòng → tự chọn nhầm chính dòng tốt nhất làm baseline, in ra "Δ +0.0000" sai hoàn toàn. Đã sửa: giờ báo rõ "không có baseline hợp lệ" thay vì tự đoán liều |
| — | 08-28 | **Hướng #2: nhánh `huyydangg/DEk21_hcmute_embedding`** (PhoBERT-family, input đã tách từ, `max_seq_length=256`) làm nguồn thứ 4 RRF | Đứng riêng (RRF 1 nguồn, top-100 pool) = 0.8410 — DƯỚI CẢ gte-Qwen2 (0.8864, đã đóng) và harrier (0.9040, đang dùng thật). Fuse vào ensemble (`harrier=0.2, bm25=0.1` cố định, sweep `dek21=0→1.5`): **GIẢM ĐƠN ĐIỆU ngay từ trọng số nhỏ nhất** — 0.9319(=0)→0.9285(0.1)→0.9244(0.3)→0.9158(0.6)→0.9058(0.9)→0.9022(1.2)→0.9008(1.5). Sanity check `dek21=0`=0.9319 khớp đúng, không phải lỗi đo | 0.8410 đứng riêng; mọi trọng số fuse đều < baseline 0.9319 | — | **ĐÓNG HẲN, KHÔNG PHẢI "tạm đóng"** (khác gte-Qwen2 — model đó còn đường lùi "thử lại ở max_seq_length khác" vì 512 chỉ là lựa chọn cấu hình; DEk21 thì `max_seq_length=256` là TRẦN KIẾN TRÚC CỨNG của RoBERTa/absolute-position-embedding, không có đường lùi nào). Kịch bản giống hệt gte-Qwen2: model dense cùng phạm trù kiến trúc rộng, đứng riêng yếu hơn v1/harrier, không tìm được cửa sổ trọng số hữu ích nào, giảm đơn điệu ngay từ đầu |
| — | 08-28 | **Hướng #3: sweep `k1`/`b` của BM25Okapi** — module mới `sweep_bm25_k1b.py`, build ma trận tần suất thô 1 LẦN (không phụ thuộc k1/b, xác minh trực tiếp qua `rank_bm25.BM25Okapi._calc_idf`), mỗi combo chỉ tính lại vector K rồi transform lại (rẻ, thuần CPU) | BM25 ĐỨNG RIÊNG: coarse (`k1∈{1.0,1.2,1.5,1.8,2.2,2.6}×b∈{0.3,0.5,0.75,0.9}`) đỉnh `k1=1.5,b=0.9`→0.7609 (mặc định `k1=1.5,b=0.75`→0.7487, Δ+0.0122); fine (`k1∈{1.2,1.35,1.5,1.65,1.8}×b∈{0.85,0.9,0.95,1.0}`) đỉnh nhỉnh hơn `k1=1.5,b=0.95`→0.7616 (Δ+0.0129) — tín hiệu đứng riêng THẬT, không phải nhiễu lưới thô. Ensemble-check (thay `ranks_bm25.npz` bằng bản tuned `k1=1.5/b=0.95`, GIỮ NGUYÊN `bm25-weight=0.1` hiện có): Recall=**0.9319**, TRÙNG KHỚP TUYỆT ĐỐI baseline chính thức 0.9319 — không đổi 1 tài liệu nào trong top-5 | Δ đứng riêng +0.0129; Δ ENSEMBLE = **0.0000 tuyệt đối** | — | Đúng như dự đoán trước khi chạy (rủi ro pha loãng đã ghi rõ trong docstring script trước khi chạy): cải thiện nội bộ BM25 THẬT nhưng chỉ đóng góp vào ensemble tổng qua đúng `bm25-weight=0.1` hiện có → bị hoà loãng hoàn toàn, không đổi được top-5 cuối. **ĐÓNG** — đúng như dự đoán, đây là hướng rẻ nhất (thuần CPU, không tốn GPU) nhưng có trần cải thiện kỳ vọng THẤP NHẤT trong 3 hướng. Không cần thử combo (k1,b) khác — cơ chế pha loãng (trọng số cố định nhỏ) áp dụng như nhau bất kể chất lượng xếp hạng nội bộ BM25 tốt tới đâu; muốn khai thác được cải thiện BM25 đứng riêng này sẽ cần tăng `bm25-weight` (đã sweep, đóng — xem dòng #11b/10) hoặc dùng làm 1 nguồn trong `--mode score` (hướng #1) thay vì RRF |

## 🔬 CHẨN ĐOÁN LẠI TOÀN PIPELINE (2026-08-28) — vì sao ~15 hướng đều ra ~0

Sau khi 3 hướng cuối (fusion/DEk21/BM25 k1-b) đều không đáng kể, dừng đề xuất hướng mới và
ĐO xem lỗi còn lại thuộc LOẠI nào. Công cụ mới, thuần CPU, dùng lại cache có sẵn:
`src/evaluation/diagnose_ceiling.py`, `test_length_bias.py`, `inspect_failures.py`,
`doctype_prior.py`.

**1. Trần pool = 99.00%, và lỗi hiện tại KHÔNG phải lỗi truy hồi.**

| chỉ số | giá trị |
|---|---|
| Tài liệu đúng nằm trong pool top-100 (hợp 3 nguồn) | **99.00%** |
| Recall@5 (đang tối ưu) | 0.9319 |
| Recall@10 | 0.9614 |
| Recall@100 (trần nếu rerank HOÀN HẢO trong pool) | **0.9890** |
| Khoảng trống rerank có thể ăn | **0.0571** |

Phân loại 70 câu sai: **(A) tài liệu đúng KHÔNG có trong pool: chỉ 10 câu (14.3%)** ·
**(B) có trong pool nhưng xếp hạng 6-100: 60 câu (85.7%)**, trong đó **32 câu nằm ngay hạng
6-10**. => Pipeline KHÔNG bão hoà về truy hồi; nó bão hoà về **khả năng phân biệt trong top-10**.
Mọi hướng đã thử đều nhằm "thêm/đổi tín hiệu ngữ nghĩa", trong khi tín hiệu ngữ nghĩa đã đủ
để KÉO VỀ tài liệu đúng, chỉ không đủ để XẾP nó lên trên.

**2. Sàn nhiễu của phép đo local — lý do TOÁN HỌC khiến mọi Δ nhỏ đều "không kết luận được".**
n=1398, Recall≈0.93 → sai số chuẩn = 0.0067, khoảng tin cậy 95% = **±0.0132**. Nghĩa là khi so
2 con số Recall RỜI RẠC, ta VẬT LÝ không phân biệt nổi cải thiện < ~0.013 — quy tắc ≥0.01-0.02
đúc kết từ kinh nghiệm hoá ra là hệ quả toán học, không phải xui.
**LỐI THOÁT ĐÃ ÁP DỤNG TỪ NAY: kiểm định GHÉP CẶP (McNemar chính xác)** — đếm b = số câu
sai→đúng, c = số câu đúng→sai trên CÙNG tập câu hỏi. Nhạy hơn hẳn: b=18/c=2 có p=0.0004 dù
Δ chỉ +0.011, còn b=12/c=8 có p=0.50 dù Δ cũng dương. **Mọi so sánh A/B từ nay phải in (b,c,p),
không chỉ 2 con số Recall.** Đây có thể là lý do một số hướng "âm nhẹ" trước đây bị đọc sai.

**3. Giả thuyết "thiên lệch độ dài khi gộp max" — ĐÃ KIỂM CHỨNG VÀ BÁC BỎ.** Corpus lệch rất
mạnh (trung vị 27 chunk/doc, lớn nhất 4.087 chunk → gấp 151x), nên `max` về lý thuyết thiên vị
tài liệu dài. Nhưng đo thật: tài liệu đè có trung vị 60 chunk vs tài liệu đúng 56 chunk (tỉ lệ
1.14x), chỉ 54.3% ca là tài liệu đè dài hơn (≈ tung đồng xu), và phạt theo `-α·log(1+n_chunk)`
làm giảm ĐƠN ĐIỆU. Giả thuyết SAI, đóng.

**4b. Cơ chế CẤU TRÚC cuối cùng — "họ văn bản" — ĐÃ ĐO, DƯ ĐỊA QUÁ NHỎ, ĐÓNG**
(`src/evaluation/analyze_families.py`, 28-08). Từ các ca ở mục 4, giả thuyết: khi top-5 đã chứa 1
văn bản, "anh em cùng họ" của nó (cùng luật khác NĂM, hoặc văn bản SỬA ĐỔI nó) là ứng viên mạnh cho
slot còn lại. Đây KHÔNG phải đặc trưng học từ thói quen gán nhãn nên KHÔNG dính bẫy đã giết prior
doc_type. Đo trên 70 câu sai:

| dạng | số ca | % câu sai |
|---|---:|---:|
| (a) đáp án CÙNG HỌ với 1 doc trong top-5 (khác năm/số hiệu) | 2/70 | 2.9% |
| (b) đáp án có quan hệ SỬA ĐỔI với 1 doc trong top-5 | 6/70 | 8.6% |
| **trần trên (a hoặc b, đếm nới rộng)** | **~8/70** | **→ tối đa +0.0057 Recall NẾU sửa HOÀN HẢO 100%** |

Trần +0.0057 khi sửa hoàn hảo đã **thấp hơn sàn nhiễu ±0.0132**, chưa kể sửa thật sẽ phá một số câu
đang đúng. ĐÓNG.

**Đồng thời giết luôn ý tưởng "đa dạng hoá slot" (MMR/diversification):** top-5 có ≥2 văn bản cùng
họ (lãng phí slot) ở **10.0% số câu SAI** nhưng cũng ở **10.5% TOÀN BỘ 1398 câu** — tỉ lệ NHƯ NHAU,
tức lãng phí slot KHÔNG liên quan gì tới việc câu đó đúng hay sai. Bỏ trùng họ sẽ không cải thiện
có hệ thống. (Khác với dedup 27 nhóm trùng lặp NỘI DUNG đã thử 08-23, cũng âm.)

**4. Đọc tay 20 ca sai → nguyên nhân thật: các văn bản cạnh tranh ĐỀU đúng chủ đề, khác nhau ở
thứ embedding không thấy được.** Ví dụ: hỏi "thời hạn nộp hồ sơ tự quyết toán thuế TNCN" →
đáp án là *Luật Quản lý thuế 2019* nhưng pipeline trả về 2 *Công văn* đúng chủ đề ở hạng 1-2;
hỏi về kết luận thanh tra → trả về *Luật Thanh tra 2022* trong khi đáp án là *Luật Thanh tra
2010*; hỏi về tiếp công dân → 5 nội quy tiếp công dân của 5 cơ quan khác nhau. Đây là phân biệt
**khung vs chuyên biệt / cũ vs mới / loại văn bản**, KHÔNG phải phân biệt ngữ nghĩa.

**5. ✅ HƯỚNG MỚI CÓ KẾT QUẢ: PRIOR THEO LOẠI VĂN BẢN — kết quả CÓ Ý NGHĨA THỐNG KÊ ĐẦU TIÊN
của dự án.** Từ phát hiện #4, đo phân bố `doc_type` (có sẵn trong `corpus.jsonl`, dữ liệu chính
thức BTC — KHÔNG thu thập ngoài, hợp lệ mục 7):

| loại | % corpus | % ground truth (train) | lift |
|---|---:|---:|---:|
| **Luật** | 6.56 | **26.10** | **3.98x** |
| Nghị định | 55.09 | 52.49 | 0.95x |
| Thông tư | 11.78 | 8.81 | 0.75x |
| Quyết định | 14.46 | 7.77 | 0.54x |
| **Công văn** | 5.11 | **1.32** | **0.26x** |
| Chỉ thị | 0.55 | 0.15 | 0.27x |

Prior ỔN ĐỊNH giữa train (6.116 nhãn) và val (1.521 nhãn): Luật 26.10% vs 27.48%, Công văn
1.32% vs 0.72% → không phải nhiễu chia tập. Áp dụng
`score'(d) = score(d) + β·log(P(type|đáp án, TRAIN)/P(type|corpus))`, **ước lượng CHỈ trên
train_split_80, đo CHỈ trên val_split_20** (script có chốt chặn cứng, từ chối chạy nếu 2 file
trùng nhau):

| β | Recall@5 | Δ | b (cứu) | c (hỏng) | p (McNemar) |
|---:|---:|---:|---:|---:|---:|
| 0 (đối chứng) | 0.9319 | — | 0 | 0 | — |
| 0.0002 | 0.9335 | +0.0017 | 3 | 1 | 0.625 |
| 0.0003 | 0.9347 | +0.0029 | 5 | 1 | 0.219 |
| 0.0004 | 0.9358 | +0.0039 | 6 | 1 | 0.125 |
| **0.0005** | **0.9390** | **+0.0072** | **10** | **1** | **0.0117** |
| 0.0006 | 0.9383 | +0.0064 | 10 | 2 | 0.039 |
| 0.0007 | 0.9369 | +0.0050 | 10 | 4 | 0.180 |
| 0.0030 | 0.9149 | −0.0170 | 14 | 40 | 0.0005 |

**Vì sao tin được dù Δ chỉ +0.0072 (dưới ngưỡng 0.01 cũ):** (i) đường cong TRƠN, MỘT ĐỈNH, tăng
đơn điệu 0→0.0005 rồi giảm — không phải gai nhiễu ở biên lưới; (ii) **c đứng yên ở đúng 1 suốt
toàn bộ vùng tăng** trong khi b tăng đều 0→1→3→5→6→10 — prior chỉ CỨU chứ gần như không PHÁ,
đúng chữ ký tín hiệu thật; (iii) p=0.0117 theo kiểm định GHÉP CẶP, là công cụ đúng cho Δ nhỏ
(ngưỡng 0.01 cũ được suy ra từ so sánh KHÔNG ghép cặp, xem mục 2); (iv) đây là prior TOÀN CỤC
ảnh hưởng ĐỀU mọi câu hỏi — đúng loại ngoại lệ mà quy tắc thẩm định đã chốt cho phép tin ở
Δ nhỏ, khác hẳn entity-boost (chỉ chạm 2/1398 câu) đã đổi dấu khi lên public.

**❌ ĐÃ NỘP PUBLIC (28-08) — THẤT BẠI, ĐỔI DẤU. ĐÓNG HƯỚNG.**
Nộp 2 bản sinh cùng lúc từ cùng trạng thái để có Δ sạch:

| bài nộp | PUBLIC |
|---|---:|
| v8 baseline (β=0, sinh mới) | **0.9208** (trùng khít mốc cũ) |
| v9 prior doc_type (β=0.0005) | **0.9193** |

**Δ public = −0.0015**, trong khi Δ local = **+0.0072** (b=10/c=1, McNemar p=0.0117). ĐỔI DẤU.

⚠️ **ĐÂY LÀ BÀI HỌC PHƯƠNG PHÁP LUẬN QUAN TRỌNG NHẤT CỦA DỰ ÁN** — quan trọng hơn bản thân
hướng bị đóng. Prior này có MỌI dấu hiệu đáng tin mà dự án từng đặt ra, và vẫn thất bại:
đường cong trơn một đỉnh · `c` đứng yên ở 1 suốt vùng tăng · p=0.0117 theo kiểm định ghép cặp ·
prior khớp giữa train (26.10%) và val (27.48%) · ước lượng trên train đo trên val, không rò rỉ ·
tỉ lệ câu bị đổi trên val (15.8%) và public (18.2%) gần trùng nhau · là prior TOÀN CỤC ảnh hưởng
đều mọi câu hỏi (đúng ngoại lệ mà quy tắc thẩm định cho phép tin ở Δ nhỏ).

**Nguyên nhân gần như chắc chắn: `train_split_80` và `val_split_20` là HAI NỬA CỦA CÙNG MỘT LÔ
GÁN NHÃN** (cùng cắt ra từ `train.json`). Chúng chia sẻ mọi thói quen/thiên kiến của người gán
nhãn lô đó — trong đó có xu hướng chọn `Luật` gấp 4 lần tỉ lệ tự nhiên. Public test được gán nhãn
ở lô khác, không nhất thiết có cùng thiên kiến. **Vì vậy "prior khớp giữa train và val" KHÔNG PHẢI
bằng chứng prior sẽ chuyển sang test** — nó chỉ chứng minh hai nửa cùng một lô thì giống nhau,
điều hiển nhiên. Mọi đặc trưng học từ THÓI QUEN GÁN NHÃN (chứ không từ nội dung văn bản) đều sẽ
dính đúng cái bẫy này, và cross-validation trên train/val KHÔNG phát hiện được.

**Hệ quả trực tiếp: KHÔNG làm bước learning-to-rank metadata đã dự định.** LTR trên năm ban hành,
cờ sửa đổi, cơ quan ban hành... là CÙNG PHẠM TRÙ (đặc trưng cấp document học từ nhãn train), chỉ
nhiều tham số hơn — tức nhiều chỗ để khớp vào thiên kiến gán nhãn hơn, khả năng thất bại CAO HƠN
chứ không thấp hơn. Lượt nộp thăm dò này đã tiết kiệm cho team mấy ngày công, đúng mục đích đặt ra.

**Thống kê tích luỹ toàn dự án: 3/3 lần Δ local dương nhưng < 0.01 đều ĐỔI DẤU trên public**
(entity-boost +0.0007 → −0.0010 · dòng #20 +0.0065 → thua · prior doc_type +0.0072 → −0.0015).
Không còn là "chưa đủ bằng chứng" nữa mà là một quy luật đã lặp 3/3: **ở dự án này, Δ local dưới
~0.01 KHÔNG mang thông tin về dấu của Δ public, kể cả khi có ý nghĩa thống kê ghép cặp.**

**6. Tính tái lập: file dự đoán KHÁC nhau nhưng ĐIỂM GIỐNG HỆT — cảnh báo ban đầu là THỔI PHỒNG.**
Khi sinh bài nộp cho prior, chốt chặn "beta=0 phải tái lập đúng bản nộp cũ" báo lỗi:
`submission_ensemble.json` (artifact 21-08) khác kết quả sinh hôm nay 822/1000 câu về thứ tự và
550/1000 câu về CẢ TẬP tài liệu. Dò lưới (`top-n ∈ {50,100,150,200}` × `rrf_k=60` × 3 bộ trọng số)
không tổ hợp nào tái lập được (khớp cao nhất 271/1000). Lúc đó tôi kết luận "pipeline đã trôi,
KHÔNG được dùng 0.9208 làm mốc".

**KẾT LUẬN ĐÓ ĐÃ SAI, và bài nộp v8 đã chứng minh:** bản baseline sinh MỚI (beta=0) chấm được
**PUBLIC 0.9208 — TRÙNG KHÍT mốc cũ.** Tức là hai file dự đoán khác nhau ở 550/1000 câu nhưng
**tương đương về điểm**: khác biệt nằm ở các slot KHÔNG chứa tài liệu đúng (đổi một tài liệu sai
lấy một tài liệu sai khác), không chạm vào Recall. Pipeline KHÔNG trôi; **mốc 0.9208 vẫn dùng được
bình thường** và toàn bộ lịch sử so sánh trước đây KHÔNG bị vô hiệu.

Bài học đúng (nhỏ hơn nhiều so với cảnh báo ban đầu): *so khớp file dự đoán* là chốt chặn QUÁ NHẠY
cho pipeline này — nhiều tài liệu đồng điểm/gần điểm nên thứ tự và các slot đuôi dao động vô hại.
Muốn kiểm tra hồi quy thì so **điểm** hoặc so **tập tài liệu đúng được tìm thấy**, đừng so nguyên
danh sách 5 doc.

## Các phát hiện phương pháp luận quan trọng

**Ngưỡng tin cậy local→public:** chênh lệch LOCAL dưới ~0.01 không mang thông tin đáng tin cậy về
PUBLIC — đã kiểm chứng trực tiếp bằng dòng #20 (local +0.0065 có ý nghĩa thống kê, tái lập trên holdout
5602 câu độc lập, vẫn thua public). Từ dòng #20, quy tắc áp dụng: chỉ tin tưởng thay đổi local khi độ
lớn ≥0.02 (mức mà 3 vị trí biên của bảng so sánh local/public khớp thứ hạng); với thay đổi nhỏ hơn,
quyết định phải dựa trên khoảng cách local→public riêng của TỪNG thành phần (model có gap lớn hơn thì
chỉ nên dùng trọng số nhỏ) — đây là cách #21 chọn đúng trọng số harrier=0.2 sau khi #20 (harrier=0.8)
thất bại.

**⚠️ NGUYÊN TẮC BỔ SUNG, đúc kết 28-08 sau khi nộp 6 bài và SOÁT LẠI kỹ từng file đã nộp (đối
chiếu SHA-256 + tái tạo độc lập — xem mục "Đã KIỂM TRA LẠI output" bên dưới, phần chẩn đoán pipeline
— phát hiện 2 chỗ tôi mô tả sai cấu hình, đã sửa):
KHÔNG BAO GIỜ tin một con số local mà chưa nộp public, BẤT KỂ cơ chế chọn tham số là gì.**
4 cấu hình được kiểm chứng ĐỘC LẬP và ĐÚNG như mô tả (prior doc_type argmax, CombSUM minmax argmax,
CombSUM zscore tại cùng điểm trọng số minmax, RRF+m3sparse argmax) đều thua baseline trên public —
**4/4**, dù cách chọn tham số khác nhau (có cái là argmax thật, có cái không). Bằng chứng KHÔNG đủ
mạnh để khẳng định cơ chế cụ thể là "argmax gây overfit" (giả thuyết ban đầu dựa một phần trên #3,
hoá ra không phải argmax thật — đã đính chính). Điều chắc chắn hơn: **cả 2 cơ chế fusion mới thử
(score-based CombSUM dưới cả 2 kiểu chuẩn hoá, và thêm bge-m3 sparse) đều không thắng RRF trên
public**, bất kể cách chọn trọng số. Từ nay: KHÔNG dùng bất kỳ con số local nào (dù chọn thận trọng
hay chọn argmax) để kết luận "cải thiện" — coi mọi cấu hình local-tốt chỉ là ỨNG VIÊN cần public
xác nhận, và SAU KHI nộp, đối chiếu lại đúng tham số/hash đã nộp trước khi rút kết luận từ điểm số
(bài học từ chính 2 sai lệch vừa sửa — đọc điểm mà không xác minh lại tham số dễ gán sai nguyên nhân).

**Cơ chế RRF multi-branch ensemble:** `score(d) = Σ w_i / (rrf_k + rank_i(d))`. Cửa sổ trọng số hữu
ích RẤT HẸP và phụ thuộc `rrf_k` — với `rrf_k=60`, trọng số dưới ~0.8 gần như vô tác dụng, ~1.0 mới đủ
để cứu tài liệu bị model chính bỏ sót, ~1.5 gây "thảm hoạ" (nguồn phụ chiếm toàn bộ top-5). Hạ `rrf_k`
làm đường cong dốc hơn, giảm trọng số cần thiết. **Vị trí cửa sổ này đổi hoàn toàn theo BẢN CHẤT nguồn
phụ** — không có công thức chung để đoán trước, phải sweep: harrier (dense, khác kiến trúc) cần
0,6-1,2; bge-m3 sparse (lexical, có phần trùng lặp với BM25 đã có sẵn) cửa sổ chỉ 0,02-0,06, cao hơn
là lấn át tín hiệu chính ngay; gte-Qwen2 (dense, cùng loại v1) không có cửa sổ nào có ích.

**Model "domain-specific" không tự động thắng model tổng quát:** lặp lại 2 lần trong dự án — reranker
`AITeamVN/Vietnamese_Reranker` (đã tune cho pháp lý) thua `BAAI/bge-reranker-v2-m3` vanilla; embedding
`vietlegal-harrier` (chuyên luật) thua `AITeamVN/Vietnamese_Embedding` (tổng quát tiếng Việt) khi đứng
riêng. Giả thuyết: quy mô + chất lượng pretrain gốc quan trọng hơn việc fine-tune hẹp trên 1 domain cụ
thể. Tuy nhiên model thua khi đứng riêng vẫn có thể là nguồn bổ trợ tốt cho ensemble nếu đủ khác biệt
kiến trúc (harrier: Qwen decoder/last-token pooling, khác hẳn v1: XLM-R encoder) — đo bằng
"complementarity" (% tài liệu model chính bỏ sót mà model phụ cứu được), ngưỡng >30-40% coi là đáng thử.

**Fine-tune thêm AITeamVN v1 nhất quán phản tác dụng** — 4 lần thử, 4 giả thuyết nguyên nhân khác nhau,
đều thất bại:
1. Full `train.json`, LR mặc định 2e-5, 3 epoch → catastrophic forgetting (PUBLIC 0.8922 < 0.9058).
2. TripletLoss margin=5 trên embedding đã chuẩn hoá L2 (khoảng cách Euclid tối đa chỉ 2) → gradient
   không bao giờ tắt → sụp đổ biểu diễn hoàn toàn (Recall 0.92→0.0046, 9 giờ GPU mất trắng).
3. Đổi sang MultipleNegativesRankingLoss (không margin, không sụp đổ) nhưng vẫn Recall -0.1080 — nghi
   ngờ do hard negative đào từ retriever mạnh (top-ranked của v1) dính ~12% false negative (tài liệu
   gần trùng nội dung với đáp án đúng, do 27 nhóm tài liệu trùng lặp + GT thiếu sót trong data gốc).
4. (08-27) Kiểm chứng trực tiếp giả thuyết (3) bằng cách lọc false-negative theo ngưỡng cosine 0,85
   trước khi mining (`--hard-negative-cosine-max`) — filter chạy đúng (412/6519 candidate bị loại,
   đúng bậc độ lớn với ước tính), nhưng Δ vẫn -0.1189, KHÔNG cải thiện so với Lần 3. **Giả thuyết false-
   negative-contamination bị bác bỏ** như nguyên nhân chính — vấn đề nằm ở chỗ khác chưa xác định được
   (ứng viên: batch_size=8 quá nhỏ khiến MNRL thiếu in-batch negative, hoặc bản thân việc fine-tune tiếp
   một model đã pretrain mạnh trên tập train hẹp gây catastrophic forgetting kiểu khác cơ chế (1)).

Kết luận: model AITeamVN gốc (1,1 triệu triplet pretrain) đã ở gần điểm tối ưu cục bộ tốt cho domain
này — cả fine-tune thêm (4/4 lần, 4 cơ chế khác nhau đã loại trừ) lẫn đổi sang model tổng quát mạnh hơn
(`gte-Qwen2-1.5B-instruct`) đều không vượt qua được nó. **Hướng fine-tune ĐÓNG HẲN, không thử lại nữa**
trừ khi có ý tưởng thực sự mới (không phải biến thể của 4 giả thuyết đã loại trừ).

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

Fine-tune embedding (4 lần, 4 cơ chế thất bại khác nhau, xem mục "Fine-tune" ở trên) · reranker mọi hình thức (4 phép thử độc lập)
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

`itdainb/PhoRanker` hoặc `Qwen/Qwen3-Reranker-0.6B`
(ưu tiên thấp, pattern reranker domain-specific đã thua 2 lần) · `CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3`
(gated, cần Hoàng tự xác minh prefix — không to hơn v1 nhưng là kiến trúc Qwen3 + domain luật, chưa
thử) · query expansion (PRF/RM3) cho nhánh BM25 — cổ điển, chưa thử, không cần GPU, chỉ dùng corpus
chính thức nên hợp lệ · thử nghiệm chunking khác (kích thước/overlap) · cập nhật
`scripts/allowed-models.md` cho harrier trong ensemble chính thức · chờ Kiên xem lại GT cho 13 câu hỏi
lương (re-crawl 20 tài liệu rỗng KHÔNG còn là việc cần làm — xem cảnh báo tuân thủ ở mục data quality).

**3 hướng đã chốt thứ tự với Hoàng (08-28): fusion theo score → DEk21 → sweep BM25 k1/b.**
**CẢ 3 HƯỚNG ĐÃ CHẠY XONG (08-28) — xem dòng log tương ứng ở trên:**

1. **Fusion theo SCORE (CombSUM): ✅ ĐÃ CHẠY (08-28), xem dòng log ở trên.** Δ+0.0049 so baseline
   RRF — dưới ngưỡng nộp nhưng cao-nguyên rộng xác nhận tín hiệu thật, giữ làm ứng viên kết hợp cùng
   bge-m3 sparse. Còn việc mở (ưu tiên thấp, không nằm trong 3 hướng đã chốt): thêm scores vào
   `bgem3_sparse.py` để combine m3sparse vào `--mode score` cùng lúc, thử `--score-norm zscore`.
2. **`huyydangg/DEk21_hcmute_embedding`: ✅ ĐÃ CHẠY, ĐÓNG HẲN (08-28), xem dòng log ở trên.** Đứng
   riêng 0.8410 (dưới cả gte-Qwen2 đã đóng), fuse giảm đơn điệu ngay từ đầu — đóng CUỐI CÙNG (không
   có đường lùi vì `max_seq_length=256` là trần kiến trúc).
3. **Sweep `k1`/`b` của BM25Okapi: ✅ ĐÃ CHẠY, ĐÓNG (08-28), xem dòng log ở trên.** BM25 đứng riêng
   tăng THẬT (+0.0129, `k1=1.5/b=0.95`) nhưng pha loãng HOÀN TOÀN ở ensemble (Δ=0.0000 tuyệt đối) vì
   chỉ đóng góp qua `bm25-weight=0.1` cố định — đúng như dự đoán trước khi chạy, hướng rẻ nhất nhưng
   trần cải thiện thấp nhất trong 3 hướng.

**Tổng kết 3 hướng đã chốt (08-28):** không hướng nào đủ điều kiện nộp public riêng lẻ (2/3 dưới
ngưỡng +0.01, 1/3 đóng hẳn vì kiến trúc).

## 🎯 CHIẾN LƯỢC MỚI SAU CHẨN ĐOÁN (08-28) — đọc mục "CHẨN ĐOÁN LẠI TOÀN PIPELINE" ở trên trước

**Bài học chung nhất:** ~15 hướng đã thử đều thuộc **CÙNG MỘT PHẠM TRÙ** — "thêm/đổi/đánh lại
trọng số một tín hiệu NGỮ NGHĨA query↔passage". Chẩn đoán cho thấy phạm trù đó đã bão hoà: tài
liệu đúng đã nằm trong pool ở 99% số câu, 85.7% lỗi còn lại là lỗi XẾP HẠNG trong top-10 giữa
các văn bản ĐỀU đúng chủ đề. Thêm một model ngữ nghĩa nữa **về mặt cấu trúc không thể** phân biệt
"Luật Thanh tra 2010 hay 2022", "Luật hay Công văn", "văn bản khung hay chuyên biệt". Đó là lý do
mọi thử nghiệm hội tụ về ~0.93 bất kể làm gì.

**Phạm trù MỚI cần khai thác: đặc trưng CẤP DOCUMENT, không phụ thuộc câu hỏi.** Đã chứng minh
khả thi bằng `doc_type` (mục 5 ở trên: b=10/c=1, p=0.0117). Các bước tiếp theo, ưu tiên giảm dần:

1. ~~Xác nhận public cho prior `doc_type`~~ → **✅ ĐÃ NỘP 28-08, THẤT BẠI (0.9193 vs 0.9208).
   ĐÓNG.** Xem mục 5 ở trên.
2. ~~Mở rộng thành learning-to-rank metadata~~ → **❌ HUỶ, KHÔNG LÀM.** Cùng phạm trù với hướng
   vừa thất bại, chỉ nhiều tham số hơn để khớp vào thiên kiến gán nhãn. Xem phân tích ở mục 5.
3. ~~Cộng dồn 3 tín hiệu nhỏ~~ → **❌ HUỶ.** Cả 3 đều dưới 0.01 local, và thống kê 3/3 cho thấy
   vùng đó không mang thông tin về dấu public. Cộng 3 thứ không mang thông tin không tạo ra
   thông tin; chỉ làm tăng khả năng khớp nhiễu.
4. **[THẤP] 10 câu (A) tài liệu đúng ngoài pool** — chỉ 0.7% tổng, kể cả sửa hết cũng chỉ +0.007,
   và cần mở rộng pool/đổi cách truy hồi (đắt). Không đáng ưu tiên.

**➡️ ĐỔI CHIẾN LƯỢC (chốt 28-08): THÔI ĐO LOCAL ĐỂ QUYẾT ĐỊNH, NỘP THẲNG LÊN PUBLIC.**

Lý do là một lập luận về ngân sách chứ không phải về kỹ thuật. Ba dữ kiện cộng lại:
1. Local **không thể** xếp hạng ứng viên ở vùng Δ<0.01 — sàn nhiễu ±0.0132, và 3/3 lần đổi dấu.
2. Không còn hướng nào có triển vọng đạt Δ local ≥0.02 (trần pool 0.9890, phần lớn lỗi còn lại là
   GT chọn tuỳ ý giữa các văn bản đều hợp lý).
3. Quy định mục 8: **"kết quả cuối lấy từ phương pháp cho điểm CAO NHẤT trên private test"** — tức
   là *giữ max*, KHÔNG có tập held-out nào phạt việc thử nhiều. Cộng với ~3 tuần public còn lại và
   3 lượt/ngày ở private, ngân sách nộp là dư dả.

=> Thử thêm một cấu hình **khác biệt về CƠ CHẾ** là việc **không có mặt trái**: kỳ vọng điểm cuối
chỉ có tăng. Việc dùng local để LOẠI ứng viên trước khi nộp (điều đã làm suốt dự án) hoá ra là tự
buộc tay mình bằng một thước đo đã chứng minh là mù. Local từ nay CHỈ dùng để **chọn trọng số bên
trong một cơ chế**, KHÔNG dùng để quyết định có nộp hay không.

**Danh mục đã nộp** (`scripts/run_portfolio.ps1`) — **KẾT QUẢ ĐẦY ĐỦ (28-08):**

⚠️ **Đã KIỂM TRA LẠI output từng bài nộp sau khi có điểm** (user yêu cầu, 28-08 tối) — đối chiếu
từng file `.json` đã nộp với việc TÁI TẠO ĐỘC LẬP từ cache (fuse lại bằng đúng tham số đã khai,
đối chiếu từng câu, cả bằng gọi hàm Python lẫn chạy lại đúng CLI, cộng SHA-256 để chắc chắn tuyệt
đối). Phát hiện **2 sai lệch giữa tài liệu tôi ghi và tham số THỰC SỰ đã nộp** — không phải bug
tính điểm hay bug code, mà là tôi mô tả sai cấu hình nào ứng với bài nộp nào. Sửa lại cho đúng:

| # | cấu hình | tham số ĐÃ XÁC MINH THỰC NỘP | local | PUBLIC | Δ vs 0.9208 |
|---|---|---|---:|---:|---:|
| 1 | RRF baseline (đang dùng) | harrier=0.2, bm25=0.1, rrf_k=60 | 0.9319 | **0.9208** | — (mốc) |
| — | prior doc_type | β=0.0005 (đúng như khai) | 0.9390 | 0.9193 | −0.0015 |
| 2 | CombSUM minmax | harrier=0.75, bm25=0.25 (đúng như khai) | 0.9368 | 0.9198 | −0.0010 |
| 3 | CombSUM zscore | **harrier=0.75, bm25=0.25** ⚠️ KHÔNG PHẢI 0.75/0.5 như tôi ghi trước | **0.9382** ⚠️ không phải 0.9400 | **0.9128** | −0.0080 |
| 4 | RRF + bge-m3 sparse | m3sparse=0.04 (đúng như khai) | 0.9346 | 0.9183 | −0.0025 |
| 5 | CombSUM + m3sparse | **m3sparse=0** ⚠️ SHA-256 trùng 100% với #2 — KHÔNG PHẢI phép thử m3sparse>0 | 0.9368 | 0.9198 (= #2, cùng file) | −0.0010 |

**Sai lệch #1 (cấu hình #3):** bài đã nộp dùng ĐÚNG trọng số của #2 (harrier=0.75/bm25=0.25, do
không truyền `-ZHarrier -ZBm25` nên script dùng mặc định), KHÔNG PHẢI đỉnh thật của sweep zscore
(harrier=0.75/bm25=**0.5**, local 0.9400) như tôi mô tả trong 2 lượt trả lời trước. Đỉnh thật của
zscore **CHƯA TỪNG được nộp public** — vẫn là ẩn số. Kết luận "cấu hình được tối ưu MẠNH NHẤT lại
cho public TỆ NHẤT" mà tôi đưa ra trước đó **không có cơ sở** vì #3 không phải cấu hình tối ưu nhất;
điều CÓ cơ sở là: cùng 1 điểm trọng số, đổi minmax→zscore làm public tệ hơn hẳn (0.9198→0.9128) dù
local nhỉnh hơn (0.9368→0.9382) — tức bản thân chuẩn hoá zscore nhạy với khác biệt phân phối điểm
giữa val/public hơn minmax, một cơ chế thất bại RIÊNG, không phải bằng chứng cho giả thuyết argmax.

**Sai lệch #2 (cấu hình #5), nghiêm trọng hơn:** đối chiếu SHA-256 cho thấy
`submission_combsum_m3sparse.json` **giống hệt byte-for-byte** `submission_combsum.json` (#2).
Truy nguyên: KHÔNG phải bug — sweep local cho #5 (bảng ở trên, mục 5) cho thấy **m3sparse=0 chính
là đỉnh thật của sweep đó** (0.9368, mọi m3sparse>0 đều THẤP HƠN) — nghĩa là gộp m3sparse vào
CombSUM đã tối ưu không mang lại gì kể cả trên local. Làm đúng theo hướng dẫn "chọn số tốt nhất từ
sweep" thì con số tốt nhất chính là 0, nên bài nộp #5 **vô tình == bài nộp #2**. Đây KHÔNG phải bằng
chứng public cho "CombSUM+m3sparse thất bại" — là bằng chứng LOCAL (đã có từ trước) rằng kết hợp đó
không có gì để thử; hai điểm public trùng khít (0.9198 = 0.9198) chỉ xác nhận hệ chấm điểm ổn định
(cùng input → cùng điểm), không phải dữ liệu mới.

**Tóm lại sau khi soát lại:** có **4 cấu hình được kiểm chứng độc lập, khác biệt thật** (prior
doc_type, CombSUM minmax, CombSUM zscore tại 0.75/0.25, RRF+m3sparse) — **cả 4/4 đều thua baseline
trên public**, kết luận cốt lõi (không có cấu hình reweighting nào thắng baseline) KHÔNG đổi. Nhưng
2 diễn giải phụ tôi đưa ra trước — "đỉnh sweep cho kết quả tệ nhất" và "đã thử CombSUM+m3sparse
thất bại" — bị RÚT LẠI vì dựa trên mô tả sai. Bài học rút ra đã sửa lại ở đoạn dưới cho khớp bằng
chứng thật (khác biệt-chuẩn-hoá, không phải mức-độ-tối-ưu-hoá).

**❌ ĐÓNG DỨT ĐIỂM cả 2 hướng "ứng viên kết hợp" còn treo, bằng bằng chứng PUBLIC trực tiếp** (không
còn chỉ là "dưới ngưỡng, chưa đủ bằng chứng" — đã có Δ public âm rõ ràng):
- **Score-based fusion (CombSUM, cả minmax lẫn zscore):** đóng hẳn. #2, #3, #5 đều âm public.
- **bge-m3 sparse làm nhánh RRF phụ:** đóng hẳn. #4 âm public (−0.0025), dù local dương +0.0027.

Cấu hình #4/#5 cần scores cho nhánh sparse — đã sửa `bgem3_sparse.py` (28-08) để `rank` lưu thêm
key `scores` (chỉ THÊM key, các script RRF cũ đọc file này vẫn chạy y hệt) — việc mở này đã xong,
kết quả cho thấy hướng bản thân nó không dùng được, không phải do thiếu code.

**➡️ KẾT LUẬN SAU DANH MỤC: KHÔNG còn cấu hình reweighting/fusion nào đáng thử tiếp trên trục này.**
Cấu hình tốt nhất đã biết vẫn là **RRF baseline gốc, PUBLIC 0.9208, không đổi**.

## 📋 KIỂM KÊ HƯỚNG CÒN LẠI (28-08) — đã đo trần cho từng cái, không còn suy đoán

Sau khi đóng cả 2 phạm trù lớn, kiểm kê TOÀN BỘ những gì chưa thử, kèm **trần đã đo** chứ không phải
cảm tính. Sàn nhiễu local là ±0.0132; ngưỡng đáng nộp là ≥0.02:

| hướng chưa thử | trần / bằng chứng | kết luận |
|---|---|---|
| Cơ chế "họ văn bản" | **+0.0057 nếu sửa HOÀN HẢO** (đo, mục 4b) | chết — dưới sàn nhiễu |
| Đa dạng hoá slot (MMR) | lãng phí slot ở 10.0% câu sai vs 10.5% toàn bộ → **không liên quan** (đo, mục 4b) | chết |
| 10 câu đáp án ngoài pool | **+0.007 nếu sửa HOÀN HẢO**, cần đổi cách truy hồi (đắt) | chết — dưới sàn nhiễu |
| Query expansion PRF/RM3 cho BM25 (dòng B2) | **đã chứng minh vô hiệu**: sweep k1/b cho BM25 đứng riêng +0.0129 → ensemble **0.0000 tuyệt đối** vì `bm25-weight=0.1` pha loãng hết | chết bởi chính phép đo đã có |
| Reranker PhoRanker / Qwen3-Reranker (dòng B1) | phạm trù reranker **4/4 phép thử độc lập đều âm** | chết |
| Đổi chunking, kích thước/overlap (dòng C1) | pool recall đã 99.00% → gần như không còn gì để "kéo về"; tốn 1,5-2h GPU encode lại | không đáng |
| `CATI-AI/Qwen3-Embedding-0.6B-vietnamese-legal-v3` | **loose end thật** — Kiên từng chạy, CHƯA có kết quả ghi lại. Nhưng 5/6 lần đổi model đã thua, và VN-MTEB cho thấy benchmark **nghịch** với task này (AITeamVN xếp CHÓT benchmark retrieval 34.18% nhưng thắng đậm ở đây) | kỳ vọng thấp, nhưng nên hỏi Kiên lấy số |
| CombSUM zscore tại đỉnh THẬT (0.75/**0.5**) | chưa nộp bao giờ (xem đính chính sai lệch #1) | vé số miễn phí, nộp thì nộp |

**Tổng trần lý thuyết của MỌI thứ còn lại, cộng dồn và giả định sửa hoàn hảo: ~+0.01**, phần lớn
chồng lấn nhau và không thể đạt được thực tế. **Không còn hướng nghiên cứu nào có đường tới +0.02.**

**➡️ HƯỚNG DUY NHẤT CÒN GIÁ TRỊ KỲ VỌNG DƯƠNG: khai thác ngân sách nộp ở vòng PRIVATE.**
Quy định mục 8: **3 lượt/ngày** và **kết quả cuối = phương pháp cho điểm CAO NHẤT trên private**.
Private 19-23/09 → khoảng **15 lượt**, và ta GIỮ MAX. Với các cấu hình chênh nhau trong khoảng nhiễu
(±0.008), lấy max của ~12-15 lượt thu được phần đuôi trên của nhiễu — lợi ích cỡ +0.005-0.01, **tương
đương mọi thứ đã đuổi theo cả tuần, nhưng được BẢO ĐẢM bởi luật chứ không phải suy đoán**, và tốn 0
giờ nghiên cứu. Việc cần làm: chuẩn bị sẵn ~12 biến thể hợp lệ (đã có sẵn code cho tất cả: baseline,
CombSUM minmax/zscore ở vài điểm trọng số, RRF+m3sparse, prior doc_type ở vài β, các rrf_k khác) để
chạy ngay khi private mở, KHÔNG nghiên cứu thêm.

Nguyên tắc kèm theo: **nộp baseline TRƯỚC TIÊN** ở private (chốt sàn ~0.92 đã biết), rồi mới tiêu
các lượt còn lại cho biến thể.

## 📚 ĐỐI CHIẾU VỚI SOTA HỌC THUẬT (tra cứu 28-08) — phát hiện 1 LỖ HỔNG KIẾN TRÚC

Tra cứu các công trình về truy hồi văn bản pháp luật tiếng Việt để xem cách tiếp cận của team đứng
ở đâu:

| công trình | kiến trúc | kết quả |
|---|---|---|
| Multi-stage IR for VN Legal Texts (arXiv 2209.14494) | BM25+ → PhoBERT sentence-transformer fine-tune 3 vòng hard-negative, gộp NHÂN | Recall@20 0.970 · F2 0.741 |
| Semi-Hard Negative Mining (arXiv 2507.14619) | Bi-encoder → **PhoRanker cross-encoder rerank**, fine-tune semi-hard negative | MRR@10 79.11 · **top-3 SoICT Hackathon 2024** |
| Synthetic Data (arXiv 2412.00657) | Llama-3-70B sinh 507K query giả → CoT-MAE pretrain → **ColBERT** | MRR@10 74.61 · Recall@10 89.29 |

**⚠️ KHÔNG so sánh được ĐIỂM SỐ trực tiếp** — khác corpus (8.532 văn bản của ta vs 114K-143K
passage), khác chỉ số (Recall@5 vs MRR@10/F2), khác độ khó. Chỉ so sánh được KIẾN TRÚC.

**Kiến trúc của team = "hybrid dense+sparse ensemble", ĐÚNG chuẩn mạnh của lĩnh vực** — bài
2507.14619 còn ghi rõ giải pháp của họ là "lightweight alternative to **ensemble-based competitors**",
tức các đội THẮNG họ đều dùng ensemble như ta.

**NHƯNG có 1 lỗ hổng: giai đoạn CROSS-ENCODER RERANK.** Gần như mọi pipeline SOTA đều có tầng này;
ta thì KHÔNG (đã thử 4 lần, 4/4 âm). ⚠️ Đối chiếu kỹ lại 4 phép thử đó với công thức SOTA:

| yếu tố | SOTA (2507.14619) | 4 phép thử của team |
|---|---|---|
| model | **PhoRanker** (chuyên tiếng Việt) | `bge-reranker-v2-m3` (đa ngữ), `AITeamVN/Vietnamese_Reranker` |
| huấn luyện | **fine-tune với semi-hard negative** (n=5) | **dùng THẲNG bản pretrained, KHÔNG fine-tune** |
| ghép vào pipeline | rerank top-90 của bi-encoder | thay thứ hạng RRF, hoặc làm nguồn RRF phụ |

=> **Ta CHƯA từng thử đúng công thức SOTA: PhoRanker + fine-tune semi-hard negative.** PhoRanker nằm
ở dòng B1 bảng phân công, vẫn "chưa chạy". Bài 2507.14619 còn chỉ rõ vì sao hard-negative gây bất ổn
(47.96% MRR) còn semi-hard tốt hơn hẳn (78.21%) — **đúng cơ chế thất bại mà team đã đâm vào 4 lần khi
fine-tune embedding bằng hard negative** (xem mục "Fine-tune", Lần 3/4). Tức thất bại fine-tune của
team có thể do CHỌN SAI LOẠI NEGATIVE, và tài liệu này đưa ra lời giải cụ thể chưa được thử.

**Đây là ngoại lệ DUY NHẤT cho kết luận "không còn hướng nào":** chẩn đoán cho thấy 85.7% lỗi còn
lại là lỗi XẾP HẠNG với đáp án ĐÃ nằm trong pool (Recall@100 = 0.9890 vs Recall@5 = 0.9319) — tức
**+0.057 dư địa mà CHỈ cơ chế rerank chạm tới được**. Chỉ cần khai thác 1/3 số đó đã là +0.019, vượt
ngưỡng 0.02. Đây là thứ DUY NHẤT còn lại có đúng bậc độ lớn cần thiết.

**✅ ĐÃ DỰNG XONG (28-08), CHỜ CHẠY GPU.** 3 file mới, test kỹ trước khi giao:
- `src/evaluation/rerank_blend.py` — `score` (chấm cross-encoder) / `blend` (sweep alpha + McNemar) /
  `submit` / `selftest`. **Trộn điểm ở cấp DOCUMENT**: `score = norm(ensemble) + alpha·norm(reranker)`,
  chấm lại đúng top-N document của ensemble, mỗi document lấy chunk đã THẮNG phép max lúc gộp.
  Vì min-max là phép đơn điệu nên **alpha=0 tái lập baseline CHÍNH XÁC** — đã kiểm chứng bằng test
  tích hợp qua CLI thật: alpha=0 ra 0.6000 khớp tuyệt đối với Recall baseline tính độc lập, alpha>0
  cứu đúng 2 câu có đáp án ở hạng 6-7 (b=2, c=0). Selftest 12/12.
- `src/training/finetune_phoranker.py` — `preview` (xem cặp train bằng mắt, không GPU) / `train` /
  `selftest`. Đào **semi-hard**: lấy top-K ứng viên → loại đáp án đúng → **lấy NGẪU NHIÊN n=5**
  (KHÔNG phải n cái đầu bảng = hard negative). Selftest 10/10 gồm cả kiểm tra semi-hard THỰC SỰ
  phân tán (30 tập khác nhau/30 seed) và có lấy cả ứng viên cuối bảng. **Chốt chặn rò rỉ CỨNG**:
  từ chối chạy nếu bất kỳ câu train nào nằm trong val_split_20 (đã test cả 2 chiều).
- `scripts/run_phoranker.ps1` — 7 giai đoạn.

**Dữ liệu đã xác minh sẵn sàng:** `ranks_hold_*.npz` chính là 5602 câu train_split_80, **giao với
val = 0 câu**; `passage_segmented`/`question_segmented` có đủ ở cả 3 tập nên **KHÔNG cần cài pyvi**
(PhoRanker bắt buộc input đã tách từ — xác minh qua model card).

**Đã kiểm tra preview cặp huấn luyện (29-08):** 5.562/5.602 câu dùng được (40 câu có đáp án ngoài
pool — khớp đúng tỉ lệ 0,7% đã đo ở mục 1), 33.810 cặp = 6.000 positive / 27.810 negative (đúng
5.562×5). Đọc tay: positive là *"Điều 4. Thời hạn cấp đăng ký, biển số xe"* trả lời trực tiếp câu
hỏi về thời hạn cấp đăng ký xe; negative đều **cùng chủ đề "thời hạn cấp" nhưng khác đối tượng**
(căn cước công dân, đăng ký doanh nghiệp, khiếu nại) — đúng loại negative cần, không phải rác ngẫu
nhiên. Chất lượng đào negative ĐẠT.

**Đã lượng hoá rủi ro truncation (29-08), trước khi tốn GPU:** chunk có trung vị 182 từ đã tách,
phân vị 75% là 291 từ, trong khi ngân sách passage trong 256 token chỉ ~148-171 từ (PhoBERT BPE
~1,3-1,5 token/từ) → **~52-56% chunk bị cắt**. NHƯNG yếu tố giảm nhẹ đo được: **55% chunk BẮT ĐẦU
bằng tiêu đề điều khoản** (`Điều/Chương/Mục/...`) nhờ chunker structure-aware, tức phần mang thông
tin phân biệt nằm ở ĐẦU chunk còn truncation thì cắt ĐUÔI. Khác hẳn tình huống DEk21: bi-encoder
mã hoá cả văn bản nên mất đuôi là mất biểu diễn, còn cross-encoder chỉ cần đủ để phán "đoạn này có
đúng chủ đề câu hỏi không" — và tiêu đề điều khoản nói đúng điều đó. **Kết luận: rủi ro THẬT nhưng
KHÔNG hiển nhiên chí tử → vẫn phải chạy cửa chặn để biết, không suy đoán tiếp.**

**CỬA CHẶN bắt buộc trước khi tốn GPU fine-tune:** chạy `-Phase score-pretrained` + `blend-pretrained`
với PhoRanker GỐC trước. `max_length=256` là trần kiến trúc cứng của PhoBERT/RoBERTa — **đúng lý do
đã đóng hẳn DEk21** — mà chunk của ta tới 2000 ký tự nên bị cắt nhiều. Nếu bản pretrained giảm ĐƠN
ĐIỆU ngay từ alpha nhỏ nhất thì truncation đã giết tín hiệu → DỪNG, không fine-tune, đóng hướng.

**Nhưng cược này rủi ro cao, phải nói thẳng:** team đang 0/4 với reranker và 0/4 với fine-tune; kể
cả local có tăng thì 3/3 tiền lệ cho thấy Δ nhỏ không chuyển sang public; và corpus của ta chỉ 8.532
văn bản với Recall@5 đã 0.93 — dư địa cấu trúc NHỎ HƠN NHIỀU so với các benchmark 114K-143K passage
mà các bài báo kia đạt được cải thiện lớn. Ước lượng thành công: thấp, nhưng KHÁC KHÔNG, và là cái
duy nhất còn lại như vậy.

*(Nghĩa vụ khai báo mô hình theo mục 7 — dòng A2, harrier chưa khai báo, hạn ~09-09 nếu tính 10
ngày trước 19-09 — vẫn CHƯA LÀM, user tạm hoãn. Cần quay lại trước đầu tháng 9.)*

**Trần thực tế cần nhớ khi đặt kỳ vọng:** Recall@100 = 0.9890 là trần tuyệt đối của mọi kỹ thuật
xếp hạng lại trên pool hiện tại. Trong 60 câu nhóm (B), một phần đáng kể là **ground truth chọn 1
trong nhiều văn bản đều hợp lý** (ví dụ 5 nội quy tiếp công dân của 5 cơ quan, câu hỏi không hề
nói cơ quan nào) — phần này KHÔNG tối ưu được bằng bất cứ mô hình nào. Nên mục tiêu thực tế của
toàn bộ phạm trù mới này là khoảng **+0.01 đến +0.02**, không phải một cú nhảy như #11.

**`BAAI/bge-m3` sparse (lexical học được) làm nhánh phụ: ❌ ĐÃ NỘP PUBLIC (28-08), ĐỔI DẤU, ĐÓNG
HẲN.** Local đứng riêng dương thật (đường cong trơn một đỉnh tại `m3sparse=0,04-0,05`, Δ+0,0027) —
nhưng nộp public (`m3sparse=0.04`, xem bảng "Danh mục đã nộp" ở mục CHẨN ĐOÁN LẠI TOÀN PIPELINE) ra
**0.9183, Δ−0.0025 so baseline 0.9208**. Không còn "chờ kết hợp cùng cải tiến khác" — bản thân
hướng đã có bằng chứng public âm, cộng với gì cũng không còn ý nghĩa để thử.

## Ghi chú tuân thủ

Tổng tham số ensemble hiện tại ~1,16 tỷ (v1 568M + harrier 596M), dưới hạn mức 4 tỷ (quy định mục 5).
Thêm model mới vào pipeline chính thức cần đề xuất cho BTC trước 10 ngày so với hạn chót private test
(quy định mục 7) — `allowed-models.md` cần cập nhật khi chốt dùng ensemble cho vòng private.
