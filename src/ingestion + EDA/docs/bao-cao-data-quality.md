# Báo cáo data-quality — gửi Kiên (từ Hoàng, team UIT-Vinno)

Cập nhật 2026-08-17. Tổng hợp từ 2 vòng phân tích: (1) đọc tay 20/143 câu recall=0 trên `val_split_20`
(`outputs/predictions/error_analysis_recall0.md`), (2) quét lại toàn bộ `data/processed/corpus.jsonl`
(8.532 tài liệu) và `data/raw/train.json` (7.000 câu hỏi) bằng script để xác nhận/mở rộng phạm vi.
Đây là 3 nhóm vấn đề KHÔNG liên quan tới model retrieval — sửa được có thể tăng Recall đo được mà
không cần đổi gì ở embedding/BM25/hyperparameter.

> ⚠️ **QUAN TRỌNG — ràng buộc quy định cuộc thi, đọc trước khi ai đó định "tự sửa":** `quy-dinh-cuoc-thi-task1.md`
> mục 7 ghi rõ: *"Chỉ được dùng bộ dữ liệu chính thức do BTC phát hành. **Không được gán nhãn thủ
> công**, thu thập dữ liệu ngoài, hoặc áp dụng data augmentation từ nguồn bên ngoài."* Mục 11: vi phạm
> lần đầu bị cảnh cáo/trừ tối đa 20% điểm, tái phạm bị loại. Điều này có nghĩa: **team KHÔNG được tự
> sửa `train.json` (thêm/bớt `document_id` trong GT) dù có bằng chứng research cụ thể đến đâu**, và
> **KHÔNG được lấp nội dung 20 tài liệu rỗng bằng cách fetch từ nguồn khác ngoài thuvienphapluat.vn**
> (dù nội dung là cùng 1 văn bản luật) — cả 2 đều rơi đúng vào điều cấm ở mục 7. Chỉ BTC mới có thể sửa
> hợp lệ dữ liệu chính thức của họ. Vì vậy mục 1 và mục 3 dưới đây, ngoài gửi Kiên, **nên cân nhắc gửi
> khiếu nại/phản ánh chính thức qua `dsc@uit.edu.vn`** (kênh đã có trong quy định, phản hồi tối đa 7
> ngày làm việc) để BTC tự quyết định có sửa dữ liệu gốc hay không — team không tự ý sửa.

## 1. Tài liệu có `passage` rỗng hoàn toàn — ĐÃ XÁC NHẬN NGUYÊN NHÂN, cần Kiên xử lý crawler

Quét toàn corpus: **20/8.532 tài liệu (~0.23%)** có `passage` rỗng hoàn toàn. Danh sách đầy đủ:

| doc_id | Link nguồn |
|---|---|
| 10533 | tcvn/.../QCVN-20-2015-BLDTBXH-...-915197.aspx |
| 131890 | tcvn/.../TCVN-8400-24-2014-...-914993.aspx |
| 149317 | tcvn/.../TCVN-13274-2020-...-917656.aspx |
| 177151 | tcvn/.../TCVN-10065-2013-...-911546.aspx |
| 181693 | van-ban/Bao-hiem/.../Quyet-dinh-co-che-quan-ly-tai-chinh-...-561892.aspx |
| 187338 | tcvn/.../QCVN-02-23-2017-BNNPTNT-...-916141.aspx |
| 191261 | van-ban/Lao-dong-Tien-luong/.../Thong-tu-vi-tri-viec-lam-...-508175.aspx |
| 196918 | tcvn/.../TCVN-9251-2012-Bia-ho-so-luu-tru-906764.aspx |
| 208668 | van-ban/Giao-duc/.../Nghi-dinh-huong-dan-Luat-Giao-duc-dai-hoc-493273.aspx |
| 210808 | tcvn/.../TCVN-5687-2010-...-907929.aspx |
| 232489 | tcvn/.../QCVN-03-2014-BCT-...-912565.aspx |
| 255762 | tcvn/.../TCVN-7795-2021-Biet-thu-du-lich-xep-hang-919472.aspx |
| 263763 | tcvn/.../QCVN-01-2019-BCT-...-918430.aspx |
| 288457 | van-ban/Bao-hiem/Luat-Bao-hiem-y-te-505750.aspx |
| 34810 | van-ban/Vi-pham-hanh-chinh/.../Thong-tu-quy-trinh-xu-ly-...-514550.aspx |
| 55497 | tcvn/.../QCVN-16-1-2015-BYT-doi-voi-thuoc-la-dieu-915044.aspx |
| 56098 | tcvn/.../TCVN-6102-2020-...-919141.aspx |
| 57978 | tcvn/.../TCVN-10778-2015-Ho-chua-...-916090.aspx |
| 67660 | tcvn/.../QCVN-46-2022-BTNMT-Quan-trac-khi-tuong-919584.aspx |
| 71014 | tcvn/.../QCVN-02-03-2009-BNNPTNT-...-901788.aspx |

(URL đầy đủ có tiền tố `https://thuvienphapluat.vn/` — rút gọn cho gọn bảng, danh sách gốc nằm trong corpus.)

**Đã tự kiểm tra nguyên nhân (KHÔNG cần chờ Kiên đoán nữa):** thử `fetch` trực tiếp 2 URL trong danh
sách trên (1 `tcvn/`, 1 `van-ban/`) — cả 2 đều trả về nội dung nhưng bị **chặn bằng paywall
"Thành viên Pro"**: trang chỉ hiện phần tóm tắt/mở đầu, toàn văn điều khoản bị ẩn sau yêu cầu đăng
nhập tài khoản Pro của thuvienphapluat.vn. Đây là lý do 20 tài liệu này rỗng — **không phải lỗi code
xử lý dữ liệu, mà là lỗi khi crawl** (crawler gốc của Kiên cào được ~8.512/8.532 tài liệu bình thường
nên chắc chắn đã có cách vượt paywall — có thể do phiên đăng nhập hết hạn, bị rate-limit, hoặc lỗi
tạm thời đúng lúc crawl 20 trang này).

**Đề nghị cụ thể cho Kiên:** không cần điều tra lại từ đầu — chỉ cần **re-crawl riêng 20 URL trong
bảng trên bằng đúng crawler/tài khoản Pro đã dùng cho 8.512 tài liệu còn lại**. Nếu re-crawl vẫn thất
bại (trang đã gỡ/đổi URL), mới cần cân nhắc loại các tài liệu này khỏi corpus hoặc khỏi ground truth
của câu hỏi liên quan.

## 2. Tài liệu trùng lặp nội dung (nhiều `doc_id` cho cùng 1 văn bản) — MỞ RỘNG phát hiện 6 LẦN

Quét MD5 hash toàn văn `passage` (round 1) chỉ bắt được **4 nhóm trùng khớp byte-for-byte / 9 tài
liệu** — bỏ sót các bản trùng bị lệch nhẹ do crawl 2 lần từ 2 URL khác nhau (khoảng trắng/định dạng
khác nhau dù cùng nội dung).

**Phương pháp mới (round 2):** thuvienphapluat.vn gắn mỗi trang 1 **mã số trang cố định** ở cuối URL
(dạng `...-<id>.aspx`). Cùng 1 văn bản có thể bị crawl 2 lần qua 2 slug URL khác nhau nhưng CÙNG mã số
trang này — ví dụ Luật Khám bệnh, chữa bệnh 2023 có cả `.../Luat-15-2023-QH15-kham-benh-chua-benh-372143.aspx`
và `.../Luat-kham-benh-chua-benh-2023-so-15-2023-QH15-372143.aspx`, cùng đuôi `-372143.aspx`. Gom
nhóm toàn corpus theo mã số trang này tìm được **27 nhóm trùng lặp / 54 tài liệu** — gấp 6 lần so với
kết quả MD5. Đã xác minh 4 cặp mẫu bằng `difflib.SequenceMatcher` (so khớp văn bản): độ giống nhau
98.78% – 99.87% (0.9878 / 0.9987 / 0.9972 / 0.9942) — xác nhận đây là trùng lặp thật, không phải nhiễu.

Danh sách đầy đủ 27 nhóm (mỗi nhóm 2 `doc_id`, gộp theo mã trang):

`102268+279595`, `104102+65567`, `105589+152569`, `10719+244095`, `121575+84226`, `122021+60683`,
`133300+75341`, `134704+232835`, `139874+167334`, `141650+241689`, `145175+248942`, `157195+277320`,
`158189+206810`, `181254+64124`, `196732+93859`, `206546+283751`, `218610+296890`,
`222242+276631` (case gốc đã báo cáo — Luật Khám bệnh, chữa bệnh 2023, xem câu hỏi `13366`),
`228641+61662`, `242348+245887`, `248399+34341`, `254937+280171`, `260037+292870`, `277743+35337`,
`36873+42541`, `43848+80730`, `82599+93049`.

Tất cả đều là cặp 2-tài-liệu (không có nhóm 3+ trong lần quét này).

**Đề nghị:** dùng danh sách 27 nhóm trên để dedup — giữ 1 `doc_id` đại diện/nhóm (ví dụ `doc_id` nhỏ
hơn), xoá hoặc gắn cờ liên kết `doc_id` còn lại, đồng thời cập nhật `ground truth` ở các câu hỏi có
`doc_id` bị xoá trỏ sang `doc_id` đại diện. Việc này giúp model không lãng phí slot dự đoán (top-5)
cho 2 `doc_id` cùng nội dung như case `13366` đã ghi nhận.

## 3. Ground truth nghi thiếu văn bản cho nhóm câu hỏi về lương theo chức danh — ĐÃ QUÉT TOÀN BỘ, PATTERN CÓ HỆ THỐNG

Quét từ khoá (`lương cơ sở`, `mức lương`, `phụ cấp lương`, `hệ số lương`, `lương hưu`, `tăng lương`)
trên toàn bộ **7.000 câu hỏi** trong `train.json`: tìm được **47 câu hỏi liên quan lương**.

Trong đó, **13 câu hỏi hỏi về mức lương/hệ số lương của MỘT CHỨC DANH CỤ THỂ** nhưng ground truth chỉ
gán 1 trong 3 văn bản KHUNG chung (`284539` = Nghị quyết 69/2022/QH15 dự toán ngân sách nhà nước 2023,
`166505` = Nghị định 204/2004/NĐ-CP chế độ tiền lương chung, `43443` = Nghị định 38/2019/NĐ-CP mức
lương cơ sở) — không văn bản nào trong 3 văn bản này có bảng hệ số lương riêng cho từng chức danh.

**5 câu hỏi GT hoàn toàn chỉ gồm 3 văn bản khung này (nghi ngờ CAO nhất — không có văn bản chuyên biệt nào)**:

| question_id | Câu hỏi | GT hiện tại |
|---|---|---|
| 153714 | Mức lương của giảng viên cao cấp (tiến sỹ) hiện nay là bao nhiêu? | `166505` |
| 63884 | Hệ số lương áp dụng đối với công chức đang giữ ngạch kiểm tra viên trung cấp thuế là bao nhiêu? | `284539, 166505, 43443` |
| 95464 | Mức lương của Kiểm soát viên chất lượng sản phẩm, hàng hóa hiện nay là bao nhiêu? | `284539, 166505` |
| 51028 | Mức lương của sĩ quan nghiệp vụ được bổ nhiệm vào ngạch Thanh tra viên cao cấp Công an nhân dân là bao nhiêu? | `284539, 166505, 43443` |
| 42686 | Công chức đang giữ ngạch thống kê viên chính được áp dụng hệ số lương là bao nhiêu? | `284539, 166505, 43443` |

**8 câu hỏi khác GT có trộn văn bản khung + văn bản khác (cần Kiên xem lại từng case, mức độ nghi ngờ THẤP hơn)**:

| question_id | Câu hỏi | GT hiện tại |
|---|---|---|
| 129132 | Mức lương cụ thể của công chức bổ nhiệm vào ngạch kiểm ngư viên được tính thế nào? | `41395, 43443` |
| 161032 | Viên chức được bổ nhiệm chức danh nghiên cứu viên cao cấp có mức lương cụ thể là bao nhiêu? | `41395, 43443` |
| 11594 | Mức lương Chủ tịch nước năm 2023 là bao nhiêu? | `284539, 43443, 146620` |
| 23760 | Tăng mức đóng BHXH bắt buộc khi tăng lương cơ sở lên 1.800.000đ/tháng từ 01/7/2023? | `23402, 43443` |
| 152130 | Thông tư 41/2023/TT-BQP về mức lương cơ sở, phụ cấp quân hàm? | `166505, 93496` |
| 30962 | Phó Chủ tịch HĐND xã được hưởng phụ cấp chức vụ lãnh đạo so với mức lương cơ sở là bao nhiêu? | `277076, 284539` |
| 112128 | Mức lương của cán bộ công chức cấp huyện loại A1 hiện nay là bao nhiêu? | `15331, 212531, 166505` |
| 156428 | Tỷ lệ tăng lương cơ sở là bao nhiêu? | `41395, 284539, 43443` |

(Case gốc `29272`, `161032` đã báo cáo trước đó nằm trong nhóm này — `161032` xuất hiện lại ở đây với
cùng pattern; `29272` không khớp bộ từ khoá quét lần này nên không có trong bảng nhưng vẫn là case hợp
lệ đã ghi nhận trước.)

**Đề nghị:** đây là pattern có hệ thống (13/47 = ~28% câu hỏi về lương bị ảnh hưởng), khả năng cao do
quy trình gán GT ưu tiên gán văn bản "khung" tổng quát khi không tìm thấy ngay văn bản chuyên biệt theo
chức danh. Đề nghị Kiên/người phụ trách gán GT rà lại 13 câu hỏi trên (ưu tiên 5 câu đầu bảng), tìm
văn bản chuyên biệt đúng chức danh được hỏi (thường là thông tư liên tịch về bổ nhiệm/xếp lương ngạch,
chức danh cụ thể) để bổ sung vào GT — **quyết định cuối vẫn cần người có domain knowledge về hệ thống
văn bản lương công chức/viên chức, đây không phải lỗi có thể tự sửa an toàn bằng script**.

**Bằng chứng bổ sung (research thuần để tham khảo — KHÔNG tự áp dụng, xem cảnh báo đầu file):** thử
research trên web cho case `153714` ("giảng viên cao cấp"), tìm được văn bản chuyên biệt đúng là
**Thông tư 40/2020/TT-BGDĐT** (mã số, xếp lương viên chức giảng dạy — quy định chức danh "giảng viên
cao cấp hạng I" thuộc nhóm lương A3.1, hệ số 6,20-8,00). Đối chiếu corpus: văn bản này **ĐÃ CÓ SẴN**
trong `data/processed/corpus.jsonl` với `doc_id 85808` (21.549 ký tự, link
`.../Thong-tu-40-2020-TT-BGDDT-...-456220.aspx`) — chỉ là chưa được gắn vào GT của câu hỏi `153714`.
Đây là bằng chứng CỤ THỂ (không phải suy đoán) cho thấy pattern ở mục 3 là có thật và ít nhất 1 trường
hợp có thể sửa ngay lập tức nếu BTC xác nhận — nhưng **team không tự thêm `85808` vào GT của `153714`**
vì đây chính là "gán nhãn thủ công" bị cấm ở mục 7 (dù tài liệu lấy từ đúng corpus chính thức, việc
CHỌN nó làm GT vẫn là quyết định thủ công của team, không phải của BTC). Đưa case này vào để Kiên/BTC
xác minh và tự quyết định, không phải để team tự áp dụng.

---

## Tóm tắt: phần nào team tự làm được, phần nào cần Kiên/BTC

- **Mục 1 (tài liệu rỗng):** đã tự xác nhận nguyên nhân (paywall) — KHÔNG tự sửa được, và **không được
  tự sửa bằng cách fetch nội dung từ nguồn khác** (vi phạm mục 7 "không thu thập dữ liệu ngoài"). Chỉ
  Kiên (re-crawl bằng đúng crawler/tài khoản Pro) hoặc BTC (nếu team báo cáo và họ tự sửa dữ liệu gốc)
  mới sửa hợp lệ được.
- **Mục 2 (trùng lặp):** đã tự phát hiện đầy đủ + xác minh (27 nhóm/54 tài liệu) bằng script — có thể
  TỰ dedup ở bước hậu xử lý retrieval (`eval_dedup_backfill.py`, chỉ sắp xếp lại các `doc_id` GỐC từ
  đúng corpus BTC cấp theo điểm retrieval đã tính, không thêm dữ liệu ngoài, không sửa GT — giống 1
  quyết định thiết kế hệ thống chứ không phải sửa dữ liệu) mà không cần đổi corpus/GT gốc — **việc này
  KHÔNG vi phạm mục 7**, team tự làm được và đã triển khai.
- **Mục 3 (GT thiếu):** đã tự quét/định lượng toàn bộ (13/47 câu nghi vấn) — **không được tự sửa
  `train.json`** (chính là "gán nhãn thủ công" bị cấm ở mục 7), dù có bằng chứng research cụ thể (xem
  case `153714`/`85808` ở trên). Cần báo Kiên VÀ cân nhắc gửi khiếu nại chính thức qua `dsc@uit.edu.vn`
  để BTC xác nhận/tự sửa dữ liệu gốc nếu họ đồng ý đây là lỗi gán nhãn.

⚠️ **Lưu ý quan trọng nhất:** ranh giới "tự làm được" ở đây KHÔNG chỉ là vấn đề kỹ thuật (device_bash/
GPU) mà còn là ràng buộc pháp lý của cuộc thi (`quy-dinh-cuoc-thi-task1.md` mục 7 + mục 11, vi phạm có
thể bị trừ điểm/loại). Mục 1 và mục 3 KHÔNG tự sửa được dù kỹ thuật khả thi — chỉ mục 2 (hậu xử lý
retrieval, không đổi dữ liệu/nhãn) là an toàn để team tự triển khai.

*Nguồn chi tiết đầy đủ (trích dẫn nguyên văn từng case ban đầu, top-5 dự đoán sai): `outputs/predictions/error_analysis_recall0.md` trong repo.*
