# UIT DATA SCIENCE CHALLENGE 2026
## Task 1 – Legal Information Retrieval (LegalIR)
**Truy vấn thông tin pháp luật tiếng Việt**

**Mục tiêu:** Với một câu hỏi pháp luật, hệ thống xác định văn bản hành chính/pháp lý chứa thông tin cần thiết và trả về ID của văn bản đó.

### 1. Mô tả bài toán
- **Input:** Một câu hỏi tiếng Việt cần truy vấn thông tin pháp luật.
- **Output:** Danh sách ID của các văn bản có thông tin cần thiết để trả lời câu hỏi.

**Ví dụ:**
```json
{
  "id": {
    "question": "Việc tổ chức vận động, tiếp nhận, sử dụng nguồn đóng góp tự nguyện được thực hiện dựa trên nguyên tắc nào?",
    "answer": [
      "177504"
    ]
  }
}
```

### 2. Cấu trúc dữ liệu
*(Sẽ được cung cấp tương ứng theo timeline cuộc thi)*

| Tệp | Nội dung |
|---|---|
| `train.json` | Tập dữ liệu huấn luyện cho các đội phát triển phương pháp. |
| `warmup.json` | Tập dữ liệu mẫu phục vụ vòng Warm-up, giúp làm quen bài toán và quy trình submission. |
| `public-official.json` | Tập dữ liệu dùng trong giai đoạn Public Test theo cấu hình chính thức. |
| `private-official.json` | Tập dữ liệu chính thức của Private Test. |
| `selected-contexts.zip` | Kho văn bản được chọn; gồm nhiều tệp `context_*.json`. |

### 3. Ví dụ cấu trúc một văn bản
Tệp `context_*.json` bao gồm các trường:
- `id`: Mã định danh duy nhất của văn bản
- `name`: Tiêu đề văn bản
- `link`: Đường dẫn nguồn
- `passage`: Nội dung văn bản được sử dụng để truy vấn

**Ví dụ:**
```json
{
  "link": "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Quyet-dinh-5868-QD-BYT-2018-co-cau-to-chuc-cua-Vu-Trang-thiet-bi-va-Cong-trinh-y-te-396608.aspx",
  "name": "Quyet-dinh-5868-QD-BYT-2018-co-cau-to-chuc-cua-Vu-Trang-thiet-bi-va-Cong-trinh-y-te-396608",
  "passage": "BỘ Y TẾ\r\n\n  -------\n\nCỘNG HÒA XÃ HỘI\r\n\n  CHỦ NGHĨA VIỆT NAM\r\n\n  Độc lập - Tự do - Hạnh phúc \r\n\n  ---------------\n\nSố: 5868/QĐ-BYT\n\nHà Nội, ngày 28\r\n\n  tháng 9 năm 2018\n\n\n\nQUYẾT ĐỊNH\n\nQUY\r\n\nĐỊNH CHỨC NĂNG, NHIỆM VỤ, QUYỀN HẠN VÀ CƠ CẤU TỔ CHỨC CỦA VỤ TRANG THIẾT BỊ VÀ CÔNG\r\n\nTRÌNH Y TẾ THUỘC BỘ Y TẾ\n\nBỘ TRƯỞNG BỘ Y TẾ\n\nCăn cứ Nghị định số 75/2017/NĐ-CP … - Lưu: VT, TCCB, TTB, PC.\n\nBỘ TRƯỞNG\n\n\r\n\n  Nguyễn Thị Kim Tiến\n\n",
  "id": 740
}
```

### 4. Đánh giá tác vụ
Hệ thống cần trả về danh sách các `document_id` liên quan. Điểm xếp hạng sử dụng **Recall** làm độ đo chính và **Precision** làm độ đo phụ (được sử dụng khi các đội có cùng điểm Recall).

| Độ đo | Vai trò | Mô tả |
|---|---|---|
| **Recall** | Độ đo chính | Đo tỷ lệ các văn bản đúng được hệ thống truy xuất. |
| **Precision** | Độ đo phụ | Đo tỷ lệ các văn bản hệ thống truy xuất là đúng. |

#### Recall - Độ đo chính
Recall đo tỷ lệ các văn bản đúng được hệ thống truy xuất. Đối với mỗi câu hỏi $i$:

$$Recall_i = \frac{|Relevant_i \cap Predicted_i|}{|Relevant_i|}$$

Trong đó:
- $Relevant_i$ là tập các `document_id` đúng của câu hỏi thứ $i$.
- $Predicted_i$ là tập các `document_id` mà hệ thống trả về.

Điểm Recall cuối cùng được tính bằng trung bình trên toàn bộ câu hỏi:
$$Recall = \frac{1}{N} \sum_{i=1}^{N} Recall_i$$
*(Trong đó $N$ là tổng số câu hỏi)*

> Recall có giá trị trong khoảng 0 đến 1, trong đó Recall càng cao càng tốt.

#### Precision - Độ đo phụ
Precision đo tỷ lệ các văn bản mà hệ thống truy xuất là đúng. Đối với mỗi câu hỏi $i$:

$$Precision_i = \frac{|Relevant_i \cap Predicted_i|}{|Predicted_i|}$$

Trong đó:
- $Relevant_i$ là tập các `document_id` đúng.
- $Predicted_i$ là tập các `document_id` được hệ thống trả về.

Điểm Precision cuối cùng được tính bằng trung bình trên toàn bộ câu hỏi:
$$Precision = \frac{1}{N} \sum_{i=1}^{N} Precision_i$$

*Lưu ý: Nếu hệ thống không trả về văn bản nào cho một câu hỏi thì Precision của câu hỏi đó được tính bằng 0.*
> Precision có giá trị trong khoảng 0 đến 1, trong đó Precision càng cao càng tốt.

### 5. Hướng dẫn nộp bài
Thí sinh nộp một tệp `submission.zip` chứa duy nhất tệp `submission.json`.

`submission.json` có định dạng JSON Object:
```json
{
  "147194": {
    "answer": ["177504", "740"]
  }
}
```

#### Ràng buộc đánh giá
- Đối với mỗi câu hỏi, hệ thống chỉ đánh giá **tối đa 05 `document_id`**.
- Nếu kết quả dự đoán của một câu hỏi chứa nhiều hơn 05 `document_id`, hệ thống sẽ gán Recall và Precision bằng 0 cho câu hỏi đó.
- Điểm Recall và Precision của toàn bộ bài nộp được tính bằng trung bình trên tất cả các câu hỏi theo công thức ở mục 4, **bao gồm cả các câu hỏi vi phạm ràng buộc trên**.
