# Phân chia công việc dự án UIT-datascience2026

Dựa trên cấu trúc thư mục `src`, dưới đây là phương án phân chia công việc cho 5 thành viên. Mỗi phần được thiết kế để gom nhóm các công việc có sự liên kết chặt chẽ với nhau.

## 👤 Thành viên 1 + 5: Data & Utilities (Dữ liệu & Tiện ích)
**Thư mục đảm nhiệm:** `src/ingestion + EDA` và `src/utils`

*   **Trách nhiệm:** 
    *   Thu thập, làm sạch và xử lý dữ liệu đầu vào.
    *   Thực hiện phân tích dữ liệu thăm dò (EDA) để hiểu rõ đặc tính dữ liệu.
    *   Xây dựng các hàm tiện ích (utils) dùng chung cho toàn bộ dự án (ví dụ: logger, xử lý file, định dạng text).
    *  Thành viên thực hiện: Đoàn Quốc Kiên, Bùi Diên Gia Bảo
    *  Input: data/raw      
    *  Output: data/processed 
## 👤 Thành viên 2 + 5: Storage & Representation (Lưu trữ & Biểu diễn)
**Thư mục đảm nhiệm:** `src/embeddings` và `src/vectorstore`

*   **Trách nhiệm:** 
    *   Thử nghiệm và tích hợp các mô hình nhúng (embedding models) để chuyển đổi text thành vector.
    *   Thiết lập, tối ưu hóa và quản lý Vector Database (như Chroma, FAISS, Milvus...).
    *   Đảm bảo việc lưu trữ và index dữ liệu diễn ra hiệu quả.
    *   Thành viên thực hiện: Nguyễn Tấn Phúc Thịnh, Bùi Diên Gia Bảo
    *  Input: data/processed      
    *  Output: FAISS local (hoặc Qdrant tự host local; không dùng cloud/API bên thứ ba)
## 👤 Thành viên 3 + 5: Core RAG (Cốt lõi Tìm kiếm & Hỏi đáp)
**Thư mục đảm nhiệm:** `src/retrieval` + `src/reranker` + `src/qa`

*   **Trách nhiệm:** 
    *   Xây dựng logic truy xuất (retrieval): lấy các tài liệu liên quan nhất từ Vectorstore dựa trên câu hỏi.
    *   Phát triển module QA: tích hợp LLM để sinh ra câu trả lời dựa trên context đã truy xuất được.
    *   Tối ưu prompt engineering.
    *   Thành viên thực hiện: Nguyễn Trường Giang, Bùi Diên Gia Bảo
    *   Input: FAISS local / retrieval artifacts cục bộ
    *   Output: JSON Task 1 (document IDs) hoặc Task 2 (answer dạng chuỗi)
## 👤 Thành viên 4 + 5: Model Tuning & Quality (Huấn luyện & Đánh giá)
**Thư mục đảm nhiệm:** `src/training` và `src/evaluation`

*   **Trách nhiệm:** 
    *   Fine-tune mô hình (nếu có yêu cầu huấn luyện lại LLM hoặc embedding model trên dữ liệu riêng).
    *   Thiết lập các metrics đánh giá hệ thống (BLEU, ROUGE, độ chính xác của retrieval, RAGAS...).
    *   Chạy các bài test để đo lường chất lượng câu trả lời.
    *   Thành viên thực hiện: Nguyễn Huy Hoàng, Bùi Diên Gia Bảo
    *   Input: 2 mô hình embedding và re-ranking trên hugging face, submission.json
    *   Output: Mô hình embedding và reranking 

## Thành viên 5: Quản lý quy trình
**Thư mục đảm nhiệm:**: all
---
> **Lưu ý làm việc nhóm:**
> - Sử dụng chung các biến ở `src/config` ngay từ đầu để tránh hard-code.
> - Thành viên 1 và 5 cần thống nhất input/output format sớm với các thành viên khác.
> - Hãy dùng GitHub/GitLab với quy trình Pull Request (PR) để tránh conflict mã nguồn.
