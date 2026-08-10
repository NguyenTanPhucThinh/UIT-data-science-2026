# Cấu Trúc Thư Mục Dự Án RAG

Tài liệu này giải thích chi tiết mục đích và chức năng của từng thư mục trong mã nguồn của dự án (nằm trong thư mục `src/`), giúp các thành viên trong nhóm dễ dàng nắm bắt workflow và quy trình phân chia công việc.

## 1. Dữ liệu và Tiền xử lý

- **`data/`**: Nơi lưu trữ các file dữ liệu. Thường bao gồm dữ liệu thô (raw) chưa qua xử lý và dữ liệu đã được làm sạch (processed).
- **`src/ingestion/`**: Chịu trách nhiệm cho bước chuẩn bị đầu vào. Code tại đây sẽ:
  - Đọc dữ liệu thô từ thư mục `data/`.
  - Làm sạch văn bản (loại bỏ ký tự rác, chuẩn hóa text).
  - Cắt nhỏ văn bản (Chunking) thành các đoạn có độ dài phù hợp để xử lý.

## 2. Lưu trữ và Nhúng (Embedding)

- **`src/embeddings/`**: Nơi định nghĩa và cấu hình các mô hình nhúng (Embedding Models - ví dụ: SentenceTransformer, OpenAI API). Các mô hình này dùng để chuyển đổi các đoạn văn bản (chunk) hoặc câu hỏi của người dùng thành các vector toán học.
- **`src/vectorstore/`**: Quản lý việc kết nối và thao tác với Vector Database (như FAISS, ChromaDB, Qdrant, Milvus). Nơi đây lưu trữ các vector được tạo ra từ `embeddings` để phục vụ cho việc tìm kiếm sau này.

## 3. Các Luồng Task Chính (RAG Workflow)

- **`src/retrieval/` (Task 1 - Information Retrieval):** Chứa logic tìm kiếm thông tin. Khi có câu hỏi, module này sẽ nhúng câu hỏi đó thành vector, sau đó truy vấn (query) vào `vectorstore` để lấy ra (retrieve) các đoạn tài liệu (chunks) có mức độ liên quan cao nhất. Các kỹ thuật như Semantic Search, BM25, hoặc Re-ranking cũng được cài đặt ở đây.
- **`src/qa/` (Task 2 - Question Answering / Generation):** Đảm nhận việc sinh câu trả lời bằng Trí tuệ Nhân tạo (LLMs). Module này sẽ kết hợp "Câu hỏi của người dùng" và "Các tài liệu đã được retrieve" vào một Prompt Template, sau đó gọi LLM để đọc, suy luận và đưa ra câu trả lời cuối cùng chính xác dựa trên bối cảnh.
- **`src/pipeline/`**: Nơi "lắp ghép" toàn bộ các module trên lại với nhau thành một quy trình xuyên suốt (End-to-End). Ví dụ: `pipeline/rag.py` sẽ định nghĩa một hàm duy nhất để nhận câu hỏi đầu vào, tự động gọi `retrieval` đi tìm tài liệu, chuyển tài liệu đó cho `qa` để sinh câu trả lời và trả kết quả về cho người dùng.

## 4. Các Thành Phần Bổ Trợ


- **`src/config/`**: Nơi lưu trữ các thông số thiết lập, biến môi trường, đường dẫn file và cấu hình hyperparameter để dễ dàng thay đổi mà không cần can thiệp sâu vào code.
- **`src/utils/`**: Chứa các hàm hỗ trợ chung (helper functions) được dùng chung ở nhiều thư mục khác nhau, giúp code không bị lặp lại.
