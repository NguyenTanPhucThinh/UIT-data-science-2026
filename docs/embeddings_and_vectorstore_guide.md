# Hướng Dẫn: Thư Mục `embeddings/` và `vectorstore/`


## 1. Tổng Quan

Hai thư mục `embeddings/` và `vectorstore/` là **đầu ra (output) của bước nhúng văn bản** trong pipeline RAG. Chúng được tạo ra **tự động** bởi script `src/embeddings/create_embeddings.py` và là nguồn dữ liệu cho module `src/retrieval/` ở bước tiếp theo.

```
data/processed/chunks_corpus.jsonl
        │
        ▼
src/embeddings/create_embeddings.py
        │
        ├──► embeddings/corpus_embeddings.npy   (vector thô dạng NumPy)
        ├──► embeddings/metadata.jsonl          (metadata từng chunk)
        └──► vectorstore/corpus.index           (FAISS index để tìm kiếm)
```

### Mô Hình Embedding Sử Dụng

| Thuộc tính | Giá trị |
|---|---|
| **Tên mô hình** | `bqbbao6/VN_legal_embedding_512` |
| **Nguồn** | HuggingFace Hub |
| **Loại** | SentenceTransformer |
| **Chiều vector (dimension)** | 512 (tương ứng token limit) |
| **Ngôn ngữ** | Tiếng Việt, chuyên biệt văn bản pháp lý |
| **Thiết bị** | GPU (CUDA) nếu có, fallback CPU |

> **Lý do chọn mô hình này:** Được huấn luyện đặc biệt trên văn bản pháp lý tiếng Việt, phù hợp với corpus `selected-contexts` của cuộc thi. Giới hạn 512 token khớp hoàn toàn với chiến lược chunking ≤ 400 từ của Phase 4.

---

## 2. Cấu Trúc Thư Mục

```
UIT-datascience2026/
│
├── embeddings/                     ← Lưu trữ vector và metadata
│   ├── corpus_embeddings.npy       ← Ma trận embedding (float32, shape: [N, 512])
│   └── metadata.jsonl              ← Metadata tương ứng từng vector (N dòng)
│
├── vectorstore/                    ← Lưu trữ FAISS index để tìm kiếm nhanh
│   └── corpus.index                ← FAISS IndexFlatIP, tương đương Cosine Similarity
│
└── src/
    └── embeddings/
        └── create_embeddings.py    ← Script sinh ra 3 file trên
```

> **Lưu ý về kích thước file:** Cả 3 file output đều **rất lớn** (~1 GB mỗi file). Chúng đã được liệt kê trong `.gitignore` và **KHÔNG được commit lên Git**. Mỗi thành viên cần chạy script để tự sinh ra các file này trên máy cục bộ.

---

## 3. Quy Trình Tạo Embeddings

Script `create_embeddings.py` thực hiện theo trình tự sau:

```
┌─────────────────────────────────────────────────────────────┐
│                   create_embeddings.py                      │
│                                                             │
│  1. Đọc chunks_corpus.jsonl (đầu vào từ Phase 4)            │
│                    │                                        │
│  2. Tải SentenceTransformer model từ HuggingFace            │
│                    │                                        │
│  3. Khởi tạo FAISS IndexFlatIP(dimension=512)               │
│                    │                                        │
│  4. Lặp từng dòng, xử lý theo batch (batch_size=2048):      │
│     ├─ Thêm prefix "passage: " vào text                     │
│     ├─ Ghi metadata vào metadata.jsonl                      │
│     ├─ Encode batch → vector float32                        │
│     ├─ Normalize L2 (để IndexFlatIP ≡ Cosine Similarity)    │
│     └─ Thêm batch vào FAISS index                           │
│                    │                                        │
│  5. Stack toàn bộ vector → lưu corpus_embeddings.npy        │
│  6. Lưu FAISS index → vectorstore/corpus.index              │
└─────────────────────────────────────────────────────────────┘
```

### Tại sao dùng prefix `"passage: "`?

Mô hình `bqbbao6/VN_legal_embedding_512` được huấn luyện theo kiến trúc **bi-encoder asymmetric** (tương tự E5):
- Văn bản trong corpus được encode với prefix `"passage: "`.
- Câu hỏi của người dùng được encode với prefix `"query: "`.

Module `src/retrieval/` **phải thêm prefix `"query: "`** khi encode câu hỏi để đảm bảo độ tương đồng ngữ nghĩa chính xác.

### Tại sao dùng `IndexFlatIP` + L2 normalize?

`IndexFlatIP` (Inner Product) trên vector đã chuẩn hóa L2 cho kết quả **hoàn toàn tương đương Cosine Similarity**. Đây là cách chuẩn để thực hiện semantic search với FAISS mà không cần chỉ số `IndexFlatL2`.

---

## 4. Chi Tiết Thư Mục `embeddings/`

### 4.1. `corpus_embeddings.npy`

| Thuộc tính | Mô tả |
|---|---|
| **Định dạng** | NumPy array (`.npy`) |
| **Shape** | `(N, 512)` — N = tổng số chunk trong corpus |
| **Dtype** | `float32` |
| **Đã normalize** | Có (L2 normalize) |
| **Kích thước ước tính** | ~1 GB |

**Cách tải và sử dụng:**
```python
import numpy as np

embeddings = np.load("embeddings/corpus_embeddings.npy")
# embeddings.shape → (N, 512)
# embeddings[i] là vector của chunk thứ i (khớp với dòng thứ i trong metadata.jsonl)
```

**Mục đích:** File này lưu vector thô để có thể:
- Tái sử dụng mà không cần inference lại từ model.
- Tính toán thống kê, phân tích phân phối vector.
- Nạp vào các vector database khác nếu cần (Qdrant, ChromaDB...).

---

### 4.2. `metadata.jsonl`

Mỗi dòng là một JSON object tương ứng với **một chunk** (cùng thứ tự với vector trong `corpus_embeddings.npy` và FAISS index).

**Schema mỗi dòng:**
```json
{
  "doc_id": "12345_ND",
  "chunk_id": "12345_ND_0",
  "legal_section": "Điều 5",
  "passage": "Nội dung đoạn văn bản pháp lý đã qua word segmentation..."
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `doc_id` | `str` | ID của văn bản pháp lý gốc |
| `chunk_id` | `str` | ID duy nhất của chunk (format: `<doc_id>_<index>`) |
| `legal_section` | `str` | Tên mục pháp lý (`Điều`, `Khoản`, `Chương`...) |
| `passage` | `str` | Nội dung văn bản gốc của chunk (đã word-segment) |

**Cách tải metadata:**
```python
import json

metadata = []
with open("embeddings/metadata.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))

# Truy cập chunk thứ i
chunk_info = metadata[i]
print(chunk_info["doc_id"], chunk_info["passage"])
```

> **Quan trọng:** Index `i` của metadata **khớp 1-1** với index `i` trong `corpus_embeddings.npy` và FAISS. Khi FAISS trả về `indices`, dùng chính các index đó để tra cứu metadata.

---

## 5. Chi Tiết Thư Mục `vectorstore/`

### 5.1. `corpus.index`

| Thuộc tính | Mô tả |
|---|---|
| **Định dạng** | FAISS binary index |
| **Loại index** | `IndexFlatIP` (Flat Inner Product) |
| **Số lượng vector** | N (bằng số chunk trong corpus) |
| **Kích thước ước tính** | ~1 GB |

**Đặc điểm `IndexFlatIP`:**
- **Brute-force search** — duyệt toàn bộ N vector để tìm top-k gần nhất.
- **Chính xác 100%** — không có sai số xấp xỉ (không phải ANN).
- **Phù hợp** khi N < vài triệu vector và cần độ chính xác cao cho bài thi.

**Cách tải và truy vấn:**
```python
import faiss
import numpy as np

# Tải index
index = faiss.read_index("vectorstore/corpus.index")

# Encode câu hỏi (prefix "query: " + normalize L2)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("bqbbao6/VN_legal_embedding_512", trust_remote_code=True)

query = "query: Điều kiện để được hưởng bảo hiểm y tế là gì?"
query_vector = model.encode([query], normalize_embeddings=True)  # shape: (1, 512)

# Tìm top-k chunks liên quan nhất
top_k = 5
scores, indices = index.search(query_vector, top_k)
# scores: (1, top_k) — điểm cosine similarity (sau normalize)
# indices: (1, top_k) — index của các chunk trong corpus
```

**Tra cứu metadata sau khi có indices:**
```python
import json

metadata = []
with open("embeddings/metadata.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))

retrieved_chunks = []
for idx, score in zip(indices[0], scores[0]):
    chunk = metadata[idx]
    chunk["score"] = float(score)
    retrieved_chunks.append(chunk)

# retrieved_chunks: danh sách top-k chunk có score cosine similarity cao nhất
```

---

## 6. Hướng Dẫn Chạy Script

### Yêu Cầu Trước

- [ ] File `data/processed/chunks_corpus.jsonl` đã tồn tại (chạy pipeline Phase 1–4 trước).
- [ ] Đã cài đặt các dependencies:

```bash
pip install sentence-transformers>=3.0.0 faiss-cpu numpy>=1.26.0 tqdm torch
```

> **Lưu ý GPU:** Nếu có GPU NVIDIA, cài `faiss-gpu` thay cho `faiss-cpu` để tăng tốc search. Script tự động dùng CUDA nếu có.

### Chạy Script

Chạy từ **thư mục gốc** của dự án (không phải từ trong `src/`):

```bash
# Từ thư mục UIT-datascience2026/
python src/embeddings/create_embeddings.py
```

### Kết Quả Mong Đợi

```
Using device: cuda          (hoặc cpu)
Loading model bqbbao6/VN_legal_embedding_512...
Reading data from data/processed/chunks_corpus.jsonl...
Generating embeddings...
Processing and Embedding: 100%|████████| XXXXX/XXXXX [...]
Saving embeddings array...
Saving FAISS index to vectorstore/corpus.index...
Embedding process completed successfully.
- Metadata saved at: embeddings/metadata.jsonl
- Embeddings saved at: embeddings/corpus_embeddings.npy
- Vectorstore saved at: vectorstore/corpus.index
```

### Thời Gian Ước Tính

| Thiết bị | Thời gian ước tính |
|---|---|
| GPU (NVIDIA RTX 3090+) | ~10–20 phút |
| GPU (NVIDIA T4 / Colab) | ~20–40 phút |
| CPU (8 cores) | ~2–4 giờ |

---

## 7. Tích Hợp Với Pipeline Retrieval

Module `src/retrieval/` sẽ sử dụng cả hai thư mục theo mô hình sau:

```python
# Ví dụ minh họa cách module retrieval tích hợp
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self):
        self.model = SentenceTransformer(
            "bqbbao6/VN_legal_embedding_512",
            trust_remote_code=True
        )
        self.index = faiss.read_index("vectorstore/corpus.index")
        
        self.metadata = []
        with open("embeddings/metadata.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        # Prefix "query: " là BẮT BUỘC với mô hình bi-encoder asymmetric
        query_text = "query: " + question
        query_vec = self.model.encode(
            [query_text],
            normalize_embeddings=True
        )
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            result = self.metadata[idx].copy()
            result["score"] = float(score)
            results.append(result)
        return results
```

---

## 8. [BẮT BUỘC] Prefix `"query: "` Khi Retrieval

> **Đây là yêu cầu kỹ thuật bắt buộc, không phải tùy chọn.** Bỏ qua bước này sẽ khiến kết quả semantic search bị sai lệch nghiêm trọng dù mọi thứ khác đều đúng.

### Tại Sao Phải Có Prefix?

Mô hình `bqbbao6/VN_legal_embedding_512` được huấn luyện theo kiến trúc **bi-encoder bất đối xứng (asymmetric bi-encoder)**, tương tự họ mô hình E5/multilingual-E5. Kiến trúc này yêu cầu **hai loại vector khác nhau** tùy theo vai trò của văn bản:

| Đầu vào | Prefix bắt buộc | Nơi thực hiện | Thời điểm |
|---|---|---|---|
| Văn bản corpus (pháp lý) | `"passage: "` | `create_embeddings.py` | Lúc build index (1 lần) |
| Câu hỏi của người dùng | `"query: "` | `src/retrieval/` | Mỗi lần có câu hỏi mới |

Nếu encode câu hỏi **không có prefix** hoặc **dùng sai prefix `"passage: "`**, vector câu hỏi sẽ nằm ở vùng không gian khác với vector corpus → cosine similarity trả về kết quả gần như ngẫu nhiên.

### Cách Thêm Prefix Đúng

**✅ ĐÚNG — Luôn thêm `"query: "` trước khi encode câu hỏi:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("bqbbao6/VN_legal_embedding_512", trust_remote_code=True)

question = "Điều kiện để được hưởng trợ cấp thất nghiệp là gì?"

# BẮT BUỘC: thêm prefix "query: "
query_text = "query: " + question
query_vec = model.encode([query_text], normalize_embeddings=True)
```

**❌ SAI — Encode trực tiếp không có prefix:**

```python
# KHÔNG làm thế này — kết quả retrieval sẽ sai!
query_vec = model.encode([question], normalize_embeddings=True)

# KHÔNG dùng sai prefix passage:
query_vec = model.encode(["passage: " + question], normalize_embeddings=True)
```

### Ảnh Hưởng Thực Tế

Thực nghiệm với các mô hình cùng họ E5 cho thấy việc bỏ prefix làm giảm **10–20% Recall@5** — tương đương mất đi toàn bộ lợi thế của dense retrieval so với BM25 baseline.

### Checklist Cho Module `src/retrieval/`

Khi viết hoặc review code retrieval, hãy kiểm tra:

- [ ] Tất cả câu hỏi đều được thêm `"query: "` trước khi `model.encode()`
- [ ] Đặt `normalize_embeddings=True` trong lệnh encode
- [ ] Không encode câu hỏi với prefix `"passage: "`
- [ ] Không encode câu hỏi hoàn toàn không có prefix nào

