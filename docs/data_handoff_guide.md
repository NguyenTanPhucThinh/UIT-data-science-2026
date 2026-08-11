# Data Handoff & Integration Guide
**For: Thành viên 2 (Storage & Representation) & Thành viên 3 (Core RAG)**
**From: Đoàn Quốc Kiên (Data & Utilities)**

This document serves as the official integration guide for the datasets produced in `data/processed/`. It explains the data schemas, the chunking strategy, and the recommended retrieval strategies based on our Data & Utilities pipeline (Phase 1-4).

---

## 1. Output Datasets Overview

All processed data is located in `data/processed/`. You will primarily interact with two levels of data:

1. **Document-Level Data**: `corpus_preprocessed.jsonl`
   - Contains the full, normalized text of all 8,532 documents.
   - **Use Case**: Indexing the first ~512 tokens (title + opening articles) to catch broad topic matches, or for BM25 full-text indexing.

2. **Chunk-Level Data**: `chunks_corpus.jsonl`
   - Contains 351,394 semantically split chunks.
   - **Use Case**: Dense vector embedding indexing (Qdrant/FAISS) to catch specific legal clause matches.

---

## 2. Chunking Strategy & Token Limits (Crucial for Embeddings)

- **Semantic Splitting**: Chunks were split structurally along `Điều` (Article), `Chương` (Chapter), and `Mục` boundaries. This preserves self-contained legal clauses.
- **Context Window Protection**: The allowed `vietnamese-legal-embedding` model has a strict 512-token limit. 
- **Validation**: Our validation script guarantees that **no chunk exceeds 400 words**, meaning you will **not** suffer from silent truncation during vector encoding.
- **Empty Passages**: The 20 documents that originally had empty passages have been recovered using URL fallbacks and are guaranteed to have at least 1 chunk.

---

## 3. Data Schemas

### Chunk-Level Schema (`chunks_corpus.jsonl`)
This is the primary file for Member 2 to index into the Vectorstore.

```json
{
  "chunk_id": "245154_chunk_003",
  "doc_id": "245154",
  "doc_name": "Bo-luat-hinh-su-2015-296661",
  "doc_type": "Luật",
  "chunk_index": 3,
  "total_chunks": 47,
  "legal_section": "Điều 3. Nguyên tắc xử lý ...",
  "passage": "Điều 3. Nguyên tắc xử lý...",
  "passage_segmented": "Điều 3 . Nguyên_tắc xử_lý..."
}
```
*Note: Use `passage_segmented` if you are building BM25 sparse indices, as it has been segmented using the highly accurate `VnCoreNLP` engine. Use `passage` for standard dense embedding models unless the embedding model specifically requires underscored words.*

### Query & Train Schema (`train_preprocessed.jsonl`)
```json
{
  "question_id": "ab12c3",
  "question": "Mức phạt lỗi chạy quá tốc độ là bao nhiêu?",
  "question_segmented": "Mức phạt lỗi chạy quá_tốc_độ là bao_nhiêu ?",
  "answer_doc_ids": ["50885"],
  "num_answers": 1
}
```

---

## 4. Retrieval & Aggregation Strategy (For Member 3)

### 🎯 Document-Level Retrieval, Not QA
The final task output requires **Document IDs**, not answer spans. Because a single document is split into multiple chunks, you must aggregate chunk scores back to the document level.

**Recommended Aggregation Methods:**
1. **Max-Score (Default & Recommended)**: The score of a document is the score of its highest-matching chunk. (If a specific `Điều` perfectly matches the query, the whole document is retrieved).
2. **Top-K Voting**: Count how many chunks from a document appear in the top 100 retrieved chunks.

### 🧩 Multi-Granularity Indexing
Because 50% of our documents are relatively short (< 23K chars), consider a hybrid index approach:
- **Index 1**: The chunks from `chunks_corpus.jsonl`.
- **Index 2**: The Document-level summary (first 512 tokens of `corpus_preprocessed.jsonl`).
If a query matches the document-level summary, boost the document's overall score.

---

## 5. NLP Engine Note
We successfully implemented **VnCoreNLP** (running on 8 parallel JVM workers) instead of PyVi. This provides significantly higher accuracy for compound legal terms (e.g., `bảo_hiểm_y_tế`), which directly improves BM25 retrieval quality. 

Please ensure any query processing you do dynamically in Phase 5 also uses `VnCoreNLP` to match the index segmentation!
