# Hybrid Legal Retrieval Pipeline (`src/retrieval`)

This document describes the design, architecture, module responsibilities, input artifacts, and test execution for the Core Retrieval pipeline in the **UIT Data Science 2026** competition.

---

## 1. Pipeline Architecture

The retrieval system employs a hybrid search strategy combining sparse BM25 keyword matching with dense vector semantic search, fused via Reciprocal Rank Fusion (RRF), aggregated to legal document candidates, and re-ranked using a CrossEncoder model.

```text
                     Question (User Query)
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     BM25 Retriever                        Dense Retriever
(embeddings/metadata.jsonl)         (vectorstore/corpus.index)
   Top-100 sparse chunks                Top-100 dense chunks
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
                    Reciprocal Rank Fusion
                           (RRF)
                               ▼
                   Chunk -> Document Aggregation
                     (max RRF score per doc_id)
                               ▼
                   Top-20 Candidate Documents
                               ▼
                          Reranker
               (AITeamVN/Vietnamese_Reranker)
                               ▼
                  Top-5 Final Document IDs
```

### Flow Execution Steps
1. **Query Encoding & Search**: The input question is searched concurrently using BM25 over the pre-segmented corpus passages and Dense Retrieval via FAISS index lookup.
2. **Chunk Fusion**: Ranked chunk results from both sparse (Top-100) and dense (Top-100) branches are fused at the chunk level using Reciprocal Rank Fusion (RRF).
3. **Document Aggregation**: Chunks belonging to the same legal document (`doc_id`) are aggregated by assigning each document the maximum RRF score among its retrieved chunks. Top-20 candidate documents are selected.
4. **Cross-Encoder Reranking**: Candidate documents (represented by their highest-scoring chunk passage) are reranked using a pre-trained Vietnamese CrossEncoder.
5. **Output Selection**: The top 5 document IDs are returned as the final output.

---

## 2. Module Responsibilities

The codebase in `src/retrieval/` is structured cleanly into decoupled components:

- **[`types.py`](file:///media/brolyn-ntg/ext4-SSD/Code/UIT-data-science-2026/src/retrieval/types.py)**: Shared data contracts and dataclasses:
  - `RetrievalResult`: Single chunk-level retrieval output (`chunk_index`, `doc_id`, `score`, `chunk_id`).
  - `DocumentCandidate`: Document-level candidate represented by its top chunk (`doc_id`, `chunk_index`, `score`, `chunk_id`).
  - `RerankResult`: Final reranked document output (`doc_id`, `score`, `chunk_index`, `chunk_id`).
- **[`bm25.py`](file:///media/brolyn-ntg/ext4-SSD/Code/UIT-data-science-2026/src/retrieval/bm25.py)**: Implements `BM25Retriever` using `rank_bm25.BM25Okapi` over whitespace-tokenized passages from `embeddings/metadata.jsonl`.
- **[`dense.py`](file:///media/brolyn-ntg/ext4-SSD/Code/UIT-data-science-2026/src/retrieval/dense.py)**: Implements `DenseRetriever` using `SentenceTransformer` and FAISS index search over `vectorstore/corpus.index`.
- **[`fusion.py`](file:///media/brolyn-ntg/ext4-SSD/Code/UIT-data-science-2026/src/retrieval/fusion.py)**: Implements `rrf_fuse` (rank-based score fusion) and `aggregate_chunks_to_documents` (max-score chunk-to-doc mapping).
- **[`reranker.py`](file:///media/brolyn-ntg/ext4-SSD/Code/UIT-data-science-2026/src/retrieval/reranker.py)**: Implements `Reranker` using `sentence_transformers.CrossEncoder`.
- **[`pipeline.py`](file:///media/brolyn-ntg/ext4-SSD/Code/UIT-data-science-2026/src/retrieval/pipeline.py)**: Orchestrates the end-to-end pipeline (`RetrievalPipeline`) and manages GPU memory handoff between the Dense encoder and the CrossEncoder model.

---

## 3. Input Artifacts & Strict Invariant

### Input Files
- **`embeddings/metadata.jsonl`**: JSONL file containing 351,394 metadata rows (`doc_id`, `chunk_id`, `legal_section`, `passage`).
- **`vectorstore/corpus.index`**: FAISS `IndexFlatIP` index file containing 351,394 normalized dense embedding vectors (768-dimension).

### Core Invariant
> **Strict 1-to-1 Index Alignment**:
> `metadata[i]` ↔ `corpus_embeddings[i]` ↔ `FAISS index vector i`

The integer index `i` (0-based) across all three artifacts corresponds to the exact same text chunk. The retrieval implementation relies on this invariant and does not rebuild or re-index the handed-off artifacts.

---

## 4. Models & Formatting Specifications

### Dense Embedding Model
- **Model Name**: `bqbbao6/VN_legal_embedding_512`
- **Output Vector Dimension**: `768` (Note: context token limit is 512, vector output dimension is 768).
- **Query Prefix**: `"query: "` (asymmetric retrieval setup).
- **Normalization**: `normalize_embeddings=True` (Cosine similarity via FAISS Inner Product `IndexFlatIP`).

### Reranker Model
- **Model Name**: `AITeamVN/Vietnamese_Reranker`
- **HuggingFace Hub**: [https://huggingface.co/AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker)
- **Input Format**: Pairs of `(question, representative_chunk_passage)`.

---

## 5. Development Configuration

The following hyperparameter values are **development defaults** used for functional correctness and pipeline execution. They are **not** tuned or grid-searched values:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `BM25_TOP_K` | `100` | Number of chunks retrieved by BM25 |
| `DENSE_TOP_K` | `100` | Number of chunks retrieved by Dense FAISS search |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant denominator offset |
| `CANDIDATE_DOCS` | `20` | Top document candidates passed to Reranker after aggregation |
| `FINAL_TOP_K` | `5` | Final top document IDs returned for Task 1 |

> [!NOTE]
> Hyperparameter tuning, grid search, and fine-tuning are reserved for downstream evaluation stages (`src/evaluation/` and `src/training/`).

---

## 6. Task 1 Output Constraints

Task 1 requires returning a list of legal document IDs for each question.
- **Maximum Output Limit**: A maximum of **5 document IDs** (`FINAL_TOP_K <= 5`) per query.
- **Evaluation Constraint**: Returning more than 5 document IDs results in a score of 0 for both Recall and Precision on that query. The `RetrievalPipeline` explicitly validates `final_top_k <= 5` upon initialization.

---

## 7. Running Unit & Integration Tests

All tests are located in `tests/` and can be run using Python in the `uit-retrieval` environment:

### Run BM25 Test
```bash
PYTHONPATH=. python tests/test_retrieval_bm25.py
```

### Run Dense Retrieval Test
```bash
PYTHONPATH=. python tests/test_retrieval_dense.py
```

### Run RRF & Aggregation Test
```bash
PYTHONPATH=. python tests/test_retrieval_rrf.py
```

### Run Reranker Test
```bash
PYTHONPATH=. python tests/test_retrieval_reranker.py
```

### Run Full End-to-End Pipeline Test
```bash
PYTHONPATH=. python tests/test_retrieval_pipeline.py
```
