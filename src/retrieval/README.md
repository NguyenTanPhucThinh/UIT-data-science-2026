# Core Retrieval Modules (`src/retrieval`)

This package implements the core hybrid retrieval pipeline for the **UIT Data Science 2026** RAG system:

`BM25 + Dense -> RRF -> Chunk-to-Document Aggregation -> CrossEncoder Reranker -> Top-5 Document IDs`

For detailed technical documentation, architecture diagrams, module responsibilities, invariants, and test instructions, see:
👉 **[`docs/retrieval.md`](file:///media/brolyn-ntg/ext4-SSD/Code/UIT-data-science-2026/docs/retrieval.md)**

## Development Configuration Defaults

- **`BM25_TOP_K`**: 100 chunks
- **`DENSE_TOP_K`**: 100 chunks
- **`RRF_K`**: 60
- **`CANDIDATE_DOCS`**: 20 candidate documents
- **`FINAL_TOP_K`**: 5 document IDs (Task 1 constraint)

*Note: These are development defaults for pipeline verification, not tuned hyperparameters.*

## Expected Input Artifacts

```text
embeddings/metadata.jsonl   # 351,394 metadata items
vectorstore/corpus.index    # 351,394 vectors (768-dim FAISS IndexFlatIP)
```

The FAISS index and metadata.jsonl maintain a strict 1-to-1 index invariant.

## Test Commands

```bash
PYTHONPATH=. python tests/test_retrieval_bm25.py
PYTHONPATH=. python tests/test_retrieval_dense.py
PYTHONPATH=. python tests/test_retrieval_rrf.py
PYTHONPATH=. python tests/test_retrieval_reranker.py
PYTHONPATH=. python tests/test_retrieval_pipeline.py
```
