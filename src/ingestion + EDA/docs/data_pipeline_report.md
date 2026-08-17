# Data Pipeline & EDA Master Report
**UIT Data Science Challenge 2026 — Task 1 (LegalIR)**
**Author:** Doan Quoc Kien (Data & Utilities — Thanh vien 1)
**Last Updated:** 2026-08-18

---

## 1. Executive Summary & Overview
This report documents the end-to-end Data & Utilities pipeline (Phases 1-4) and baseline validation (Phase 5). All 8,532 raw context files and 7,000 training queries have been audited, cleaned, tokenized, and structurally chunked into the production datasets in `data/processed/`.

---

## 2. Phase 1: Data Audit & Integrity Findings
- **Total Context Files**: 8,532 parsed successfully with 0 duplicate document IDs.
- **Answer Coverage**: All 3,105 unique document IDs referenced in `train.json` (100%) exist in our corpus.
- **Empty Document Audit & Recovery**:
  - 20 documents in the raw corpus had empty passages (`""`).
  - **Critical Finding**: **6 of these 20 empty documents** are target answers for **11 questions** in `train.json` — confirmed paywall issue on `thuvienphapluat.vn` at crawl time.
  - **Mitigation**: Constructed fallback passages from URL slugs so they can be retrieved by downstream models.

---

## 2.5. Phase 1.5: Corpus Deduplication
- **Method**: Extracted the unique thuvienphapluat.vn page ID from each document's `link` field (trailing numeric ID before `.aspx`). Documents sharing the same page ID are the same legal text crawled under different URL slugs.
- **Result**: **27 duplicate groups / 54 documents** identified and confirmed (matching teammate Hoang's independent analysis via `bao-cao-data-quality.md`).
- **Canonical Selection**: For each group, the document with the **longer passage** is kept as canonical (higher-quality crawl). The shorter alias is remapped.
- **Output**: `data/processed/dedup_mapping.json` — 27 alias->canonical mappings for use by the retrieval layer.
- **Compliance**: Does NOT modify corpus or ground truth labels. Only a retrieval-layer design decision, permitted under competition rules (muc 7).

---

## 3. Phase 2: Text Preprocessing & Segmentation
- **Unicode Normalization**: Applied NFC standard normalization across 100% of corpus text and queries.
- **Vietnamese Word Segmentation**: Implemented `VnCoreNLP` (running in parallel `ProcessPoolExecutor` with 8 workers) to compound multi-word legal terms (e.g. `bao_hiem_y_te`).
- **Character Compression**: Raw corpus size reduced from ~353.7M characters to ~345.0M clean characters without losing structural context.

---

## 4. Phase 3: Exploratory Data Analysis (EDA)

### 4.1 - Document Type Taxonomy & Target Rates
| Document Type | Total Count | % of Corpus | Target Docs in Train | Target Rate |
|---|---|---|---|---|
| **Nghi dinh** | 4,700 | 55.1% | 1,814 | **38.6%** |
| **Quyet dinh** | 1,234 | 14.5% | 372 | **30.1%** |
| **Thong tu** | 1,005 | 11.8% | 370 | **36.8%** |
| **Luat** | 560 | 6.6% | 291 | **52.0%** |
| **Khac** | 471 | 5.5% | 149 | **31.6%** |
| **Cong van** | 436 | 5.1% | 72 | **16.5%** |
| **Nghi quyet** | 79 | 0.9% | 26 | **32.9%** |
| **Chi thi** | 47 | 0.6% | 11 | **23.4%** |

### 4.2 - Passage Length Distribution
Target answer documents are on average ~36% longer (median 33,702 chars) than unreferenced documents (median 18,629 chars), indicating that comprehensive legal decrees carry denser domain knowledge.

### 4.3 - Question Multiplicity
- **Single-answer queries**: 92.1% (6,447 questions).
- **Multi-answer queries**: 7.9% (553 questions requiring 2 to 5 document synthesis).

---

## 5. Phase 4: Adaptive Structural Chunking
- **Structural Boundary Splitting**: Primary splits at `Dieu` (Article), `Chuong` (Chapter), and `Muc` regex markers to retain complete legal clauses.
- **Sliding Window Fallback**: Large single-line sections are recursively split by word boundaries to enforce a strict **max 400-word limit** (~512 tokens).
- **Output**: 351,394 clean chunks saved to `chunks_corpus.jsonl` with 0 empty passages.

---

## 6. Phase 5: Baseline Pipeline Proof (BM25)
- **Model**: BM25 Sparse Retrieval using `VnCoreNLP` tokenized queries and passages.
- **Aggregation**: Max-Score chunk aggregation mapped back to `doc_id`.

| Configuration | Recall@5 | Precision@5 | MRR@5 |
|---|---|---|---|
| **Baseline (no dedup)** | TBD after re-run | TBD | TBD |
| **+ Dedup remapping** | TBD | TBD | TBD |

*Note: Dedup remapping applies `dedup_mapping.json` to remap alias doc_ids -> canonical before scoring. Results to be updated.*
