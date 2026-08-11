# UIT Data Science Challenge 2026 - RAG Legal Q&A

This repository contains the architecture and implementation for the UIT Data Science 2026 competition. The system is a Retrieval-Augmented Generation (RAG) pipeline designed specifically for Vietnamese legal documents.

## Repository Structure

- `src/ingestion + EDA/`: Scripts for data loading, analysis, and preprocessing.
- `src/evaluation/`: Scripts for testing retrieval baselines (e.g., BM25).
- `src/utils/`: Shared utilities like standard loggers, file readers, and profiling timers.
- `src/config/`: Shared configuration logic and path resolutions.
- `docs/`: Master planning documents, task distribution, and data handoff guides.
- `data/processed/`: The final cleaned, segmented, and chunked legal text ready for embedding.

## Data & Utilities Preprocessing Pipeline (Phase 1-4)
The foundation of the repository (Data Processing) has been completed, resulting in the pristine dataset output found in `data/processed/`.

### Key Features
1. **High-Accuracy Segmentation**: Text is heavily segmented using `VnCoreNLP` initialized in parallel JVM execution pools to correctly compound complex legal jargon (e.g., `bảo_hiểm y_tế`).
2. **Adaptive Structural Chunking**: 8,500 legal documents are semantically chunked according to their legal architecture (`Chương`, `Mục`, `Điều`) rather than arbitrary windows. 
3. **Safe Context Bounds**: Chunks strictly adhere to a 400-word limit, seamlessly supporting standard Vietnamese legal embedding models (512 token limit).
4. **Strong Baseline**: Even on a purely sparse BM25 baseline without dense vectors, the dataset achieves an **80.0% Recall@5** on the validation suite.

## Integration Guide
For the `Core RAG`, `Embeddings`, and `Vectorstore` team members, please consult:
1. `docs/data_handoff_guide.md` for JSON schemas, aggregation strategies, and integration guidelines.
2. `src/utils/metrics.py` for standard metrics execution (Recall, MRR, NDCG).
3. `src/utils/timer.py` and `logger.py` for profiling and system logs.

*Maintainer for Data & Utilities: Đoàn Quốc Kiên (Thành viên 1)*
