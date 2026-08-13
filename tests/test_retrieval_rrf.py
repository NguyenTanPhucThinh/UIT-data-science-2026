from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import (
    rrf_fuse,
    aggregate_chunks_to_documents,
)

QUESTION = (
    "Việc tổ chức vận động, tiếp nhận, sử dụng nguồn đóng góp "
    "tự nguyện được thực hiện dựa trên nguyên tắc nào?"
)

metadata_path = "embeddings/metadata.jsonl"

bm25 = BM25Retriever(metadata_path)
dense = DenseRetriever(
    index_path="vectorstore/corpus.index",
    metadata_path=metadata_path,
)

bm25_results = bm25.retrieve(QUESTION, top_k=100)
dense_results = dense.retrieve(QUESTION, top_k=100)

fused_scores = rrf_fuse(
    dense_results,
    bm25_results,
    rrf_k=60,
)

candidates = aggregate_chunks_to_documents(
    fused_scores,
    dense.metadata,
    top_k=20,
)

print("=" * 80)
print("RRF + DOCUMENT AGGREGATION DEVELOPMENT TEST")
print("=" * 80)

print(f"BM25 chunks : {len(bm25_results)}")
print(f"Dense chunks: {len(dense_results)}")
print(f"Fused chunks: {len(fused_scores)}")
print(f"Candidates  : {len(candidates)}")
print()

for rank, candidate in enumerate(candidates, start=1):
    print(
        f"{rank:>2}. "
        f"doc_id={candidate.doc_id} | "
        f"chunk_id={candidate.chunk_id} | "
        f"score={candidate.score:.6f}"
    )