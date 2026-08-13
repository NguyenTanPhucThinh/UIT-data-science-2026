from src.retrieval.dense import DenseRetriever


QUESTION = (
    "Việc tổ chức vận động, tiếp nhận, sử dụng nguồn đóng góp "
    "tự nguyện được thực hiện dựa trên nguyên tắc nào?"
)

dense = DenseRetriever(
    index_path="vectorstore/corpus.index",
    metadata_path="embeddings/metadata.jsonl",
)

results = dense.retrieve(
    QUESTION,
    top_k=100,
)

print("=" * 80)
print("DENSE DEVELOPMENT TEST")
print("=" * 80)

print(f"Returned: {len(results)} chunks\n")

for rank, result in enumerate(results[:10], start=1):
    print(
        f"{rank:>2}. "
        f"doc_id={result.doc_id} | "
        f"chunk_id={result.chunk_id} | "
        f"score={result.score:.6f}"
    )