from src.retrieval.pipeline import RetrievalPipeline

QUESTION = (
    "Việc tổ chức vận động, tiếp nhận, sử dụng nguồn đóng góp "
    "tự nguyện được thực hiện dựa trên nguyên tắc nào?"
)


def main():
    print("=" * 80)
    print("END-TO-END RETRIEVAL PIPELINE DEVELOPMENT TEST")
    print("=" * 80)

    pipeline = RetrievalPipeline(device="auto")

    print(f"Dense device   : {pipeline.dense.device}")
    print(f"Reranker device: {pipeline._resolve_device('auto')}")

    doc_ids = pipeline.retrieve(QUESTION)

    print(f"Reranker loaded: {pipeline.reranker.model_name}")

    print("\n" + "=" * 80)
    print("FINAL TOP-5 DOCUMENT IDs")
    print("=" * 80)

    for rank, doc_id in enumerate(doc_ids, start=1):
        print(f"{rank:>2}. doc_id={doc_id}")

    assert len(doc_ids) <= 5, "Task 1 allows at most 5 document IDs"
    assert doc_ids[0] == "177504", f"Expected 177504 at rank 1, got {doc_ids[0]}"

    print("\nEnd-to-end pipeline test completed successfully.")


if __name__ == "__main__":
    main()
