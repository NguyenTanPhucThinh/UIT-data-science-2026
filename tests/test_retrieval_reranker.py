from src.retrieval.reranker import Reranker
from src.retrieval.types import DocumentCandidate


QUESTION = (
    "Việc tổ chức vận động, tiếp nhận, sử dụng nguồn đóng góp "
    "tự nguyện được thực hiện dựa trên nguyên tắc nào?"
)

METADATA_PATH = "embeddings/metadata.jsonl"

# 20 candidate documents từ output RRF test trước đó.
# Giữ nguyên ở mức development configuration.
CANDIDATES = [
    DocumentCandidate(
        doc_id="177504",
        chunk_index=0,
        score=0.030536,
        chunk_id="177504_chunk_005",
    ),
    DocumentCandidate(
        doc_id="287259",
        chunk_index=0,
        score=0.024494,
        chunk_id="287259_chunk_036",
    ),
    DocumentCandidate(
        doc_id="218452",
        chunk_index=0,
        score=0.024003,
        chunk_id="218452_chunk_029",
    ),
    DocumentCandidate(
        doc_id="40528",
        chunk_index=0,
        score=0.023969,
        chunk_id="40528_chunk_027",
    ),
    DocumentCandidate(
        doc_id="101100",
        chunk_index=0,
        score=0.023088,
        chunk_id="101100_chunk_027",
    ),
    DocumentCandidate(
        doc_id="119715",
        chunk_index=0,
        score=0.022903,
        chunk_id="119715_chunk_029",
    ),
    DocumentCandidate(
        doc_id="221292",
        chunk_index=0,
        score=0.022704,
        chunk_id="221292_chunk_027",
    ),
    DocumentCandidate(
        doc_id="169914",
        chunk_index=0,
        score=0.021508,
        chunk_id="169914_chunk_023",
    ),
    DocumentCandidate(
        doc_id="300434",
        chunk_index=0,
        score=0.021094,
        chunk_id="300434_chunk_032",
    ),
    DocumentCandidate(
        doc_id="34535",
        chunk_index=0,
        score=0.021012,
        chunk_id="34535_chunk_033",
    ),
    DocumentCandidate(
        doc_id="167439",
        chunk_index=0,
        score=0.020939,
        chunk_id="167439_chunk_036",
    ),
    DocumentCandidate(
        doc_id="19890",
        chunk_index=0,
        score=0.020698,
        chunk_id="19890_chunk_031",
    ),
    DocumentCandidate(
        doc_id="260505",
        chunk_index=0,
        score=0.020637,
        chunk_id="260505_chunk_021",
    ),
    DocumentCandidate(
        doc_id="206197",
        chunk_index=0,
        score=0.020557,
        chunk_id="206197_chunk_031",
    ),
    DocumentCandidate(
        doc_id="243124",
        chunk_index=0,
        score=0.020460,
        chunk_id="243124_chunk_013",
    ),
    DocumentCandidate(
        doc_id="179849",
        chunk_index=0,
        score=0.020417,
        chunk_id="179849_chunk_037",
    ),
    DocumentCandidate(
        doc_id="109743",
        chunk_index=0,
        score=0.020410,
        chunk_id="109743_chunk_003",
    ),
    DocumentCandidate(
        doc_id="174693",
        chunk_index=0,
        score=0.020254,
        chunk_id="174693_chunk_036",
    ),
    DocumentCandidate(
        doc_id="39897",
        chunk_index=0,
        score=0.020228,
        chunk_id="39897_chunk_027",
    ),
    DocumentCandidate(
        doc_id="126831",
        chunk_index=0,
        score=0.020050,
        chunk_id="126831_chunk_032",
    ),
]


def main():
    metadata = []

    print("=" * 80)
    print("M5 — RERANKER DEVELOPMENT TEST")
    print("=" * 80)

    print("\nLoading metadata...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(__import__("json").loads(line))

    print(f"Metadata records: {len(metadata):,}")

    # ---------------------------------------------------------
    # IMPORTANT:
    # chunk_index must point to the actual metadata row.
    #
    # The manually copied candidates above only contain chunk_id,
    # so resolve chunk_id -> real metadata index here.
    # ---------------------------------------------------------
    chunk_to_index = {
        item["chunk_id"]: idx
        for idx, item in enumerate(metadata)
    }

    resolved_candidates = []

    for candidate in CANDIDATES:
        chunk_index = chunk_to_index.get(candidate.chunk_id)

        if chunk_index is None:
            raise RuntimeError(
                f"Chunk not found in metadata: {candidate.chunk_id}"
            )

        resolved_candidates.append(
            DocumentCandidate(
                doc_id=candidate.doc_id,
                chunk_index=chunk_index,
                score=candidate.score,
                chunk_id=candidate.chunk_id,
            )
        )

    print(f"Resolved candidates: {len(resolved_candidates)}")

    # ---------------------------------------------------------
    # Load the required Vietnamese reranker
    # ---------------------------------------------------------
    reranker = Reranker(
        metadata=metadata,
        model_name="AITeamVN/Vietnamese_Reranker",
        device="auto",
    )

    print(f"Model: {reranker.model_name}")
    print(f"Device: {reranker.device}")

    # ---------------------------------------------------------
    # Rerank top-20 candidates -> top-5
    # ---------------------------------------------------------
    results = reranker.rerank(
        question=QUESTION,
        candidates=resolved_candidates,
        top_k=5,
    )

    print("\n" + "=" * 80)
    print("RERANKED TOP-5")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:>2}. "
            f"doc_id={result.doc_id} | "
            f"chunk_id={result.chunk_id} | "
            f"score={result.score:.6f}"
        )

    print("\nReranker test completed successfully.")


if __name__ == "__main__":
    main()
