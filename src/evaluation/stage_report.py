"""
Tổng hợp Recall/Precision từng GIAI ĐOẠN của pipeline hybrid (dense-only, BM25-only, sau hybrid
merge RRF - trước rerank, sau rerank+top-k) thành 1 bảng - để xem giai đoạn nào đang là điểm yếu,
thay vì chỉ có 1 con số cuối cùng.

Dùng SAU KHI đã chạy full_pipeline_eval.py với --stage-eval (sinh ra 4 file dự đoán: --output gốc
+ <output>.dense_only.json + <output>.bm25_only.json + <output>.hybrid_fused.json):

    python -m src.evaluation.full_pipeline_eval \\
        --embedding-model <path_hoặc_tên_model> \\
        --reranker-model <path_hoặc_tên_model> \\
        --chunks data/processed/chunks_corpus.jsonl \\
        --questions data/processed/val_preprocessed.jsonl \\
        --output outputs/stage_eval.json \\
        --embedding-cache outputs/cache_stage_eval.npz \\
        --stage-eval

    python -m src.evaluation.stage_report \\
        --ground-truth data/processed/val_split_20.json \\
        --base outputs/stage_eval.json

LƯU Ý: --ground-truth PHẢI là file có nhãn thật (val_split_20.json hoặc train.json), KHÔNG dùng
public-official.json (answer luôn null) - xem cùng lưu ý ở full_pipeline_eval.py/evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .evaluate import load_ground_truth, load_predictions
    from .metrics import evaluate_predictions
except ImportError:  # cho phép chạy trực tiếp
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evaluation.evaluate import load_ground_truth, load_predictions
    from src.evaluation.metrics import evaluate_predictions

# (stage_key, nhãn hiển thị) - thứ tự đúng theo luồng chạy thật của pipeline (xem full_pipeline_eval.py).
# stage_key="final" là trường hợp đặc biệt: dùng ĐÚNG file --base (không có hậu tố .final.json nào cả).
STAGES = [
    ("dense_only", "1. Dense-only (chỉ embedding, chưa BM25/hybrid/rerank)"),
    ("bm25_only", "2. BM25-only (chỉ sparse, chưa dense/hybrid/rerank)"),
    ("hybrid_fused", "3. Sau hybrid merge (RRF dense+BM25), TRƯỚC rerank"),
    ("final", "4. Sau rerank + top-k (KẾT QUẢ CUỐI - giống submission.json)"),
]


def stage_file_path(base: str, stage_key: str) -> str:
    if stage_key == "final":
        return base
    p = Path(base)
    return str(p.with_suffix(f".{stage_key}{p.suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", required=True, help="vd data/processed/val_split_20.json - PHẢI có nhãn thật, không dùng public-official.json")
    parser.add_argument("--base", required=True, help="Đúng đường dẫn --output đã dùng khi chạy full_pipeline_eval.py --stage-eval")
    parser.add_argument("--report", default=None, help="Nếu chỉ định, lưu báo cáo chi tiết từng giai đoạn (JSON) ra đường dẫn này")
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)

    print("=" * 78)
    print(f"{'Giai đoạn':<50s} {'Recall':>9s} {'Precision':>11s}")
    print("-" * 78)
    results = {}
    for stage_key, label in STAGES:
        path = stage_file_path(args.base, stage_key)
        if not Path(path).exists():
            print(f"{label:<50s} {'(thiếu file: ' + path + ')'}")
            continue
        predictions = load_predictions(path)
        result = evaluate_predictions(ground_truth, predictions)
        results[stage_key] = result
        print(f"{label:<50s} {result['avg_recall']:9.4f} {result['avg_precision']:11.4f}")
    print("=" * 78)

    if not results:
        print(
            "\nKhông tìm thấy file nào - kiểm tra lại --base có đúng đường dẫn --output đã dùng khi "
            "chạy full_pipeline_eval.py VỚI CỜ --stage-eval hay không (nếu chạy thiếu --stage-eval "
            "thì chỉ có file kết quả cuối, không có 3 file giai đoạn trung gian)."
        )
        return

    keys_present = [k for k, _ in STAGES if k in results]
    if len(keys_present) >= 2:
        print("\nChênh lệch Recall giữa các giai đoạn liên tiếp (giai đoạn nào tụt nhiều nhất = điểm yếu):")
        for prev_k, cur_k in zip(keys_present, keys_present[1:]):
            delta = results[cur_k]["avg_recall"] - results[prev_k]["avg_recall"]
            sign = "+" if delta >= 0 else ""
            flag = "  <-- GIẢM, đáng chú ý" if delta < -0.01 else ""
            print(f"  {prev_k:<14s} -> {cur_k:<14s}: {sign}{delta:.4f}{flag}")

        best_of_dense_bm25 = max(
            results.get("dense_only", {}).get("avg_recall", 0.0),
            results.get("bm25_only", {}).get("avg_recall", 0.0),
        )
        if "hybrid_fused" in results and "final" in results:
            print(
                "\nGợi ý đọc kết quả:\n"
                f"  - Nếu 'hybrid_fused' ({results['hybrid_fused']['avg_recall']:.4f}) đã thấp hơn "
                f"cả max(dense_only, bm25_only) ({best_of_dense_bm25:.4f}) -> bước RRF fusion đang "
                "làm mất thông tin (kiểm tra lại dense_top_n/bm25_top_n/rrf_k), không phải lỗi model.\n"
                f"  - Nếu 'final' ({results['final']['avg_recall']:.4f}) thấp hơn hẳn 'hybrid_fused' "
                "-> RERANKER đang làm giảm chất lượng (đẩy đúng document ra khỏi top-k), không phải "
                "retrieval yếu - nên xem lại reranker (model gốc / dữ liệu hard-negative train reranker).\n"
                "  - Nếu cả 2 điều trên đều KHÔNG xảy ra (mỗi giai đoạn đều >= giai đoạn trước, hoặc "
                "giảm rất nhẹ) -> pipeline đang hoạt động đúng như kỳ vọng, muốn tăng điểm cần cải "
                "thiện CHẤT LƯỢNG MODEL (fine-tune tốt hơn) chứ không phải sửa lỗi cấu trúc pipeline."
            )

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu báo cáo chi tiết: {args.report}")


if __name__ == "__main__":
    main()
