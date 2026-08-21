"""
Phân tích lỗi định tính: lọc ra các câu hỏi bị MISS HOÀN TOÀN (recall=0 - không tài liệu đúng nào
lọt vào top-k dự đoán) và in ra đầy đủ ngữ cảnh (câu hỏi + tài liệu ĐÚNG bị miss + các tài liệu SAI
đã dự đoán) để đọc và phân loại nguyên nhân bằng mắt thường.

Mục đích: quyết định hướng tối ưu tiếp theo cho Task 1 (đọc lỗi TRƯỚC khi vặn hyperparameter hay đổi
embedding, để không đoán mò) - xem claude/submission-log-task1.md mục quyết định 2026-08-14.

Mặc định chạy trên đúng bộ dữ liệu đã có sẵn (không cần chạy lại gì): dự đoán từ
`outputs/predictions/stage_eval_dong4.hybrid_fused.json` (cấu hình KHÔNG rerank hiện đang dùng, đo
trên val_split_20 - không leak data) so với đáp án `data/processed/val_split_20.json`.

Cách dùng:
    python -m src.evaluation.error_analysis \\
        --ground-truth data/processed/val_split_20.json \\
        --predictions outputs/predictions/stage_eval_dong4.hybrid_fused.json \\
        --questions data/processed/val_clean.jsonl \\
        --corpus data/processed/corpus.jsonl \\
        --limit 20 \\
        --output outputs/predictions/error_analysis_recall0.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_ground_truth(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    gt = {}
    for qid, item in raw.items():
        answer = item.get("answer") or []
        gt[qid] = [str(d) for d in answer]
    return gt


def load_predictions(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    preds = {}
    for qid, item in raw.items():
        answer = item.get("answer") if isinstance(item, dict) else None
        preds[qid] = [str(d) for d in (answer or [])]
    return preds


def load_question_text(path: str) -> dict:
    """Chấp nhận .jsonl (1 dòng/câu, có question_id+question) hoặc .json dict {qid: {"question":...}}."""
    texts = {}
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                texts[str(d["question_id"])] = d["question"]
    else:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for qid, item in raw.items():
            texts[str(qid)] = item.get("question", "")
    return texts


def load_corpus(path: str) -> dict:
    """doc_id -> {"passage": str, "link": str, "name": str|None}."""
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            corpus[str(d["doc_id"])] = {
                "passage": d.get("passage", ""),
                "link": d.get("link", ""),
                "name": d.get("name"),
            }
    return corpus


def snippet(text: str, max_len: int) -> str:
    text = " ".join(text.split())  # gộp khoảng trắng/xuống dòng thừa cho dễ đọc
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    parser.add_argument("--predictions", default="outputs/predictions/stage_eval_dong4.hybrid_fused.json",
                         help="Mặc định = dự đoán cấu hình KHÔNG rerank hiện đang dùng (dòng #9), đo trên val_split_20")
    parser.add_argument("--questions", default="data/processed/val_clean.jsonl")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--limit", type=int, default=20, help="Số câu recall=0 tối đa để in ra")
    parser.add_argument("--snippet-len", type=int, default=400, help="Số ký tự hiển thị mỗi tài liệu")
    parser.add_argument("--output", default="outputs/predictions/error_analysis_recall0.md")
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)
    preds = load_predictions(args.predictions)
    question_text = load_question_text(args.questions)
    corpus = load_corpus(args.corpus)

    recall0_cases = []
    for qid, correct_ids in gt.items():
        if not correct_ids:
            continue
        predicted_ids = preds.get(qid, [])
        hit = set(correct_ids) & set(predicted_ids)
        if len(hit) == 0:
            recall0_cases.append((qid, correct_ids, predicted_ids))

    total_with_gt = sum(1 for v in gt.values() if v)
    print(f"Tổng số câu có ground truth: {total_with_gt}")
    print(f"Số câu recall=0 (miss hoàn toàn): {len(recall0_cases)} ({len(recall0_cases)/total_with_gt*100:.1f}%)")

    # Ưu tiên hiển thị câu có nhiều tài liệu đúng nhất trước (n_relevant cao - theo log trước đây là
    # nhóm khó nhất, đáng đọc trước) - rồi mới tới các câu n_relevant=1 phổ biến nhất.
    recall0_cases.sort(key=lambda x: -len(x[1]))
    selected = recall0_cases[: args.limit]

    lines = []
    lines.append(f"# Phân tích lỗi recall=0 — {len(selected)}/{len(recall0_cases)} câu (tổng {len(recall0_cases)} câu miss hoàn toàn / {total_with_gt} câu có đáp án)")
    lines.append("")
    lines.append("Nguồn: dự đoán = `" + args.predictions + "`, đáp án = `" + args.ground_truth + "`.")
    lines.append("")
    lines.append("Đọc từng câu, phân loại nguyên nhân vào 1 trong các nhóm gợi ý sau (ghi chú ngay cạnh câu khi đọc):")
    lines.append("- **[CHUNK]** tài liệu đúng có nội dung liên quan nhưng bị cắt/chia chunk kém (câu trả lời nằm vắt qua 2 chunk)")
    lines.append("- **[EMBED]** tài liệu đúng không liên quan rõ về mặt từ vựng/ngữ nghĩa với câu hỏi theo cách embedding hiện tại nhận ra — có thể cần embedding tốt hơn")
    lines.append("- **[AMBIG]** câu hỏi mơ hồ / có thể có nhiều tài liệu cùng trả lời đúng nhưng ground truth chỉ ghi 1")
    lines.append("- **[GT]** nghi ngờ ground truth sai/thiếu (tài liệu dự đoán ra thực ra cũng đúng hoặc đúng hơn)")
    lines.append("- **[OTHER]** khác — ghi rõ")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, (qid, correct_ids, predicted_ids) in enumerate(selected, 1):
        q_text = question_text.get(qid, "(không tìm thấy text câu hỏi)")
        lines.append(f"## {i}. Câu hỏi `{qid}` — {len(correct_ids)} tài liệu đúng, miss cả {len(correct_ids)}")
        lines.append("")
        lines.append(f"**Câu hỏi:** {q_text}")
        lines.append("")
        lines.append(f"**Nguyên nhân (điền tay):** ")
        lines.append("")
        lines.append("**Tài liệu ĐÚNG (bị miss):**")
        for doc_id in correct_ids:
            doc = corpus.get(doc_id)
            if doc is None:
                lines.append(f"- `{doc_id}` — ⚠️ KHÔNG TÌM THẤY trong corpus (kiểm tra lại doc_id)")
                continue
            lines.append(f"- `{doc_id}` ({doc['link']})")
            lines.append(f"  > {snippet(doc['passage'], args.snippet_len)}")
        lines.append("")
        lines.append(f"**Top-{len(predicted_ids)} tài liệu ĐÃ DỰ ĐOÁN (sai):**")
        if not predicted_ids:
            lines.append("- (không có dự đoán nào)")
        for doc_id in predicted_ids:
            doc = corpus.get(doc_id)
            if doc is None:
                lines.append(f"- `{doc_id}` — ⚠️ KHÔNG TÌM THẤY trong corpus")
                continue
            lines.append(f"- `{doc_id}` ({doc['link']})")
            lines.append(f"  > {snippet(doc['passage'], args.snippet_len)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Đã lưu báo cáo {len(selected)} câu tại: {args.output}")


if __name__ == "__main__":
    main()
