"""
Trich ra day du ngu canh (cau hoi + tai lieu DUNG bi mieu + tai lieu SAI da du doan) cho cac cau hoi
co n_relevant>=2 (can >=2 tai lieu lien quan) MA BI MIEU IT NHAT 1 tai lieu dung - day la nhom yeu
nhat xuyen suot MOI cau hinh da thu (Recall chi 0.70-0.76 o n=2, 0.27-0.50 o n=3, xem
submission-log-task1.md). Khac voi src/evaluation/error_analysis.py (chi loc cau MIEU HOAN TOAN,
recall=0) - script nay lay CA cau MIEU MOT PHAN (vd dung 2/3 tai lieu, van con 1 bi mieu) vi day la
phan lon cac case o nhom n_relevant>=2 (49/97 cau n=2 la mieu hoan toan hay mot phan can kiem tra).

Da thu tune 3 huong khac (fusion 3 chieu voi gte-Qwen2, tune dense-top-n/bm25-top-n/rrf-k) deu KHONG
co tin hieu cai thien nhom nay - buoc tiep theo la DOC TRUC TIEP cac case that de tim nguyen nhan goc
(chunking cat roi noi dung, BM25 tokenize kem thuat ngu phap ly, tai lieu lien quan qua khac nhau ve
ngu nghia...) thay vi tiep tuc do tham so.

Cach dung (tu thu muc goc UIT-data-science-2026):
    python scripts/analyze_multi_relevant_misses.py

Mac dinh tu dong tim file predictions THAT (uu tien outputs/predictions/sweep_bm25_weight/bm25_weight_0.1.json
- dung bm25-weight=0.1 CHINH THUC dang dung - neu khong co thi fallback sang
outputs/predictions/aiteamvn_rerank_test.hybrid_fused.json, bm25-weight=0.4, gan giong, in canh bao
ro rang neu phai dung fallback).

Ket qua luu ra file markdown de doc + dien tay nguyen nhan, giong quy uoc da dung o
error_analysis_recall0.md truoc do.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_ground_truth(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(qid): [str(d) for d in (item.get("answer") or [])] for qid, item in raw.items()}


def load_predictions(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    preds = {}
    for qid, item in raw.items():
        answer = item.get("answer") if isinstance(item, dict) else None
        preds[str(qid)] = [str(d) for d in (answer or [])]
    return preds


def load_question_text(path: str) -> dict:
    texts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            texts[str(d["question_id"])] = d["question"]
    return texts


def load_corpus_by_doc(path: str) -> dict:
    """chunks_corpus.jsonl la CAP CHUNK (nhieu chunk / 1 doc_id) - gom lai thanh doc_id -> [passage,...]."""
    by_doc: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            by_doc.setdefault(str(d["doc_id"]), []).append(d.get("passage", ""))
    return by_doc


def snippet(text: str, max_len: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def find_default_predictions() -> tuple[str, bool]:
    """Uu tien file predictions THAT dung bm25-weight=0.1 (chinh thuc dang dung), neu khong co thi
    fallback sang file bm25=0.4 da co san tu lan test reranker (gan giong, chenh Recall LOCAL chi
    +0.0036 nen van dung tam duoc de doc loi dinh tinh). Tra ve (path, is_fallback)."""
    official = Path("outputs/predictions/sweep_bm25_weight/bm25_weight_0.1.json")
    if official.exists():
        return str(official), False
    fallback = Path("outputs/predictions/aiteamvn_rerank_test.hybrid_fused.json")
    return str(fallback), True


def main() -> None:
    default_pred, _ = find_default_predictions()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    parser.add_argument("--predictions", default=default_pred)
    parser.add_argument("--questions", default="data/processed/val_preprocessed.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--min-n-relevant", type=int, default=2, help="Chi lay cau hoi co >= N tai lieu dung")
    parser.add_argument("--snippet-len", type=int, default=500)
    parser.add_argument("--max-chunks-per-doc", type=int, default=2, help="So chunk toi da hien thi cho moi doc_id (doc co the co nhieu chunk)")
    parser.add_argument("--output", default="outputs/predictions/error_analysis_multi_relevant.md")
    args = parser.parse_args()

    _, is_fallback = find_default_predictions()
    if args.predictions == default_pred and is_fallback:
        print(
            "[CANH BAO] KHONG tim thay file predictions chinh thuc bm25-weight=0.1 tai "
            "outputs/predictions/sweep_bm25_weight/bm25_weight_0.1.json - dang dung fallback "
            f"'{args.predictions}' (bm25-weight=0.4, Recall LOCAL 0.9267, chenh +0.0036 so voi "
            "0.1 dang dung chinh thuc - van dung tam duoc de doc loi dinh tinh, khong anh huong "
            "nhieu vi 2 cau hinh rat gan nhau)."
        )

    gt = load_ground_truth(args.ground_truth)
    preds = load_predictions(args.predictions)
    question_text = load_question_text(args.questions)
    corpus = load_corpus_by_doc(args.chunks)

    cases = []
    for qid, correct_ids in gt.items():
        if len(correct_ids) < args.min_n_relevant:
            continue
        predicted_ids = preds.get(qid, [])
        missed = [d for d in correct_ids if d not in predicted_ids]
        if not missed:
            continue  # cau nay khong yeu - bo qua
        cases.append((qid, correct_ids, predicted_ids, missed))

    total_multi = sum(1 for v in gt.values() if len(v) >= args.min_n_relevant)
    print(f"Tong so cau hoi co n_relevant >= {args.min_n_relevant}: {total_multi}")
    print(f"So cau bi mieu it nhat 1 tai lieu dung: {len(cases)} ({len(cases)/total_multi*100:.1f}%)")

    # Uu tien hien thi cau mieu NHIEU nhat truoc (mieu hoan toan > mieu 1 phan), roi den n_relevant cao hon.
    cases.sort(key=lambda x: (-len(x[3]), -len(x[1])))

    lines = []
    lines.append(f"# Phân tích lỗi câu hỏi nhiều tài liệu liên quan (n_relevant≥{args.min_n_relevant}) bị miss")
    lines.append("")
    lines.append(f"Nguồn: dự đoán = `{args.predictions}`" + (" (⚠️ FALLBACK bm25=0.4, không phải bm25=0.1 chính thức)" if is_fallback else " (bm25=0.1, chính thức)") + f", đáp án = `{args.ground_truth}`.")
    lines.append("")
    lines.append(f"**{len(cases)}/{total_multi} câu hỏi (n_relevant≥{args.min_n_relevant}) bị miss ít nhất 1 tài liệu đúng.**")
    lines.append("")
    lines.append("Đọc từng câu, phân loại nguyên nhân (ghi ngay cạnh câu khi đọc):")
    lines.append("- **[CHUNK]** tài liệu đúng bị chia chunk kém — nội dung liên quan nằm ở chunk KHÁC (không phải chunk đại diện được xếp hạng)")
    lines.append("- **[EMBED]** tài liệu đúng không liên quan rõ về từ vựng/ngữ nghĩa với câu hỏi theo cách embedding hiện tại nhận ra")
    lines.append("- **[BM25]** BM25 không khớp được thuật ngữ pháp lý viết khác nhau (đồng nghĩa, viết tắt, số hiệu điều luật)")
    lines.append("- **[TOPK]** câu hỏi cần >5 tài liệu liên quan hoặc top-5 bị các tài liệu SAI khác chiếm chỗ dù đúng đã được xếp hạng gần đúng")
    lines.append("- **[GT]** nghi ngờ ground truth sai/thiếu")
    lines.append("- **[OTHER]** khác — ghi rõ")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, (qid, correct_ids, predicted_ids, missed) in enumerate(cases, 1):
        q_text = question_text.get(qid, "(không tìm thấy text câu hỏi)")
        n_hit = len(correct_ids) - len(missed)
        lines.append(f"## {i}. Câu hỏi `{qid}` — {len(correct_ids)} tài liệu đúng, đúng {n_hit}/{len(correct_ids)}, miss {len(missed)}")
        lines.append("")
        lines.append(f"**Câu hỏi:** {q_text}")
        lines.append("")
        lines.append(f"**Nguyên nhân (điền tay):** ")
        lines.append("")
        lines.append("**Tài liệu ĐÚNG:**")
        for doc_id in correct_ids:
            status = "❌ BỊ MISS" if doc_id in missed else "✅ đã tìm thấy"
            doc_chunks = corpus.get(doc_id)
            lines.append(f"- `{doc_id}` — {status}")
            if doc_chunks is None:
                lines.append("  > ⚠️ KHÔNG TÌM THẤY trong corpus (kiểm tra lại doc_id)")
                continue
            lines.append(f"  ({len(doc_chunks)} chunk trong corpus)")
            for chunk_text in doc_chunks[: args.max_chunks_per_doc]:
                lines.append(f"  > {snippet(chunk_text, args.snippet_len)}")
        lines.append("")
        lines.append(f"**Top-{len(predicted_ids)} tài liệu ĐÃ DỰ ĐOÁN:**")
        if not predicted_ids:
            lines.append("- (không có dự đoán nào)")
        for doc_id in predicted_ids:
            status = "✅ đúng" if doc_id in correct_ids else "❌ sai"
            doc_chunks = corpus.get(doc_id)
            lines.append(f"- `{doc_id}` — {status}")
            if doc_chunks is None:
                lines.append("  > ⚠️ KHÔNG TÌM THẤY trong corpus")
                continue
            for chunk_text in doc_chunks[: args.max_chunks_per_doc]:
                lines.append(f"  > {snippet(chunk_text, args.snippet_len)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Đã lưu {len(cases)} case tại: {args.output}")
    print("Mở file này để đọc trực tiếp và điền nguyên nhân — hoặc gửi lại nội dung file cho Claude để đọc giúp.")


if __name__ == "__main__":
    main()
