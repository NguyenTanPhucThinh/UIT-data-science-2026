"""
Boost/injection dua tren TEN THUC THE (co quan/to chuc) trich xuat tu cau hoi - sua nhom loi [TOPK]
"nham lan van ban khuon mau" da thay trong 53 case da doc thu cong (submission-log-task1.md, muc
2026-08-16 toi). Vi du that:
  - Case 11 (`28816`): "Kiểm soát viên Tổng công ty Giấy Việt Nam..." - top-5 toan la Tổng công ty
    Thủy sản/Sông Đà/Lương thực/Lâm nghiệp - CUNG khuon mau tieu de, SAI ten cong ty.
  - Case 34 (`20456`): "...ngành Hải quan..." - top-5 toan la Cục Hàng không/BVTV/BHXH/TAND - CUNG
    "Quy chế/Nội quy tiếp công dân", SAI nganh.

Co che KHAC citation-graph (dua tren trich dan van ban): CHI kich hoat khi (a) cau hoi CO the phat
hien duoc 1 cum ten thuc the ro rang (theo mau "<tien to co quan> <Ten Rieng>"), VA (b) KHONG van ban
nao trong top-5 hien co CHUA cum tu do trong TIEU DE (chunk dau tien) cua no - dieu kien chat hon
nhieu so voi "chi can co edge trong graph" (ly do citation-graph bi bom qua nhieu/sai o lan truoc).
Khi kich hoat, tim trong TOAN BO corpus (chi quet TIEU DE - chunk dau - de tranh nham voi cac lan
nhac ten trong menh de "Can cu..." o chunk sau) xem co van ban nao KHAC chua dung cum tu do khong -
day la string match THUAN TUY, khong can GPU/embedding, rat re.

Cach dung (tu thu muc goc UIT-data-science-2026):
    python scripts/apply_entity_boost_injection.py

In luon thong ke chan doan (bao nhieu cau hoi phat hien duoc thuc the, bao nhieu "mat tich" khoi
top-5, bao nhieu tim lai duoc trong corpus) VA ghi file predictions moi de do Recall NGAY trong 1 lan
chay (rut kinh nghiem tu citation-graph - lan do phai chay 2 script rieng, nhieu vong qua lai).

Sau do do lai bang evaluate.py nhu moi lan khac:
    python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/entity_boost_injected.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Sap xep DAI truoc NGAN (vd "Tổng công ty" phai duoc thu truoc "Công ty", "Ủy ban nhân dân" truoc
# "Ủy ban") de khong bi khop nham tien to con NGAN hon nam trong 1 tien to DAI hon.
ORG_PREFIXES = sorted(
    [
        "Tổng công ty", "Tổng cục", "Tập đoàn", "Ngân hàng", "Học viện",
        "Ủy ban nhân dân", "Ủy ban", "ngành", "Sở Giao dịch", "Sở",
        "Viện kiểm sát nhân dân", "Viện kiểm sát", "Tòa án nhân dân",
        "Kho bạc Nhà nước", "Chi cục", "Cục", "Viện", "Trường", "Bộ", "Công ty",
    ],
    key=len, reverse=True,
)

# v2 (sau khi chay thu: ten co quan tieng Viet THUONG co tu noi VIET THUONG nam GIUA cum (vd "Y **tế**",
# "Giao dich **chung khoan**", "Dau **khi**", "Ke hoach **va** Dau tu") - yeu cau MOI tu deu viet hoa
# lien tiep (v1) khien cum bi CAT CUT thanh manh vo nghia (vd "Sở Giao" thay vi "Sở Giao dịch Chứng
# khoán") -> tim kiem theo cum cut nay khop BUA vao nhieu van ban khong lien quan, gay hai (Recall
# giam nhe 0.9303->0.9295 o lan chay dau). Sua: CHI tu DAU TIEN sau tien to bat buoc viet hoa (xac
# nhan day la khoi dau 1 danh tu rieng, tranh khop nham cum khong phai ten rieng nhu "Bo truong X" ->
# "truong" thuong, bi loai tu dau), CAC TU SAU do duoc phep viet thuong, CHI dung lai khi gap 1 trong
# STOP_WORDS (tu bao hieu het cum danh tu, bat dau menh de vi ngu cua cau hoi) hoac dau cau hoac dat
# toi do dai toi da.
STOP_WORDS = {
    "phải", "có", "là", "được", "cần", "sẽ", "đã", "đang", "không",
    "nếu", "khi", "muốn", "nên", "để", "mà", "thì", "nhằm", "hay",
    "gì", "như", "sao", "này", "đó", "ai", "bao", "nhiêu", "vậy",
}
MAX_ENTITY_WORDS = 8  # gioi han tren (bao gom ca tien to) de tranh "an" ca phan con lai cua cau


def is_cap_word(w: str) -> bool:
    return len(w) > 0 and w[0].isupper()


_WORD_PUNCT_RE = re.compile(r"^([^\s,.;:?!]+)([,.;:?!]?)")


def extract_entities(question: str) -> list:
    """Tra ve danh sach cum tu thuc the DAY DU (vd 'Tổng công ty Giấy Việt Nam', 'ngành Hải quan',
    'Sở Giao dịch Chứng khoán Hà Nội'), theo mau '<tien to co quan> <danh tu rieng, tu dau viet hoa,
    cac tu sau co the viet thuong (tu noi) cho toi khi gap STOP_WORDS/dau cau/qua dai>'."""
    entities = []
    for prefix in ORG_PREFIXES:
        idx = 0
        while True:
            pos = question.find(prefix, idx)
            if pos == -1:
                break
            after = question[pos + len(prefix):]
            if not after or after[0] not in " \t":
                # tien to phai duoc theo sau boi 1 khoang trang (ranh gioi tu that su).
                idx = pos + len(prefix)
                continue
            raw_words = after.lstrip().split(" ")
            name_words = []
            for i, w in enumerate(raw_words):
                m = _WORD_PUNCT_RE.match(w)
                if not m:
                    break
                w_core, has_punct = m.group(1), m.group(2)
                if i == 0 and not is_cap_word(w_core):
                    break  # tu DAU TIEN sau tien to phai viet hoa - xac nhan bat dau danh tu rieng
                if w_core.lower() in STOP_WORDS:
                    break
                name_words.append(w_core)
                if has_punct or len(name_words) + 1 >= MAX_ENTITY_WORDS:  # +1 tinh ca tien to
                    break
            if name_words:
                full = " ".join([prefix] + name_words)
                entities.append(" ".join(full.split()))
            idx = pos + len(prefix)
    # loai cum bi TRUNG LAP hoac la CON CUA 1 cum khac dai hon da tim duoc (vd "Bộ Tài chính" bi
    # chua trong "Tổng cục Hải quan Bộ Tài chính" - giu ban DAI nhat, du thua)
    entities = sorted(set(entities), key=len, reverse=True)
    deduped = []
    for e in entities:
        if not any(e != other and e in other for other in deduped):
            deduped.append(e)
    return deduped


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def load_titles(path: str) -> dict:
    """doc_id -> text CHUNK DAU TIEN (tieu de/preamble ngan). CHI lay chunk dau (khac PREAMBLE_CHUNKS=3
    cua build_citation_graph.py) - ten thuc the trong TIEU DE that moi la tin hieu manh, tranh nham
    voi cac lan nhac ten trong menh de "Can cu..." o chunk sau (vd van ban nao cung "Can cu Luat To
    chuc Chinh phu" nhung khong co nghia van ban do NOI VE Chinh phu)."""
    titles = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc_id = str(d["doc_id"])
            if doc_id not in titles:
                titles[doc_id] = d.get("passage", "")
    return titles


def load_ground_truth_questions(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(qid): item.get("question", "") for qid, item in raw.items()}


def load_predictions(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    preds = {}
    for qid, item in raw.items():
        answer = item.get("answer") if isinstance(item, dict) else None
        preds[str(qid)] = [str(d) for d in (answer or [])]
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", default="data/processed/val_split_20.json")
    parser.add_argument("--predictions", default="outputs/predictions/sweep_bm25_weight/bm25_weight_0.1.json")
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--output", default="outputs/predictions/entity_boost_injected.json")
    parser.add_argument("--max-inject", type=int, default=1)
    args = parser.parse_args()

    question_texts = load_ground_truth_questions(args.ground_truth)
    preds = load_predictions(args.predictions)
    print(f"Doc tieu de (chunk dau) tu {args.chunks}...")
    titles = load_titles(args.chunks)
    titles_norm = {doc_id: normalize(t) for doc_id, t in titles.items()}
    print(f"  {len(titles_norm)} van ban.")

    n_with_entity = 0
    n_already_covered = 0
    n_missing_from_top5 = 0
    n_found_in_corpus = 0
    n_injected_questions = 0
    n_total_injected = 0
    examples = []

    output = {}
    for qid, predicted_ids in preds.items():
        q_text = question_texts.get(qid, "")
        entities = extract_entities(q_text)
        result = list(predicted_ids)

        if entities:
            n_with_entity += 1
            already_covered = any(
                normalize(e) in titles_norm.get(doc_id, "") for doc_id in predicted_ids for e in entities
            )
            if already_covered:
                n_already_covered += 1
            else:
                n_missing_from_top5 += 1
                # tim trong TOAN BO corpus (uu tien cum DAI nhat/cu the nhat truoc)
                candidates = []
                for e in sorted(entities, key=len, reverse=True):
                    e_norm = normalize(e)
                    for doc_id, t in titles_norm.items():
                        if doc_id in predicted_ids:
                            continue
                        if e_norm in t:
                            candidates.append((doc_id, e))
                if candidates:
                    n_found_in_corpus += 1
                    n_inj = 0
                    for doc_id, e in candidates:
                        if n_inj >= args.max_inject:
                            break
                        if doc_id in result:
                            continue
                        if len(result) >= 5:
                            result[-1] = doc_id
                        else:
                            result.append(doc_id)
                        n_inj += 1
                        if len(examples) < 10:
                            examples.append((qid, e, doc_id))
                    if n_inj > 0:
                        n_injected_questions += 1
                        n_total_injected += n_inj

        output[qid] = {"answer": result}

    print("=" * 70)
    print(f"Tong so cau hoi: {len(preds)}")
    print(f"So cau hoi phat hien duoc >=1 thuc the trong cau hoi: {n_with_entity} ({n_with_entity/len(preds)*100:.1f}%)")
    print(f"  - Da co san trong tieu de 1 trong top-5 (khong can lam gi): {n_already_covered}")
    print(f"  - VANG MAT khoi tieu de ca top-5 (can tim them): {n_missing_from_top5}")
    print(f"    - Tim thay trong corpus (co ung vien de bom): {n_found_in_corpus}")
    print(f"So cau hoi THUC SU bi sua doi (da bom): {n_injected_questions} ({n_injected_questions/len(preds)*100:.1f}%)")
    print(f"Tong so tai lieu da bom: {n_total_injected}")
    print("=" * 70)
    print("\nVi du da bom (kiem tra thu bang mat):")
    for qid, entity, doc_id in examples:
        print(f"  Cau hoi {qid}: thuc the '{entity}' -> bom doc_id={doc_id}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu predictions MOI tai: {args.output}")
    print(
        "\nBuoc tiep theo - do va so sanh Recall:\n"
        f"  python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json "
        f"--predictions {args.predictions}\n"
        f"  python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json "
        f"--predictions {args.output}\n"
        "Gui lai CA 2 ket qua (dong 'Recall trung binh' + bang breakdown theo n_relevant)."
    )


if __name__ == "__main__":
    main()
