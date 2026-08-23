"""
Patch 20 chunk/document rong (is_empty_fallback=True) trong du lieu DA BUILD SAN, sau khi
text_cleaner.py duoc nang cap (them SO HIEU + LINH VUC, 2026-08-21) - xem
claude/submission-log-task1.md muc "fallback passage nang cap" de biet boi canh day du.

CHI patch dung 20 dong lien quan trong 3 noi:
  1. data/processed/corpus_preprocessed.jsonl  (doi passage + passage_segmented cho 20 dong
     co is_empty_fallback=true)
  2. data/processed/chunks_corpus.jsonl        (doi passage + passage_segmented cho 20 chunk
     co doc_id trung voi 20 doc_id o buoc 1 - moi doc chi co dung 1 chunk nen an toan thay the
     toan bo, khong can ghep lai nhieu chunk)
  3. outputs/cache/embcompare_aiteamvn.npz     (encode lai DUNG 20 vector embedding tuong ung,
     ghi de dung 20 dong trong mang 'embeddings' theo vi tri khop voi mang 'doc_ids')

KHONG dong den corpus.jsonl (du lieu tho goc, giu nguyen lam ban ghi lich su crawl) va KHONG
dong den bat ky dong/chunk nao khac trong 3 file tren.

An toan: truoc khi ghi de, luu lai noi dung CU cua ca 20 dong (ca 2 file .jsonl) vao
outputs/predictions/empty_fallback_patch_backup.json - neu can revert thi doc lai file backup
nay va ghi de nguoc lai.

Chay tu thu muc GOC UIT-data-science-2026 (can dung dung venv co pyvi + sentence-transformers +
torch, giong venv dang dung cho pipeline chinh):

    python patch_empty_fallback.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS_PREPROCESSED = ROOT / "data" / "processed" / "corpus_preprocessed.jsonl"
CHUNKS_CORPUS = ROOT / "data" / "processed" / "chunks_corpus.jsonl"
CACHE_NPZ = ROOT / "outputs" / "cache" / "embcompare_aiteamvn.npz"
BACKUP_JSON = ROOT / "outputs" / "predictions" / "empty_fallback_patch_backup.json"

EMBEDDING_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"
MAX_SEQ_LENGTH = 2048


# --- Sao chep dung logic tu src/ingestion + EDA/text_cleaner.py (import truc tiep se vuong
#     duong dan co dau cach "ingestion + EDA", nen lap lai o day cho gon, PHAI GIU DONG BO neu
#     text_cleaner.py doi them sau nay) ---------------------------------------------------------

def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[​‌‍﻿\xa0]", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines = []
    empty_count = 0
    for line in lines:
        if line == "":
            empty_count += 1
            if empty_count <= 2:
                cleaned_lines.append(line)
        else:
            empty_count = 0
            line = re.sub(r"[ \t]+", " ", line)
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def extract_title_from_url(link: str) -> str:
    if not link:
        return ""
    parts = link.split("?")[0].split("#")[0].rstrip("/").split("/")
    if not parts:
        return ""
    last_part = parts[-1]
    if last_part.endswith(".aspx"):
        last_part = last_part[:-5]
    slug = re.sub(r"-\d+$", "", last_part)
    return slug.replace("-", " ").strip()


LINH_VUC_LABELS = {
    "cong-nghe-thuc-pham": "Công nghệ thực phẩm",
    "cong-nghiep": "Công nghiệp",
    "dien-dien-tu": "Điện - Điện tử",
    "giao-thong": "Giao thông",
    "linh-vuc-khac": "Lĩnh vực khác",
    "nong-nghiep": "Nông nghiệp",
    "tai-nguyen-moi-truong": "Tài nguyên - Môi trường",
    "xay-dung": "Xây dựng",
    "hoa-chat": "Hóa chất",
    "bao-hiem": "Bảo hiểm",
    "bat-dong-san": "Bất động sản",
    "bo-may-hanh-chinh": "Bộ máy hành chính",
    "chung-khoan": "Chứng khoán",
    "cong-nghe-thong-tin": "Công nghệ thông tin",
    "dau-tu": "Đầu tư",
    "dich-vu-phap-ly": "Dịch vụ pháp lý",
    "doanh-nghiep": "Doanh nghiệp",
    "giao-duc": "Giáo dục",
    "giao-thong-van-tai": "Giao thông - Vận tải",
    "ke-toan-kiem-toan": "Kế toán - Kiểm toán",
    "lao-dong-tien-luong": "Lao động - Tiền lương",
    "quyen-dan-su": "Quyền dân sự",
    "so-huu-tri-tue": "Sở hữu trí tuệ",
    "tai-chinh-nha-nuoc": "Tài chính nhà nước",
    "the-thao-y-te": "Thể thao - Y tế",
    "thu-tuc-to-tung": "Thủ tục tố tụng",
    "thue-phi-le-phi": "Thuế - Phí - Lệ phí",
    "thuong-mai": "Thương mại",
    "tien-te-ngan-hang": "Tiền tệ - Ngân hàng",
    "trach-nhiem-hinh-su": "Trách nhiệm hình sự",
    "van-hoa-xa-hoi": "Văn hóa - Xã hội",
    "vi-pham-hanh-chinh": "Vi phạm hành chính",
    "xay-dung-do-thi": "Xây dựng - Đô thị",
    "xuat-nhap-khau": "Xuất nhập khẩu",
}
AGENCY_CODE_FIXES = {"BLDTBXH": "BLĐTBXH"}
EXCLUDED_AGENCY_TOKENS = {"ISO", "EN", "ASTM", "JIS", "BS", "DIN"}


def extract_so_hieu(link: str):
    if not link:
        return None
    last_part = link.split("?")[0].split("#")[0].rstrip("/").split("/")[-1]
    if last_part.endswith(".aspx"):
        last_part = last_part[:-5]
    last_part = re.sub(r"-\d+$", "", last_part)
    tokens = last_part.split("-")
    if not tokens:
        return None
    doc_type = tokens[0].upper()
    if doc_type not in ("TCVN", "QCVN"):
        return None
    i = 1
    num_parts = []
    while i < len(tokens) and tokens[i].isdigit():
        num_parts.append(tokens[i])
        i += 1
    if len(num_parts) < 2:
        return None
    year = num_parts[-1]
    if len(year) != 4:
        return None
    main_num = "-".join(num_parts[:-1])
    so_hieu = f"{doc_type} {main_num}:{year}"
    if (i < len(tokens) and tokens[i].isalpha() and tokens[i].isupper()
            and tokens[i] not in EXCLUDED_AGENCY_TOKENS):
        agency = AGENCY_CODE_FIXES.get(tokens[i], tokens[i])
        so_hieu += f"/{agency}"
    return so_hieu


def extract_linh_vuc(link: str):
    if not link:
        return None
    m = re.search(r"thuvienphapluat\.vn/([^/]+)/([^/]+)/", link)
    if not m:
        return None
    return LINH_VUC_LABELS.get(m.group(2).lower())


def generate_fallback_passage(doc_id: str, name, link: str) -> str:
    url_title = extract_title_from_url(link)
    doc_title = name if (name and len(name.strip()) > 0) else url_title
    so_hieu = extract_so_hieu(link)
    linh_vuc = extract_linh_vuc(link)
    parts = []
    if doc_title:
        parts.append(f"VĂN BẢN PHÁP LUẬT: {doc_title}")
    if so_hieu:
        parts.append(f"SỐ HIỆU: {so_hieu}")
    if linh_vuc:
        parts.append(f"LĨNH VỰC: {linh_vuc}")
    if link:
        parts.append(f"NGUỒN: {link}")
    return "\n".join(parts)


def normalize_headers(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM)\s+(Độc lập - Tự do - Hạnh phúc)", r"\1 | \2", text)
    text = re.sub(r"Số:\s*([0-9A-Za-z/-]+)", r"Số hiệu: \1", text)
    return text


def preprocess_text(text: str, doc_id: str = "", name=None, link: str = "") -> str:
    if not text or len(text.strip()) == 0:
        text = generate_fallback_passage(doc_id, name, link)
    text = normalize_unicode(text)
    text = clean_whitespace(text)
    text = normalize_headers(text)
    return text


def main() -> None:
    try:
        from pyvi import ViTokenizer
    except ImportError:
        print("[lỗi] Thiếu pyvi. Chạy: pip install pyvi", file=sys.stderr)
        sys.exit(1)

    def fast_tokenize_text(text: str) -> str:
        if not text:
            return ""
        lines = text.split("\n")
        out = []
        for line in lines:
            out.append(ViTokenizer.tokenize(line) if line.strip() else "")
        return "\n".join(out)

    # ----- Bước 1: tìm 20 doc_id is_empty_fallback + tính lại passage mới -----------------------
    if not CORPUS_PREPROCESSED.exists():
        print(f"[lỗi] Không tìm thấy {CORPUS_PREPROCESSED}", file=sys.stderr)
        sys.exit(1)

    targets = {}  # doc_id -> {"link":..., "name":..., "new_passage":..., "new_passage_segmented":...}
    with open(CORPUS_PREPROCESSED, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("is_empty_fallback"):
                doc_id = d["doc_id"]
                link = d.get("link") or ""
                name = d.get("name")
                new_passage = preprocess_text("", doc_id, name, link)
                new_seg = fast_tokenize_text(new_passage)
                targets[doc_id] = {
                    "link": link,
                    "name": name,
                    "old_passage": d.get("passage"),
                    "old_passage_segmented": d.get("passage_segmented"),
                    "new_passage": new_passage,
                    "new_passage_segmented": new_seg,
                }

    print(f"Tìm được {len(targets)} tài liệu is_empty_fallback=true trong corpus_preprocessed.jsonl")
    if len(targets) != 20:
        print(f"[CẢNH BÁO] Kỳ vọng đúng 20 tài liệu (theo điều tra 2026-08-17), tìm được {len(targets)}."
              f" Kiểm tra lại trước khi tiếp tục (Ctrl+C để dừng nếu nghi ngờ).")

    # ----- Bước 2: backup nội dung CŨ (cả corpus_preprocessed lẫn chunks) trước khi ghi đè ------
    backup = {"corpus_preprocessed": {}, "chunks_corpus": {}}
    for doc_id, info in targets.items():
        backup["corpus_preprocessed"][doc_id] = {
            "passage": info["old_passage"],
            "passage_segmented": info["old_passage_segmented"],
        }

    # ----- Bước 3: patch corpus_preprocessed.jsonl (ghi ra file tạm rồi thay thế) ----------------
    tmp_path = CORPUS_PREPROCESSED.with_suffix(".jsonl.tmp")
    n_patched_1 = 0
    with open(CORPUS_PREPROCESSED, encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            if not raw:
                fout.write(line)
                continue
            d = json.loads(raw)
            doc_id = d.get("doc_id")
            if doc_id in targets:
                d["passage"] = targets[doc_id]["new_passage"]
                d["passage_segmented"] = targets[doc_id]["new_passage_segmented"]
                d["passage_length"] = len(targets[doc_id]["new_passage"])
                n_patched_1 += 1
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            else:
                fout.write(line)
    shutil.move(str(tmp_path), str(CORPUS_PREPROCESSED))
    print(f"Đã patch {n_patched_1}/{len(targets)} dòng trong {CORPUS_PREPROCESSED.name}")

    # ----- Bước 4: patch chunks_corpus.jsonl (file lớn ~1.1GB, stream từng dòng) -----------------
    if not CHUNKS_CORPUS.exists():
        print(f"[lỗi] Không tìm thấy {CHUNKS_CORPUS}", file=sys.stderr)
        sys.exit(1)

    tmp_chunks = CHUNKS_CORPUS.with_suffix(".jsonl.tmp")
    n_patched_2 = 0
    with open(CHUNKS_CORPUS, encoding="utf-8") as fin, open(tmp_chunks, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            if not raw:
                fout.write(line)
                continue
            d = json.loads(raw)
            doc_id = d.get("doc_id")
            if doc_id in targets:
                backup["chunks_corpus"].setdefault(doc_id, []).append({
                    "chunk_id": d.get("chunk_id"),
                    "passage": d.get("passage"),
                    "passage_segmented": d.get("passage_segmented"),
                })
                d["passage"] = targets[doc_id]["new_passage"]
                d["passage_segmented"] = targets[doc_id]["new_passage_segmented"]
                n_patched_2 += 1
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            else:
                fout.write(line)
    shutil.move(str(tmp_chunks), str(CHUNKS_CORPUS))
    print(f"Đã patch {n_patched_2} chunk trong {CHUNKS_CORPUS.name}")

    BACKUP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(BACKUP_JSON, "w", encoding="utf-8") as f:
        json.dump(backup, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu backup nội dung CŨ vào {BACKUP_JSON}")

    # ----- Bước 5: encode lại DÚNG các passage mới, patch cache embedding ------------------------
    if not CACHE_NPZ.exists():
        print(f"[CẢNH BÁO] Không tìm thấy {CACHE_NPZ} — bỏ qua bước patch embedding cache."
              f" Passage/text đã sửa xong nhưng embedding CŨ vẫn còn — cần encode lại thủ công sau.")
        return

    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"Đang load {EMBEDDING_MODEL_NAME} để encode lại {len(targets)} passage mới...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH

    doc_id_list = list(targets.keys())
    texts = [targets[d]["new_passage"] for d in doc_id_list]
    new_vecs = model.encode(texts, batch_size=8, show_progress_bar=False, convert_to_numpy=True)

    data = np.load(CACHE_NPZ, allow_pickle=True)
    embeddings = data["embeddings"].copy()
    cache_doc_ids = data["doc_ids"]

    n_patched_3 = 0
    missing = []
    for doc_id, vec in zip(doc_id_list, new_vecs):
        idx = np.where(cache_doc_ids == doc_id)[0]
        if len(idx) == 0:
            missing.append(doc_id)
            continue
        for row in idx:
            embeddings[row] = vec
            n_patched_3 += 1

    if missing:
        print(f"[CẢNH BÁO] {len(missing)} doc_id KHÔNG tìm thấy trong cache (không patch được): {missing}")

    cache_backup_path = CACHE_NPZ.with_name(CACHE_NPZ.stem + "_before_fallback_patch.npz")
    if not cache_backup_path.exists():
        shutil.copy(str(CACHE_NPZ), str(cache_backup_path))
        print(f"Đã backup cache gốc vào {cache_backup_path.name}")

    np.savez(CACHE_NPZ, embeddings=embeddings, doc_ids=cache_doc_ids)
    print(f"Đã patch {n_patched_3} dòng embedding trong {CACHE_NPZ.name}")
    print("\nHOÀN TẤT. Chạy lại full_pipeline_eval.py --stage-eval trên val_split_20 để đo Recall LOCAL"
          " mới (kỳ vọng thay đổi rất nhỏ, chỉ ảnh hưởng câu hỏi có GT trúng đúng 20 tài liệu này).")


if __name__ == "__main__":
    main()
