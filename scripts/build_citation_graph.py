"""
Xay dung "citation graph" - do thi lien ket phap ly giua cac van ban trong corpus, dua tren cac so
hieu van ban duoc trich dan ngay trong phan mo dau (preamble: dong "So hieu:", "Luat so:", "Can cu...",
"Sua doi, bo sung...") cua moi van ban - du lieu nay da co san trong corpus, khong can nguon ngoai.

Dong luc: doc truc tiep 53 case n_relevant>=2 bi miss (xem submission-log-task1.md, muc 2026-08-16
toi) phat hien ~70% nguyen nhan la "van ban lien ket phap ly" bi miss - vd tim dung Nghi dinh GOC
nhung miss chinh Nghi dinh SUA DOI no, du 2 van ban khong co overlap tu vung/ngu nghia ro rang nao
ngoai so hieu duoc trich dan cheo. Y tuong: parse quan he trich dan nay ra thanh 1 do thi, dung de
"bom" them van ban lien quan vao ket qua retrieval o buoc sau.

**Day la PHAN 1 CUA PLAN - CHI xay do thi + bao cao thong ke de kiem tra tinh kha thi truoc khi viet
PHAN 2 (bom vao pipeline retrieval).** Chua sua gi trong full_pipeline_eval.py. Chay xong, gui lai
toan bo output terminal (đặc biệt % trich xuat duoc so hieu rieng, % van ban co edge, vai vi du edge)
de quyet dinh buoc tiep theo - neu ty le parse qua thap thi huong nay khong kha thi, can dung lai va
nghi cach khac truoc khi ton cong viet Phan 2.

Cach dung (tu thu muc goc UIT-data-science-2026):
    python scripts/build_citation_graph.py

Khong can GPU, khong can model - chi doc text + regex, uoc ~1-2 phut cho 351K chunk.

Output: outputs/citation_graph.json ({doc_id: [doc_id lien quan, ...]}) + in thong ke ra terminal.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# So hieu van ban phap luat VN: vd "82/NQ-CP", "22/2008/QH12", "111/2022/ND-CP", "10/CT-TTg",
# "04/2015/TT-BCA". Yeu cau co doan chu hoa (ma loai van ban/co quan) o cuoi de tranh nham voi ngay
# thang (vd "18/6/2022" se KHONG khop vi khong co doan chu).
NUM_RE = re.compile(
    r"\b(\d{1,4}[A-Za-zĐ]{0,3}(?:/\d{4})?/[A-ZĐ]{2,10}(?:-[A-ZĐ]{2,10})*\d{0,3})\b"
)

# Nhan thuong dung ngay truoc so hieu CHINH cua van ban (phan biet voi so hieu duoc TRICH DAN o noi
# khac trong van ban).
OWN_NUM_RE = re.compile(
    r"(?:Số hiệu|Số|Luật số|Nghị quyết số|Quyết định số|Pháp lệnh số|Lệnh số|Nghị định số|"
    r"Thông tư số|Thông tư liên tịch số|Chỉ thị số)\s*:\s*"
    r"(\d{1,4}[A-Za-zĐ]{0,3}(?:/\d{4})?/[A-ZĐ]{2,10}(?:-[A-ZĐ]{2,10})*\d{0,3})"
)

# v2 (sau khi thay chay thu: bam BAT KY so hieu nao trong preamble tao ra 1 "hub" gia (max degree 309)
# - gan chac la do trich dan cac luat khung chung chung ma VAN BAN NAO CUNG co trong "Can cu..." (vd
# Luat To chuc Chinh phu) - khong lien quan chu de cau hoi, chi la noise. CHI giu lai so hieu nam GAN
# (trong 1 cua so ky tu ngan SAU) 1 cum tu tin hieu THAT - dung cum tu da thay lap lai trong 53 case
# loi da doc (submission-log-task1.md, muc 2026-08-16 toi): "sua doi, bo sung", "huong dan thi hanh",
# "quy dinh chi tiet", "thay the", "bai bo" - day la cach VAN BAN "CON" tu mo ta quan he voi van ban
# "GOC" trong chinh tieu de/loi noi dau cua no.
TRIGGER_RE = re.compile(
    r"(?:sửa đổi\s*,?\s*bổ sung|bổ sung một số điều|sửa đổi một số điều|hướng dẫn thi hành|"
    r"hướng dẫn thực hiện|quy định chi tiết|thay thế|bãi bỏ)",
    re.IGNORECASE,
)
TRIGGER_WINDOW = 150  # so ky tu toi da sau moi cum tu tin hieu de tim so hieu van ban duoc nhac toi

# v3 (sau khi soi tiep vi du that): TRIGGER_WINDOW=150 van bi "tran cua so" sang menh de "Can cu...;"
# KE TIEP (vd "...sua doi, bo sung mot so dieu cua Luat X;\nCan cu Nghi dinh so Y..." - Y KHONG lien
# quan gi den "sua doi" ma chi la mot Can cu khac ngay sau dau ";"). Van ban phap luat VN dung dau ";"
# lam ranh gioi giua cac menh de "Can cu...;" lien tiep mot cach rat nhat quan - CHAN cua so tim kiem
# ngay tai dau ";" GAN NHAT sau cum tin hieu de khong tran sang menh de khac. LUU Y: KHONG dung "\n"
# lam ranh gioi - text corpus co xuong dong tuy tien GIUA CAU (noise tu qua trinh trich xuat PDF/HTML
# goc, xem vi du "SUA DOI, BO SUNG MOT SO DIEU\nCUA NGHI DINH SO 45/2010" - xuong dong ngay giua cum
# tu, cat o day se lam mat ca cac match THAT).
CLAUSE_BOUNDARY_RE = re.compile(r";")

PREAMBLE_CHUNKS = 3  # so chunk dau tien / doc dung de tim so hieu rieng + cac so hieu duoc trich dan
                      # (KHONG load toan bo 351K chunk vao bo nho - chi giu toi da 3 chunk dau/doc)


def load_preambles(path: str) -> dict:
    """doc_id -> preamble text (noi cac chunk dau tien, gioi han PREAMBLE_CHUNKS chunk/doc)."""
    preambles: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc_id = str(d["doc_id"])
            lst = preambles.setdefault(doc_id, [])
            if len(lst) < PREAMBLE_CHUNKS:
                lst.append(d.get("passage", ""))
    return {doc_id: " ".join(chunks) for doc_id, chunks in preambles.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", default="data/processed/chunks_corpus.jsonl")
    parser.add_argument("--output", default="outputs/citation_graph.json")
    parser.add_argument("--examples", type=int, default=15, help="So vi du edge in ra de kiem tra bang mat")
    args = parser.parse_args()

    print(f"Doc chunks tu {args.chunks} (chi lay {PREAMBLE_CHUNKS} chunk dau/doc de tiet kiem bo nho)...")
    preambles = load_preambles(args.chunks)
    print(f"Tong so van ban (doc_id) trong corpus: {len(preambles)}")

    # Pass 1: tim so hieu RIENG cua tung van ban -> xay bang tra nguoc so_hieu -> doc_id
    own_number: dict[str, str] = {}       # doc_id -> so_hieu
    number_to_doc: dict[str, str] = {}    # so_hieu -> doc_id
    collisions = 0
    for doc_id, text in preambles.items():
        m = OWN_NUM_RE.search(text[:1500])  # so hieu rieng luon o rat dau van ban
        if not m:
            continue
        num = m.group(1)
        own_number[doc_id] = num
        if num in number_to_doc and number_to_doc[num] != doc_id:
            collisions += 1
            continue
        number_to_doc[num] = doc_id

    n_with_own_number = len(own_number)
    print(
        f"Trich xuat duoc so hieu RIENG cho {n_with_own_number}/{len(preambles)} van ban "
        f"({n_with_own_number/len(preambles)*100:.1f}%) - {collisions} truong hop trung so hieu (bo qua)."
    )

    # Pass 2a (CHAN DOAN, khong dung de xuat do thi cuoi): bam BAT KY so hieu nao trong preamble - de
    # so sanh voi cach lam co muc tieu hon (v2) ben duoi, xem giam noise duoc bao nhieu.
    naive_edges: dict[str, set] = defaultdict(set)
    for doc_id, text in preambles.items():
        self_num = own_number.get(doc_id)
        for m in NUM_RE.finditer(text):
            num = m.group(1)
            if num == self_num:
                continue
            target = number_to_doc.get(num)
            if target and target != doc_id:
                naive_edges[doc_id].add(target)
                naive_edges[target].add(doc_id)
    naive_degree = {d: len(v) for d, v in naive_edges.items() if v}
    if naive_degree:
        top5_naive = sorted(naive_degree.items(), key=lambda kv: -kv[1])[:5]
        print("\n[CHAN DOAN] Cach cu (bam MOI so hieu trong preamble, kho de bi 'hub' gia do Can cu... chung chung):")
        print(f"  So canh: {sum(naive_degree.values())//2}, bac trung binh: {sum(naive_degree.values())/len(naive_degree):.2f}, max: {max(naive_degree.values())}")
        print("  Top 5 van ban co bac cao nhat (nghi ngo la 'hub' luat khung, khong phai tin hieu that):")
        for doc_id, deg in top5_naive:
            print(f"    doc_id={doc_id} ({own_number.get(doc_id, '?')}) - bac={deg}")

    # Pass 2b (DUNG DE XUAT DO THI CHINH THUC): chi bam so hieu nam GAN 1 cum tu tin hieu THAT
    # (TRIGGER_RE) - loai bo hau het cac trich dan "Can cu Luat khung..." chung chung khong lien quan
    # chu de cau hoi.
    edges: dict[str, set] = defaultdict(set)
    citation_context: dict[str, list] = defaultdict(list)  # target_doc_id -> [(citing_doc_id, citing_so_hieu, window_text), ...] toi da 3 vi du/target, de kiem tra bang mat
    n_docs_with_edge = 0
    n_total_edges = 0
    for doc_id, text in preambles.items():
        self_num = own_number.get(doc_id)
        found_here = set()
        for trig in TRIGGER_RE.finditer(text):
            raw_window = text[trig.end(): trig.end() + TRIGGER_WINDOW]
            boundary = CLAUSE_BOUNDARY_RE.search(raw_window)
            window = raw_window[: boundary.start()] if boundary else raw_window
            m = NUM_RE.search(window)
            if not m:
                continue
            num = m.group(1)
            if num == self_num:
                continue
            target = number_to_doc.get(num)
            if target and target != doc_id:
                found_here.add(target)
                if len(citation_context[target]) < 3:
                    trig_text = trig.group(0)
                    citation_context[target].append(
                        (doc_id, own_number.get(doc_id, "?"), f"...{trig_text}{window.strip()[:180]}...")
                    )
        if found_here:
            n_docs_with_edge += 1
            for target in found_here:
                if target not in edges[doc_id]:
                    edges[doc_id].add(target)
                    edges[target].add(doc_id)  # 2 chieu - vi trong 53 case da doc, chieu bi miss co
                    n_total_edges += 1          # the la A(goc)->B(sua doi) hoac nguoc lai, tuy case

    print(
        f"\n[CHINH THUC - dung TRIGGER_RE] {n_docs_with_edge}/{len(preambles)} van ban "
        f"({n_docs_with_edge/len(preambles)*100:.1f}%) trich dan >=1 van ban khac TRONG CORPUS voi "
        f"cum tu tin hieu ro rang (sua doi/bo sung/huong dan thi hanh/quy dinh chi tiet/thay the/bai bo)."
    )
    print(f"Tong so canh (edge) duy nhat trong do thi: {n_total_edges}")

    degree = {doc_id: len(v) for doc_id, v in edges.items() if v}
    if degree:
        avg_deg = sum(degree.values()) / len(degree)
        max_deg = max(degree.values())
        print(f"So van ban co >=1 edge: {len(degree)}. Bac trung binh: {avg_deg:.2f}, max: {max_deg}")
        top5 = sorted(degree.items(), key=lambda kv: -kv[1])[:5]
        print("  Top 5 van ban co bac cao nhat (kiem tra xem con la 'hub' gia khong):")
        for doc_id, deg in top5:
            title = preambles.get(doc_id, "")[:220].replace("\n", " ")
            print(f"    doc_id={doc_id} ({own_number.get(doc_id, '?')}) - bac={deg}")
            print(f"      Tieu de/dau van ban: {title}...")
            for citer_id, citer_num, ctx in citation_context.get(doc_id, []):
                print(f"      <- doc_id={citer_id} ({citer_num}) trich dan qua doan: {ctx}")

    print(f"\n{args.examples} vi du edge (kiem tra thu bang mat, doi chieu voi so hieu that):")
    shown = 0
    for doc_id, targets in edges.items():
        if shown >= args.examples or not targets:
            if shown >= args.examples:
                break
            continue
        t = sorted(targets)[0]
        print(f"  doc_id={doc_id} ({own_number.get(doc_id, '?')}) <-> doc_id={t} ({own_number.get(t, '?')})")
        shown += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph_out = {doc_id: sorted(v) for doc_id, v in edges.items() if v}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_out, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu do thi tai: {args.output} ({len(graph_out)} van ban co it nhat 1 lien ket)")
    print(
        "\nBuoc tiep theo: gui lai TOAN BO output tren cho Claude - se quyet dinh co dang viet Phan 2 "
        "(bom vao pipeline retrieval + do lai Recall tren val_split_20) hay khong dua vao ty le parse "
        "thuc te nay."
    )


if __name__ == "__main__":
    main()
