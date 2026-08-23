import re
import unicodedata
from typing import Optional

def normalize_unicode(text: str) -> str:
    """Normalize text to Unicode NFC standard form."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)

def clean_whitespace(text: str) -> str:
    """Clean and normalize line endings, spaces, and linebreaks."""
    if not text:
        return ""
    # Convert \r\n and \r to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove zero-width spaces & non-breaking spaces
    text = re.sub(r'[​‌‍﻿\xa0]', ' ', text)
    # Strip line-trailing whitespace
    lines = [line.strip() for line in text.split("\n")]
    # Reduce consecutive empty lines to maximum 2
    cleaned_lines = []
    empty_count = 0
    for line in lines:
        if line == "":
            empty_count += 1
            if empty_count <= 2:
                cleaned_lines.append(line)
        else:
            empty_count = 0
            # Normalize internal multiple spaces
            line = re.sub(r'[ \t]+', ' ', line)
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def extract_title_from_url(link: str) -> str:
    """Extract and format human-readable title from thu vien phap luat URL slug."""
    if not link:
        return ""
    parts = link.split('?')[0].split('#')[0].rstrip('/').split('/')
    if not parts:
        return ""
    last_part = parts[-1]
    if last_part.endswith('.aspx'):
        last_part = last_part[:-5]
    # Remove trailing document numeric ID like -915197
    slug = re.sub(r'-\d+$', '', last_part)
    title = slug.replace('-', ' ')
    return title.strip()

# --- Bảng tra CỐ ĐỊNH dùng cho fallback passage (2026-08-21) ---------------------------------
# Chỉ dùng dữ liệu ĐÃ CÓ SẴN trong corpus chính thức (link/doc_id) — KHÔNG crawl/thu thập dữ liệu
# ngoài (tuân thủ quy-dinh-cuoc-thi-task1.md mục 7). Các bảng tra dưới đây là kiến thức hành chính/
# ngôn ngữ TỔNG QUÁT, cố định, không đặc thù cho bất kỳ văn bản cụ thể nào trong corpus — quét toàn
# bộ 8.532 link trong corpus chỉ tìm được đúng 96 cặp (loại văn bản, lĩnh vực) khác nhau nên đủ nhỏ
# để tra cứu thủ công 1 lần, không cần cập nhật theo dữ liệu.

LINH_VUC_LABELS = {
    'cong-nghe-thuc-pham': 'Công nghệ thực phẩm',
    'cong-nghiep': 'Công nghiệp',
    'dien-dien-tu': 'Điện - Điện tử',
    'giao-thong': 'Giao thông',
    'linh-vuc-khac': 'Lĩnh vực khác',
    'nong-nghiep': 'Nông nghiệp',
    'tai-nguyen-moi-truong': 'Tài nguyên - Môi trường',
    'xay-dung': 'Xây dựng',
    'hoa-chat': 'Hóa chất',
    'bao-hiem': 'Bảo hiểm',
    'bat-dong-san': 'Bất động sản',
    'bo-may-hanh-chinh': 'Bộ máy hành chính',
    'chung-khoan': 'Chứng khoán',
    'cong-nghe-thong-tin': 'Công nghệ thông tin',
    'dau-tu': 'Đầu tư',
    'dich-vu-phap-ly': 'Dịch vụ pháp lý',
    'doanh-nghiep': 'Doanh nghiệp',
    'giao-duc': 'Giáo dục',
    'giao-thong-van-tai': 'Giao thông - Vận tải',
    'ke-toan-kiem-toan': 'Kế toán - Kiểm toán',
    'lao-dong-tien-luong': 'Lao động - Tiền lương',
    'quyen-dan-su': 'Quyền dân sự',
    'so-huu-tri-tue': 'Sở hữu trí tuệ',
    'tai-chinh-nha-nuoc': 'Tài chính nhà nước',
    'the-thao-y-te': 'Thể thao - Y tế',
    'thu-tuc-to-tung': 'Thủ tục tố tụng',
    'thue-phi-le-phi': 'Thuế - Phí - Lệ phí',
    'thuong-mai': 'Thương mại',
    'tien-te-ngan-hang': 'Tiền tệ - Ngân hàng',
    'trach-nhiem-hinh-su': 'Trách nhiệm hình sự',
    'van-hoa-xa-hoi': 'Văn hóa - Xã hội',
    'vi-pham-hanh-chinh': 'Vi phạm hành chính',
    'xay-dung-do-thi': 'Xây dựng - Đô thị',
    'xuat-nhap-khau': 'Xuất nhập khẩu',
}

# Mã cơ quan ban hành viết hoa xuất hiện trong slug (vd "...-BLDTBXH-...") đôi khi thiếu dấu so với
# tên viết tắt chính thức — chỉ sửa đúng những mã ĐÃ BIẾT CHẮC CHẮN (kiến thức hành chính công khai,
# không suy đoán riêng cho văn bản nào). Mã không có trong bảng này giữ nguyên như trong URL.
AGENCY_CODE_FIXES = {
    'BLDTBXH': 'BLĐTBXH',  # Bộ Lao động - Thương binh và Xã hội
}

# Token đứng ngay sau năm nhưng KHÔNG phải mã cơ quan ban hành VN (vd tiêu chuẩn quốc tế được viện
# dẫn trong tên TCVN, như "...TCVN-6102-2020-ISO-7202-2018-...") — loại trừ để không bị nhận nhầm
# thành SỐ HIỆU/<mã cơ quan>.
EXCLUDED_AGENCY_TOKENS = {'ISO', 'EN', 'ASTM', 'JIS', 'BS', 'DIN'}


def extract_so_hieu(link: str) -> Optional[str]:
    """Tách SỐ HIỆU trực tiếp từ slug URL — chỉ dùng dữ liệu chính thức đã có sẵn (link), không suy
    đoán/thu thập ngoài. Chỉ nhận diện được cho slug dạng TCVN-.../QCVN-... (có số + năm rõ ràng ngay
    sau tiền tố loại văn bản). Với slug dạng "van-ban"/"cong-van" (thường chỉ mô tả tên văn bản, không
    có số hiệu tách được an toàn từ URL) trả về None — KHÔNG đoán để tránh sai lệch."""
    if not link:
        return None
    last_part = link.split('?')[0].split('#')[0].rstrip('/').split('/')[-1]
    if last_part.endswith('.aspx'):
        last_part = last_part[:-5]
    last_part = re.sub(r'-\d+$', '', last_part)  # bỏ mã trang cuối (vd -915197)
    tokens = last_part.split('-')
    if not tokens:
        return None
    doc_type = tokens[0].upper()
    if doc_type not in ('TCVN', 'QCVN'):
        return None
    i = 1
    num_parts = []
    while i < len(tokens) and tokens[i].isdigit():
        num_parts.append(tokens[i])
        i += 1
    if len(num_parts) < 2:
        return None  # cần ít nhất 1 số hiệu + 1 năm
    year = num_parts[-1]
    if len(year) != 4:
        return None
    main_num = '-'.join(num_parts[:-1])
    so_hieu = f"{doc_type} {main_num}:{year}"
    if (i < len(tokens) and tokens[i].isalpha() and tokens[i].isupper()
            and tokens[i] not in EXCLUDED_AGENCY_TOKENS):
        agency = AGENCY_CODE_FIXES.get(tokens[i], tokens[i])
        so_hieu += f"/{agency}"
    return so_hieu


def extract_linh_vuc(link: str) -> Optional[str]:
    """Tra LĨNH VỰC từ chính đường dẫn URL (đoạn thứ 2 sau domain, vd .../tcvn/Cong-nghiep/... ->
    'cong-nghiep') qua bảng LINH_VUC_LABELS cố định — dữ liệu này vốn đã có sẵn trong link chính thức,
    chỉ đang chuyển từ dạng slug không dấu sang nhãn tiếng Việt đầy đủ."""
    if not link:
        return None
    m = re.search(r'thuvienphapluat\.vn/([^/]+)/([^/]+)/', link)
    if not m:
        return None
    return LINH_VUC_LABELS.get(m.group(2).lower())


def generate_fallback_passage(doc_id: str, name: Optional[str], link: str) -> str:
    """Constructs a fallback passage for documents with empty passages.

    Cập nhật 2026-08-21: bổ sung SỐ HIỆU + LĨNH VỰC (tách/tra từ chính link chính thức đã có sẵn,
    xem extract_so_hieu()/extract_linh_vuc() ở trên) — KHÔNG crawl/thu thập dữ liệu ngoài, tuân thủ
    quy-dinh-cuoc-thi-task1.md mục 7. Tên văn bản (VĂN BẢN PHÁP LUẬT) CỐ TÌNH giữ nguyên dạng không
    dấu từ slug URL — phục hồi dấu tiếng Việt cho tên riêng từng văn bản cụ thể đòi hỏi biết trước nội
    dung thật của văn bản đó, rủi ro tương đương "thu thập dữ liệu ngoài" nên không thực hiện.
    """
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
    """Concise header normalization without aggressive content stripping."""
    if not text:
        return ""

    # Format standard legal preamble markers to cleaner line representations
    text = re.sub(r'(CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM)\s+(Độc lập - Tự do - Hạnh phúc)', r'\1 | \2', text)
    # Format "Số: 123/2022/NĐ-CP"
    text = re.sub(r'Số:\s*([0-9A-Za-z/-]+)', r'Số hiệu: \1', text)
    return text

def preprocess_text(text: str, doc_id: str = "", name: Optional[str] = None, link: str = "") -> str:
    """Full preprocessing pipeline for text passage."""
    if not text or len(text.strip()) == 0:
        text = generate_fallback_passage(doc_id, name, link)

    text = normalize_unicode(text)
    text = clean_whitespace(text)
    text = normalize_headers(text)
    return text
