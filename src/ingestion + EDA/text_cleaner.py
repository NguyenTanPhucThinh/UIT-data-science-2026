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
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0]', ' ', text)
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

def generate_fallback_passage(doc_id: str, name: Optional[str], link: str) -> str:
    """Constructs a concise fallback passage for documents with empty passages."""
    url_title = extract_title_from_url(link)
    doc_title = name if (name and len(name.strip()) > 0) else url_title
    
    parts = []
    if doc_title:
        parts.append(f"VĂN BẢN PHÁP LUẬT: {doc_title}")
    if link:
        parts.append(f"Nguồn: {link}")
    parts.append(f"Nội dung văn bản (định danh ID {doc_id}): {doc_title}")
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
