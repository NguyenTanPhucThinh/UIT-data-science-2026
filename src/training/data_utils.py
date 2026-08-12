"""
Tiện ích load dữ liệu dùng chung cho training/evaluation Task 1 - LegalIR.

Đọc trực tiếp từ selected-contexts.zip (không cần giải nén 8.532 file JSON ra đĩa) và train.json.
"""
from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterator, List

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    name: str
    link: str
    passage: str


class CorpusReader:
    """Đọc lười (lazy) từng văn bản từ selected-contexts.zip.

    Tránh load hết ~97MB / 8.532 văn bản vào RAM cùng lúc - chỉ đọc văn bản nào thực sự cần.
    """

    _PATTERN = re.compile(r"context_(\d+)\.json$")

    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self._zip = zipfile.ZipFile(zip_path)
        self._name_by_id: Dict[str, str] = {}
        for name in self._zip.namelist():
            m = self._PATTERN.search(name)
            if m:
                self._name_by_id[m.group(1)] = name

    def __len__(self) -> int:
        return len(self._name_by_id)

    def ids(self) -> List[str]:
        return list(self._name_by_id.keys())

    def get(self, doc_id: str) -> Document:
        name = self._name_by_id.get(str(doc_id))
        if name is None:
            raise KeyError(f"document_id không tồn tại trong corpus: {doc_id}")
        raw = json.loads(self._zip.read(name))
        return Document(
            id=str(raw.get("id", doc_id)),
            name=raw.get("name", ""),
            link=raw.get("link", ""),
            passage=raw.get("passage", ""),
        )

    def iter_documents(self) -> Iterator[Document]:
        for doc_id in self._name_by_id:
            yield self.get(doc_id)


def load_labeled_questions(train_json_path: str) -> Dict[str, dict]:
    """Đọc train.json -> {qid: {"question": str, "answer": [doc_id, ...]}}.

    Tự động bỏ qua các câu hỏi có answer=null/rỗng (đề phòng lỡ trỏ nhầm vào public-official.json).
    """
    with open(train_json_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {qid: item for qid, item in raw.items() if item.get("answer")}


# Nhận diện ranh giới "Điều X" trong văn bản luật VN để chunk theo cấu trúc pháp lý
# thay vì cắt cứng theo độ dài - xem đề xuất pipeline đã thống nhất trong docs/.
_ARTICLE_SPLIT_RE = re.compile(r"(?=\n?Điều\s+\d+[\.\:])")


def chunk_passage(passage: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """Chunk văn bản pháp luật: ưu tiên tách theo 'Điều X', fallback cắt theo cửa sổ ký tự có overlap.

    LƯU Ý: đây là bản triển khai BASELINE dùng tạm cho training/evaluation. Khi module
    `src/ingestion + EDA` của nhóm hoàn thiện chunker chính thức (chuẩn hóa Unicode, xử lý
    header/footer, metadata...), nên thay hàm này bằng chunker dùng chung, để đảm bảo training
    và serving (retrieval thật) dùng cùng một cách chia đoạn văn bản - tránh lệch phân bố.
    """
    passage = passage.strip()
    if not passage:
        return []

    articles = [a.strip() for a in _ARTICLE_SPLIT_RE.split(passage) if a.strip()]
    if len(articles) <= 1:
        articles = [passage]

    chunks: List[str] = []
    for article in articles:
        if len(article) <= max_chars:
            chunks.append(article)
            continue
        start = 0
        while start < len(article):
            end = start + max_chars
            chunks.append(article[start:end])
            if end >= len(article):
                break
            start = end - overlap
    return chunks


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text)}


def resolve_device(requested: str = "auto") -> str:
    """Chọn device ("cuda" hoặc "cpu") cho model embedding/reranker, và LUÔN in rõ ra log đang
    chạy trên device nào.

    Lý do cần hàm này: sentence-transformers/CrossEncoder nếu không truyền `device` sẽ tự chọn
    ngầm - có thể lặng lẽ chạy trên CPU dù máy có GPU (vd. bản torch cài thiếu CUDA), khiến việc
    encode toàn bộ corpus (hàng trăm nghìn chunk) mất HÀNG GIỜ thay vì vài phút mà không có cảnh
    báo gì, trông như treo máy.
    """
    import torch

    device = requested if requested != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cpu":
        logger.warning(
            "Đang chạy trên CPU - với corpus lớn (hàng trăm nghìn chunk) việc encode có thể mất "
            "HÀNG GIỜ. Nếu máy có GPU NVIDIA, kiểm tra: "
            'python -c "import torch; print(torch.cuda.is_available())" '
            "- nếu ra False dù có GPU, khả năng cao bản torch đang cài KHÔNG có CUDA. Kiểm tra "
            "bản CUDA của driver bằng lệnh `nvidia-smi`, rồi cài lại torch đúng bản tương ứng "
            "theo hướng dẫn tại https://pytorch.org/get-started/locally/ "
            "(vd: pip install torch --index-url https://download.pytorch.org/whl/cu121)."
        )
    else:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "?"
        logger.info("Đang dùng GPU: %s", gpu_name)

    return device


def select_best_chunk(question: str, chunks: List[str]) -> str:
    """Chọn chunk có độ trùng từ vựng (word overlap) cao nhất với câu hỏi.

    BTC chỉ gán nhãn ở CẤP DOCUMENT (không chỉ rõ đoạn/Điều nào trả lời câu hỏi), nên khi cần
    một đoạn văn bản cụ thể làm ví dụ "positive" để fine-tune embedding, ta cần tự chọn ra đoạn
    khả năng cao nhất chứa câu trả lời. Heuristic đơn giản: đếm số từ trùng giữa câu hỏi và từng
    chunk, chọn chunk có điểm cao nhất.

    Đây KHÔNG phải chunk đầu tiên theo mặc định - chunk đầu của một văn bản luật thường chỉ là
    phần quốc hiệu/tiêu đề ("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM...", số hiệu, ngày ban hành...),
    hầu như không bao giờ chứa nội dung trả lời câu hỏi - nếu lấy nhầm sẽ dạy model sai.
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]

    q_tokens = _tokenize(question)
    if not q_tokens:
        return chunks[0]

    best_chunk = chunks[0]
    best_score = -1
    for chunk in chunks:
        score = len(q_tokens & _tokenize(chunk))
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk
