from __future__ import annotations

import re
import unicodedata


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_XML_WRAPPER_RE = re.compile(
    r"</?(?:answer|major_premise|minor_premise|conclusion)\b[^>]*>",
    re.IGNORECASE,
)
_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:###\s*)?(?:trả\s*lời|answer)\s*:\s*", re.IGNORECASE
)
_LEADING_META_EVIDENCE_RE = re.compile(
    r"^\s*Dựa\s+(?:trên|vào)\s+"
    r"(?=[^,:;\n]{0,220}(?:ngữ\s*cảnh|tài\s*liệu|thông\s*tin\s+được\s+cung\s*cấp))"
    r"[^,:;\n]{1,220}[,:;]\s*",
    re.IGNORECASE,
)
_LEADING_NUMBERED_CONTEXT_RE = re.compile(
    r"^\s*Dựa\s+(?:trên|vào)\s+(?:các\s+)?ngữ\s*cảnh(?:\s+\d+)?"
    r"(?:\s*\([^)]{0,1200}\))?\s*[,;:]\s*",
    re.IGNORECASE,
)
_LEADING_ANSWER_BRIDGE_RE = re.compile(
    r"^\s*(?:câu\s+trả\s+lời\s+là|có\s+thể\s+trả\s+lời\s+(?:là|như\s+sau))\s*[:,-]?\s*",
    re.IGNORECASE,
)


def _uppercase_first_letter(text: str) -> str:
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1 :]
    return text


def strip_leading_meta_prefix(text: str) -> tuple[str, str | None]:
    """Remove a metric-unfriendly evidence preamble without touching legal premises.

    Only a leading ``Dựa trên/vào ...`` clause that explicitly refers to supplied
    context, documents, or information is removed.  Substantive premises such as
    ``Dựa trên quy định tại Điều 5`` are deliberately preserved.
    """
    if not text:
        return text, None
    match = _LEADING_NUMBERED_CONTEXT_RE.match(text)
    if match is None:
        match = _LEADING_META_EVIDENCE_RE.match(text)
    if match is None:
        return text, None
    prefix = match.group(0).strip()
    cleaned = text[match.end() :]
    cleaned = _LEADING_ANSWER_BRIDGE_RE.sub("", cleaned, count=1).strip()
    return _uppercase_first_letter(cleaned), prefix


def clean_generated_answer(text: str) -> str:
    """Remove reasoning wrappers and normalize text for Task 2 submission."""
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = _THINK_RE.sub("", text)
    text = _XML_WRAPPER_RE.sub("", text)
    text = _LEADING_LABEL_RE.sub("", text)
    text = text.replace("```", "")
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text, _ = strip_leading_meta_prefix(text.strip())
    return text


def unsupported_legal_markers(answer: str, context: str) -> list[str]:
    """Return high-risk legal markers present in the answer but absent from evidence."""
    marker_re = re.compile(
        r"(?:\bĐiều\s+\d+[A-Za-z]?\b|\b\d+(?:[.,]\d+)?\s*%|"
        r"\b\d+(?:[.,]\d+)?\s*(?:đồng|triệu đồng|tỷ đồng|ngày|tháng|năm)\b)",
        re.IGNORECASE,
    )
    # Legal documents frequently split headings across lines (for example
    # ``Điều\n\n86``). Collapse whitespace before comparison to avoid reporting
    # citations that are in fact present in the supplied evidence.
    context_folded = re.sub(
        r"\s+", " ", unicodedata.normalize("NFC", context).casefold()
    ).strip()
    unsupported: list[str] = []
    for marker in marker_re.findall(answer):
        marker_folded = re.sub(
            r"\s+", " ", unicodedata.normalize("NFC", marker).casefold()
        ).strip()
        if marker_folded not in context_folded and marker not in unsupported:
            unsupported.append(marker)
    return unsupported
