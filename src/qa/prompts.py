from __future__ import annotations


SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp pháp luật Việt Nam. Hãy trả lời trung thực, chính xác "
    "và chỉ dựa trên ngữ cảnh được cung cấp."
)


def build_user_prompt(question: str, context: str) -> str:
    return f"""Yêu cầu:
- Trả lời trực tiếp và đầy đủ đúng trọng tâm câu hỏi.
- Bắt đầu ngay bằng nội dung trả lời hoặc căn cứ thực sự có trong ngữ cảnh.
- Tuyệt đối không mở đầu bằng các cụm như "Dựa trên ngữ cảnh được cung cấp", "Dựa trên các tài liệu được cung cấp" hoặc lời dẫn tương tự.
- Ưu tiên giữ nguyên thuật ngữ, con số, điều/khoản và tên văn bản trong ngữ cảnh.
- Không tự tạo quy định, số điều, số liệu hoặc tên văn bản không có trong ngữ cảnh.
- Không xuất suy luận nội bộ, thẻ XML, Markdown hoặc lời cảnh báo chung.
- Nếu ngữ cảnh chưa đủ, chỉ nêu phần có thể xác nhận và không suy đoán.

Ngữ cảnh:
{context}

Câu hỏi:
{question.strip()}

Trả lời:"""


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, context)},
    ]
