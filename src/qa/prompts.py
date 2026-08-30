from __future__ import annotations


SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp pháp luật Việt Nam. Hãy trả lời trung thực, chính xác "
    "và chỉ dựa trên ngữ cảnh được cung cấp."
)


PROMPT_VARIANTS: dict[str, str] = {
    "baseline": "Cấu hình production hiện tại: trả lời trực tiếp, grounded.",
    "answer_first": "Đưa kết luận/câu trả lời chính ở câu đầu, sau đó mới nêu điều kiện.",
    "legal_structured": "Ưu tiên kết luận, căn cứ pháp lý có trong evidence và điều kiện áp dụng.",
    "grounded_concise": "Chỉ nêu thông tin xác nhận được; giữ câu trả lời ngắn, sát đáp án.",
}


def available_prompt_variants() -> tuple[str, ...]:
    """Return stable prompt-variant names accepted by the LegalQA pipeline."""
    return tuple(PROMPT_VARIANTS)


def _variant_rules(prompt_variant: str) -> str:
    try:
        return {
            "baseline": """- Trả lời trực tiếp và đầy đủ đúng trọng tâm câu hỏi.
- Bắt đầu ngay bằng nội dung trả lời hoặc căn cứ thực sự có trong ngữ cảnh.
- Ưu tiên giữ nguyên thuật ngữ, con số, điều/khoản và tên văn bản trong ngữ cảnh.
- Nếu ngữ cảnh chưa đủ, chỉ nêu phần có thể xác nhận và không suy đoán.""",
            "answer_first": """- Câu đầu tiên phải trả lời thẳng kết luận mà câu hỏi cần biết; không lặp lại câu hỏi.
- Chỉ sau kết luận mới nêu điều kiện, ngoại lệ hoặc căn cứ thực sự cần thiết.
- Viết ngắn gọn, tránh dẫn nhập và tránh liệt kê thông tin không giúp trả lời câu hỏi.
- Nếu ngữ cảnh chưa đủ, chỉ nêu phần có thể xác nhận và không suy đoán.""",
            "legal_structured": """- Nêu kết luận trực tiếp trước.
- Khi ngữ cảnh có căn cứ pháp lý, nêu đúng điều/khoản/tên văn bản đó và điều kiện áp dụng liên quan.
- Không tạo mục, tiêu đề hoặc danh sách hình thức nếu câu trả lời ngắn là đủ.
- Không nêu căn cứ, điều kiện hoặc ngoại lệ không xuất hiện trong ngữ cảnh.""",
            "grounded_concise": """- Trả lời ngắn gọn, trực tiếp, ưu tiên cách diễn đạt gần với một đáp án pháp luật.
- Chỉ giữ các chi tiết cần thiết để trả lời; không diễn giải lại toàn bộ ngữ cảnh.
- Nêu số liệu, điều/khoản hoặc tên văn bản chỉ khi chúng xuất hiện trong ngữ cảnh.
- Nếu ngữ cảnh chưa đủ, chỉ trả lời phần xác nhận được, không suy đoán hoặc bổ sung kiến thức ngoài ngữ cảnh.""",
        }[prompt_variant]
    except KeyError as exc:
        allowed = ", ".join(available_prompt_variants())
        raise ValueError(
            f"Unknown prompt_variant={prompt_variant!r}; choose one of: {allowed}"
        ) from exc


def build_user_prompt(
    question: str, context: str, prompt_variant: str = "baseline"
) -> str:
    return f"""Yêu cầu:
{_variant_rules(prompt_variant)}
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


def build_messages(
    question: str, context: str, prompt_variant: str = "baseline"
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(question, context, prompt_variant),
        },
    ]

