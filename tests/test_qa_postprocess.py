from src.qa.postprocess import (
    clean_generated_answer,
    strip_leading_meta_prefix,
    unsupported_legal_markers,
)


def test_clean_generated_answer_removes_reasoning_wrappers():
    raw = "<think>bí mật</think><answer>### Trả lời: Theo Điều 5.</answer>"
    assert clean_generated_answer(raw) == "Theo Điều 5."


def test_unsupported_legal_markers_detects_only_missing_values():
    answer = "Theo Điều 5, thời hạn là 30 ngày và mức là 20%."
    context = "Điều 5 quy định thời hạn 30 ngày."
    markers = unsupported_legal_markers(answer, context)
    assert "20%" in markers
    assert "Điều 5" not in markers
    assert "30 ngày" not in markers


def test_unsupported_legal_markers_accepts_heading_split_across_lines():
    answer = "Căn cứ Điều 86 và Điều 87."
    context = "Điều\n\n86. Quyền của người đại diện\nĐiều\r\n87. Trường hợp khác"
    assert unsupported_legal_markers(answer, context) == []


def test_strip_leading_meta_prefix_removes_only_supplied_context_preamble():
    cleaned, prefix = strip_leading_meta_prefix(
        "Dựa trên các tài liệu được cung cấp, người lao động được nghỉ 05 ngày."
    )
    assert cleaned == "Người lao động được nghỉ 05 ngày."
    assert prefix == "Dựa trên các tài liệu được cung cấp,"


def test_strip_leading_meta_prefix_removes_answer_bridge():
    cleaned, prefix = strip_leading_meta_prefix(
        "Dựa trên thông tin từ ngữ cảnh 1 và ngữ cảnh 2, câu trả lời là: thời hạn là 30 ngày."
    )
    assert cleaned == "Thời hạn là 30 ngày."
    assert prefix is not None


def test_strip_leading_meta_prefix_handles_context_header_with_commas():
    cleaned, prefix = strip_leading_meta_prefix(
        "Dựa trên ngữ cảnh 4 (Tài liệu 4, doc_id=10, mục=Điều 5), câu trả lời là: người lao động được nghỉ."
    )
    assert cleaned == "Người lao động được nghỉ."
    assert prefix is not None


def test_strip_leading_meta_prefix_preserves_substantive_legal_premise():
    answer = "Dựa trên quy định tại Điều 5, thời hạn là 30 ngày."
    assert strip_leading_meta_prefix(answer) == (answer, None)


def test_clean_generated_answer_also_removes_meta_prefix():
    raw = "<answer>Dựa trên ngữ cảnh được cung cấp, người sử dụng lao động phải đóng bảo hiểm.</answer>"
    assert clean_generated_answer(raw) == "Người sử dụng lao động phải đóng bảo hiểm."
