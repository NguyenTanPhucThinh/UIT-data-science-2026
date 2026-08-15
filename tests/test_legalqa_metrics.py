import pytest

from src.evaluation.legalqa_metrics import evaluate_legalqa, meteor, rouge_l


def test_legalqa_metrics_exact_match():
    text = "Người lao động được nghỉ 05 ngày."
    # Standard METEOR keeps a small fragmentation penalty even for exact multi-token matches.
    assert meteor(text, text) == pytest.approx(0.9990234375)
    assert rouge_l(text, text)["f1"] == 1.0


def test_legalqa_metrics_disjoint_and_aggregate():
    assert meteor("một", "hai") == 0.0
    assert rouge_l("một", "hai")["f1"] == 0.0
    result = evaluate_legalqa({"1": "đúng", "2": "khác"}, {"1": "đúng"})
    assert result["n_questions"] == 2
    assert result["n_missing_predictions"] == 1
    # A single-token exact match scores 0.5 in standard METEOR; averaged with one miss -> 0.25.
    assert result["meteor"] == 0.25
