from __future__ import annotations

import re
import unicodedata
from typing import Sequence


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class _IdentityStemmer:
    def stem(self, token: str) -> str:
        return token


class _NoWordNet:
    def synsets(self, _: str) -> list:
        return []


def tokenize_vietnamese(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text or "").casefold()
    return _TOKEN_RE.findall(text)


def meteor(reference: str, prediction: str) -> float:
    """METEOR with exact Vietnamese tokens, identity stemming and no English WordNet."""
    ref_tokens = tokenize_vietnamese(reference)
    pred_tokens = tokenize_vietnamese(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError as exc:
        raise RuntimeError("Install nltk to calculate Task 2 METEOR") from exc
    return float(
        meteor_score(
            [ref_tokens],
            pred_tokens,
            stemmer=_IdentityStemmer(),
            wordnet=_NoWordNet(),
        )
    )


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference: str, prediction: str) -> dict[str, float]:
    ref_tokens = tokenize_vietnamese(reference)
    pred_tokens = tokenize_vietnamese(prediction)
    if not ref_tokens or not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_legalqa(
    references: dict[str, str], predictions: dict[str, str]
) -> dict:
    details: dict[str, dict[str, float | int]] = {}
    meteor_scores: list[float] = []
    rouge_scores: list[float] = []
    missing = 0
    for qid, reference in references.items():
        prediction = predictions.get(qid, "")
        if qid not in predictions:
            missing += 1
        meteor_value = meteor(reference, prediction)
        rouge_value = rouge_l(reference, prediction)
        meteor_scores.append(meteor_value)
        rouge_scores.append(rouge_value["f1"])
        details[qid] = {
            "meteor": meteor_value,
            "rouge_l_f1": rouge_value["f1"],
            "reference_chars": len(reference),
            "prediction_chars": len(prediction),
        }
    count = len(references)
    return {
        "meteor": sum(meteor_scores) / count if count else 0.0,
        "rouge_l_f1": sum(rouge_scores) / count if count else 0.0,
        "n_questions": count,
        "n_missing_predictions": missing,
        "per_question": details,
    }
