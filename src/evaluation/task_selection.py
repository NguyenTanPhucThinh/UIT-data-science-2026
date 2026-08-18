from __future__ import annotations

import json
from pathlib import Path


_TASK1_NAMES = {
    "train.json",
    "public-official.json",
    "train_clean.jsonl",
    "val_clean.jsonl",
    "public_test.jsonl",
    "train_preprocessed.jsonl",
    "val_preprocessed.jsonl",
    "public_test_preprocessed.jsonl",
}
_TASK2_NAMES = {
    "train-2.json",
    "public-official-2.json",
    "train_2_clean.jsonl",
    "public_test_2.jsonl",
    "train_2_preprocessed.jsonl",
    "public_test_2_preprocessed.jsonl",
}


def detect_question_task(path: str | Path) -> str | None:
    """Detect canonical task files, with answer-schema fallback for custom files."""
    question_path = Path(path)
    name = question_path.name.lower()
    if name in _TASK1_NAMES:
        return "1"
    if name in _TASK2_NAMES:
        return "2"

    with question_path.open("r", encoding="utf-8") as f:
        if question_path.suffix.lower() == ".jsonl":
            items = (json.loads(line) for line in f if line.strip())
        else:
            raw = json.load(f)
            if not isinstance(raw, dict):
                return None
            items = iter(raw.values())

        for item in items:
            if not isinstance(item, dict):
                continue
            answer = item.get("answer")
            if isinstance(answer, str):
                return "2"
            if isinstance(answer, list) or "answer_doc_ids" in item:
                return "1"
    return None


def require_question_task(path: str | Path, expected_task: str) -> None:
    detected = detect_question_task(path)
    if detected is not None and detected != str(expected_task):
        raise ValueError(
            f"Question file {path} belongs to Task {detected}, but this command expects "
            f"Task {expected_task}. Choose the matching raw/preprocessed artifact."
        )
