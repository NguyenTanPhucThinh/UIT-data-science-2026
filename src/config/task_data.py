from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


@dataclass(frozen=True)
class TaskDataPaths:
    task: str
    raw_train: Path
    raw_public: Path
    clean_train: Path
    clean_public: Path
    preprocessed_train: Path
    preprocessed_public: Path


TASK_DATA: dict[str, TaskDataPaths] = {
    "1": TaskDataPaths(
        task="1",
        raw_train=RAW_DATA_DIR / "train.json",
        raw_public=RAW_DATA_DIR / "public-official.json",
        clean_train=PROCESSED_DATA_DIR / "train_clean.jsonl",
        clean_public=PROCESSED_DATA_DIR / "public_test.jsonl",
        preprocessed_train=PROCESSED_DATA_DIR / "train_preprocessed.jsonl",
        preprocessed_public=PROCESSED_DATA_DIR / "public_test_preprocessed.jsonl",
    ),
    "2": TaskDataPaths(
        task="2",
        raw_train=RAW_DATA_DIR / "train-2.json",
        raw_public=RAW_DATA_DIR / "public-official-2.json",
        clean_train=PROCESSED_DATA_DIR / "train_2_clean.jsonl",
        clean_public=PROCESSED_DATA_DIR / "public_test_2.jsonl",
        preprocessed_train=PROCESSED_DATA_DIR / "train_2_preprocessed.jsonl",
        preprocessed_public=PROCESSED_DATA_DIR / "public_test_2_preprocessed.jsonl",
    ),
}


def normalize_task(task: str | int) -> str:
    value = str(task).strip().lower().replace("task", "")
    if value not in TASK_DATA and value != "all":
        raise ValueError("task must be one of: 1, 2, all")
    return value


def selected_tasks(task: str | int) -> tuple[str, ...]:
    normalized = normalize_task(task)
    return ("1", "2") if normalized == "all" else (normalized,)


def get_task_data(task: str | int) -> TaskDataPaths:
    normalized = normalize_task(task)
    if normalized == "all":
        raise ValueError("A concrete task (1 or 2) is required here")
    return TASK_DATA[normalized]
