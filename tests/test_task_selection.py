import importlib.util
import json
from pathlib import Path

import pytest

from src.config.task_data import get_task_data, selected_tasks
from src.evaluation.task_selection import detect_question_task, require_question_task


def _load_train_builder():
    path = Path(__file__).resolve().parents[1] / "src" / "ingestion + EDA" / "train_builder.py"
    spec = importlib.util.spec_from_file_location("train_builder_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_task_paths_are_distinct():
    task1 = get_task_data("1")
    task2 = get_task_data("2")
    assert task1.raw_public.name == "public-official.json"
    assert task2.raw_public.name == "public-official-2.json"
    assert task1.preprocessed_public != task2.preprocessed_public
    assert selected_tasks("all") == ("1", "2")


def test_task2_builder_preserves_string_answers(tmp_path):
    train = tmp_path / "train-2.json"
    public = tmp_path / "public-official-2.json"
    train.write_text(
        json.dumps({"q1": {"question": "Hỏi?", "answer": "Trả lời."}}, ensure_ascii=False),
        encoding="utf-8",
    )
    public.write_text(
        json.dumps({"q2": {"question": "Hỏi public?", "answer": None}}, ensure_ascii=False),
        encoding="utf-8",
    )

    builder = _load_train_builder()
    builder.build_task2_questions(str(train), str(public), str(tmp_path))

    train_record = json.loads((tmp_path / "train_2_clean.jsonl").read_text(encoding="utf-8"))
    public_record = json.loads((tmp_path / "public_test_2.jsonl").read_text(encoding="utf-8"))
    assert train_record["answer"] == "Trả lời."
    assert public_record["answer"] is None


def test_legalqa_rejects_canonical_task1_questions(tmp_path):
    task1 = tmp_path / "public_test_preprocessed.jsonl"
    task1.write_text(
        json.dumps({"question_id": "1", "question": "Hỏi?", "answer_doc_ids": None}),
        encoding="utf-8",
    )
    assert detect_question_task(task1) == "1"
    with pytest.raises(ValueError, match="belongs to Task 1"):
        require_question_task(task1, expected_task="2")


def test_legalqa_accepts_canonical_task2_questions(tmp_path):
    task2 = tmp_path / "public_test_2_preprocessed.jsonl"
    task2.write_text(
        json.dumps({"question_id": "2", "question": "Hỏi?", "answer": None}),
        encoding="utf-8",
    )
    require_question_task(task2, expected_task="2")
