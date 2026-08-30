"""Create a deterministic answered Task 2 validation subset for prompt experiments."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.evaluation.task_selection import require_question_task


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No records found in {path}")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} in {path} must be an object")
        if not isinstance(row.get("question_id"), (str, int)):
            raise ValueError(f"Record {index} is missing question_id")
        if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
            raise ValueError(f"Record {index} must contain string question and answer")
    return rows


def _write_jsonl(rows: list[dict], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Answered Task 2 train JSONL")
    parser.add_argument("--questions-output", required=True)
    parser.add_argument("--references-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs; otherwise preserve the previous fixed split.",
    )
    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("--size must be > 0")
    require_question_task(args.source, expected_task="2")
    destinations = [
        Path(args.questions_output),
        Path(args.references_output),
        Path(args.manifest_output),
    ]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Validation split already exists ({names}). Reuse it for a fair comparison, "
            "or pass --overwrite intentionally."
        )

    rows = _read_jsonl(args.source)
    if args.size > len(rows):
        raise ValueError(f"--size={args.size} exceeds source size={len(rows)}")
    shuffled = list(rows)
    random.Random(args.seed).shuffle(shuffled)
    selected = shuffled[: args.size]
    question_ids = [str(row["question_id"]) for row in selected]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Validation subset contains duplicate question_id values")

    _write_jsonl(selected, args.questions_output)
    references = {qid: {"answer": row["answer"]} for qid, row in zip(question_ids, selected)}
    references_path = Path(args.references_output)
    references_path.parent.mkdir(parents=True, exist_ok=True)
    references_path.write_text(
        json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "source": str(Path(args.source)),
        "size": args.size,
        "seed": args.seed,
        "question_ids": question_ids,
        "questions_output": str(Path(args.questions_output)),
        "references_output": str(references_path),
    }
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created fixed Task 2 validation split: {len(selected)} questions")


if __name__ == "__main__":
    main()
