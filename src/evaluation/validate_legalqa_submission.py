from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


_FORBIDDEN_OUTPUT_RE = re.compile(r"<(?:think|answer)\b|```", re.IGNORECASE)


def validate_legalqa_submission(
    predictions_path: str | Path, questions_path: Optional[str | Path] = None
) -> list[str]:
    errors: list[str] = []
    try:
        with Path(predictions_path).open("r", encoding="utf-8") as f:
            predictions = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Cannot read valid JSON: {exc}"]
    if not isinstance(predictions, dict):
        return ["Submission root must be a JSON object"]

    for qid, item in predictions.items():
        if not isinstance(item, dict):
            errors.append(f"[{qid}] value must be an object")
            continue
        if set(item) != {"answer"}:
            errors.append(f"[{qid}] item must contain only the 'answer' field")
            continue
        answer = item["answer"]
        if not isinstance(answer, str):
            errors.append(f"[{qid}] answer must be a string")
        elif not answer.strip():
            errors.append(f"[{qid}] answer must not be empty")
        elif _FORBIDDEN_OUTPUT_RE.search(answer):
            errors.append(f"[{qid}] answer contains reasoning/code wrappers")

    if questions_path:
        with Path(questions_path).open("r", encoding="utf-8") as f:
            raw_questions = json.load(f)
        if not isinstance(raw_questions, dict):
            errors.append("Questions reference must be a JSON object")
        else:
            expected = {str(qid) for qid in raw_questions}
            actual = {str(qid) for qid in predictions}
            missing = expected - actual
            extra = actual - expected
            if missing:
                errors.append(f"Missing {len(missing)} question IDs; sample={sorted(missing)[:10]}")
            if extra:
                errors.append(f"Found {len(extra)} unexpected question IDs; sample={sorted(extra)[:10]}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Task 2 submission JSON")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--questions", default=None)
    args = parser.parse_args()
    errors = validate_legalqa_submission(args.predictions, args.questions)
    if errors:
        print(f"Invalid Task 2 submission ({len(errors)} errors):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print("Task 2 submission is valid")


if __name__ == "__main__":
    main()
