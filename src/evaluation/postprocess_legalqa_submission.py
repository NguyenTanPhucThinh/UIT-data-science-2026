"""Remove metric-unfriendly meta preambles from a Task 2 submission."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.qa.postprocess import strip_leading_meta_prefix


def _write_json_atomic(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)


def _prefix_category(prefix: str) -> str:
    folded = prefix.casefold()
    if "ngữ cảnh" in folded:
        return "context"
    if "tài liệu" in folded:
        return "documents"
    return "provided_information"


def postprocess_submission(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    source_path = Path(input_path)
    with source_path.open("r", encoding="utf-8") as f:
        submission = json.load(f)
    if not isinstance(submission, dict):
        raise ValueError("Submission root must be a JSON object")

    changed = 0
    categories: Counter[str] = Counter()
    cleaned_submission: dict[str, dict[str, str]] = {}
    for qid, item in submission.items():
        if not isinstance(item, dict) or set(item) != {"answer"}:
            raise ValueError(
                f"question_id={qid} must contain only an 'answer' field"
            )
        answer = item["answer"]
        if not isinstance(answer, str):
            raise ValueError(f"question_id={qid} answer must be a string")
        cleaned, removed_prefix = strip_leading_meta_prefix(answer)
        if removed_prefix is not None:
            changed += 1
            categories[_prefix_category(removed_prefix)] += 1
        cleaned_submission[str(qid)] = {"answer": cleaned}

    _write_json_atomic(cleaned_submission, output_path)
    return {
        "input": str(source_path),
        "output": str(output_path),
        "total_answers": len(cleaned_submission),
        "changed_answers": changed,
        "unchanged_answers": len(cleaned_submission) - changed,
        "removed_prefix_categories": dict(sorted(categories.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw Task 2 submission JSON")
    parser.add_argument("--output", required=True, help="Cleaned Task 2 submission JSON")
    parser.add_argument("--report", default=None, help="Optional cleanup report JSON")
    args = parser.parse_args()

    report = postprocess_submission(args.input, args.output)
    if args.report:
        _write_json_atomic(report, args.report)
    print(
        "LegalQA prefix cleanup completed: "
        f"changed={report['changed_answers']}/{report['total_answers']} "
        f"output={report['output']}"
    )


if __name__ == "__main__":
    main()
