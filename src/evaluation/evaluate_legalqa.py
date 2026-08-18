from __future__ import annotations

import argparse
import json
from pathlib import Path

from .legalqa_metrics import evaluate_legalqa


def _load_answers(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    answers: dict[str, str] = {}
    for qid, item in raw.items():
        if not isinstance(item, dict) or not isinstance(item.get("answer"), str):
            raise ValueError(f"question_id={qid} must contain a string answer")
        answers[str(qid)] = item["answer"]
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 2 METEOR and ROUGE-L locally")
    parser.add_argument("--references", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    result = evaluate_legalqa(
        _load_answers(args.references), _load_answers(args.predictions)
    )
    print(f"METEOR     : {result['meteor']:.6f}")
    print(f"ROUGE-L F1 : {result['rouge_l_f1']:.6f}")
    print(f"Questions  : {result['n_questions']}")
    print(f"Missing    : {result['n_missing_predictions']}")
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
