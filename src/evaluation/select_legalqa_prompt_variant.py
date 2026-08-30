"""Select a LegalQA prompt variant from local METEOR and ROUGE-L reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_named(items: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--report must be VARIANT=PATH, got {item!r}")
        variant, raw_path = item.split("=", 1)
        variant = variant.strip()
        if not variant or not raw_path.strip() or variant in result:
            raise ValueError(f"Invalid or duplicate --report entry: {item!r}")
        result[variant] = Path(raw_path)
    return result


def _load_metrics(path: Path) -> dict[str, float | int]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    required = ("meteor", "rouge_l_f1", "n_questions", "n_missing_predictions")
    if not all(key in report for key in required):
        raise ValueError(f"Invalid evaluation report: {path}")
    if int(report["n_missing_predictions"]) != 0:
        raise ValueError(f"Cannot select from incomplete predictions: {path}")
    return {key: report[key] for key in required}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, metavar="VARIANT=PATH")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument(
        "--max-rouge-drop",
        type=float,
        default=0.0,
        help="Maximum permitted ROUGE-L decrease against baseline.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_rouge_drop < 0:
        raise ValueError("--max-rouge-drop must be >= 0")

    reports = _parse_named(args.report)
    if args.baseline not in reports:
        raise ValueError(f"Baseline {args.baseline!r} has no report")
    metrics = {name: _load_metrics(path) for name, path in reports.items()}
    baseline = metrics[args.baseline]
    minimum_rouge = float(baseline["rouge_l_f1"]) - args.max_rouge_drop
    eligible = {
        name: value
        for name, value in metrics.items()
        if float(value["rouge_l_f1"]) >= minimum_rouge
    }
    if not eligible:
        eligible = {args.baseline: baseline}
    # Deterministic tie-breakers: METEOR, then ROUGE-L, then baseline, then variant name.
    best_variant = sorted(
        eligible,
        key=lambda name: (
            -float(eligible[name]["meteor"]),
            -float(eligible[name]["rouge_l_f1"]),
            name != args.baseline,
            name,
        ),
    )[0]
    result = {
        "selection_rule": {
            "primary_metric": "meteor",
            "baseline": args.baseline,
            "baseline_rouge_l_f1": baseline["rouge_l_f1"],
            "max_rouge_drop": args.max_rouge_drop,
            "minimum_eligible_rouge_l_f1": minimum_rouge,
        },
        "best_variant": best_variant,
        "best_metrics": metrics[best_variant],
        "all_metrics": metrics,
        "eligible_variants": sorted(eligible),
        "reports": {name: str(path) for name, path in reports.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "Selected prompt variant: "
        f"{best_variant} (METEOR={float(metrics[best_variant]['meteor']):.6f}, "
        f"ROUGE-L={float(metrics[best_variant]['rouge_l_f1']):.6f})"
    )


if __name__ == "__main__":
    main()
