from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project_paths import MODEL_OUTPUT_ROOT


DEFAULT_OUTPUTS = {
    "baseline": MODEL_OUTPUT_ROOT / "baseline_real_scene_perceiver_io",
    "improved": MODEL_OUTPUT_ROOT / "improved_real_scene_anchor2_perceiver_io",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def print_metric_table(metrics: dict[str, Any]) -> None:
    final_metrics = metrics.get("final_metrics", {})
    best = metrics.get("best_checkpoint", {})
    print("[best]")
    for key in ("epoch", "val_acc", "val_loss", "selection_score"):
        if key in best:
            print(f"  {key:20s} {best[key]}")

    print("[test]")
    for key in (
        "test_joint_acc",
        "test_intent_acc",
        "test_scene_acc",
        "test_loss",
    ):
        if key in final_metrics:
            print(f"  {key:20s} {final_metrics[key]}")


def print_report(path: Path, title: str) -> None:
    if not path.exists():
        return
    print(f"[{title}]")
    print(path.read_text(encoding="utf-8").strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Report/test entry for trained MM-Intent outputs.")
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument("--output-dir")
    parser.add_argument("--show-reports", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUTS[args.model]
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file: {metrics_path}")

    print(f"[output] {output_dir}")
    metrics = load_json(metrics_path)
    print_metric_table(metrics)
    if args.show_reports:
        print_report(output_dir / "classification_report.txt", "joint_report")
        print_report(output_dir / "intent_classification_report.txt", "intent_report")
        print_report(output_dir / "scene_classification_report.txt", "scene_report")


if __name__ == "__main__":
    main()
