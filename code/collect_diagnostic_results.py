from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from project_paths import MODEL_OUTPUT_ROOT


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect diagnostic analysis summaries into CSV.")
    parser.add_argument("--root", default=str(MODEL_OUTPUT_ROOT / "diagnostics"))
    parser.add_argument("--out", default=str(MODEL_OUTPUT_ROOT / "diagnostics" / "diagnostic_summary.csv"))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = []
    for path in sorted(root.glob("*/analysis_summary.json")):
        summary = load_json(path)
        rows.append(
            {
                "name": summary.get("name", path.parent.name),
                "missing_modalities": ",".join(summary.get("missing_modalities", [])),
                "samples": summary.get("samples", ""),
                "joint_acc": summary.get("joint_acc", ""),
                "intent_acc": summary.get("intent_acc", ""),
                "scene_acc": summary.get("scene_acc", ""),
                "joint_ece": summary.get("joint_ece", ""),
                "intent_ece": summary.get("intent_ece", ""),
                "scene_ece": summary.get("scene_ece", ""),
                "avg_joint_confidence": summary.get("avg_joint_confidence", ""),
                "avg_wrong_joint_confidence": summary.get("avg_wrong_joint_confidence", ""),
            }
        )

    if not rows:
        raise SystemExit(f"No analysis_summary.json files found under: {root}")

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {output_path}")
    print(f"[rows] {len(rows)}")


if __name__ == "__main__":
    main()
