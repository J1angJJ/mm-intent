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


def row_from_metrics(metrics_path: Path, root: Path) -> dict[str, object]:
    metrics = load_json(metrics_path)
    final = metrics.get("final_metrics", {})
    best = metrics.get("best_checkpoint", {})
    rel_parent = metrics_path.parent.relative_to(root)
    return {
        "experiment": str(rel_parent).replace("\\", "/"),
        "best_epoch": best.get("epoch", ""),
        "best_val_acc": best.get("val_acc", ""),
        "best_val_loss": best.get("val_loss", ""),
        "test_joint_acc": final.get("test_joint_acc", ""),
        "test_intent_acc": final.get("test_intent_acc", ""),
        "test_scene_acc": final.get("test_scene_acc", ""),
        "test_loss": final.get("test_loss", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment metrics into CSV.")
    parser.add_argument("--root", default=str(MODEL_OUTPUT_ROOT))
    parser.add_argument("--out", default=str(MODEL_OUTPUT_ROOT / "experiment_summary.csv"))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = [
        row_from_metrics(path, root)
        for path in sorted(root.glob("**/metrics.json"))
    ]
    if not rows:
        raise SystemExit(f"No metrics.json files found under: {root}")

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {output_path}")
    print(f"[rows] {len(rows)}")


if __name__ == "__main__":
    main()
