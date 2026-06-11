from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ARTIFACT_ROOT, require_existing


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def pick_metric(row: dict[str, str], metric: str) -> float:
    value = row.get(metric, "")
    return float(value) if value not in ("", None) else 0.0


def save_metric_bar(rows: list[dict[str, str]], metric: str, output_path: Path) -> None:
    labels = [row.get("name") or row.get("experiment") or str(index) for index, row in enumerate(rows, start=1)]
    values = [pick_metric(row, metric) for row in rows]
    fig, axis = plt.subplots(figsize=(max(7, len(labels) * 0.7), 4.8))
    axis.bar(labels, values, color="#54A24B", edgecolor="white", linewidth=0.8)
    axis.set_title(metric)
    axis.set_ylim(0, 1.05 if metric.endswith("acc") or metric.endswith("f1") or "accuracy" in metric else max(values + [1.0]) * 1.1)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize probe or experiment summary CSV files.")
    parser.add_argument("--summary", default=str(ARTIFACT_ROOT / "probes" / "summary.csv"))
    parser.add_argument("--out-dir", default=str(ARTIFACT_ROOT / "figures" / "results"))
    parser.add_argument("--metrics", nargs="*", default=["accuracy", "macro_f1", "weighted_f1", "test_joint_acc", "test_intent_acc", "test_scene_acc"])
    args = parser.parse_args()

    summary_path = Path(args.summary)
    require_existing(summary_path, "Summary CSV not found")
    rows = load_rows(summary_path)
    if not rows:
        raise SystemExit(f"Empty summary: {summary_path}")

    out_dir = Path(args.out_dir)
    available = set(rows[0].keys())
    for metric in args.metrics:
        if metric in available:
            save_metric_bar(rows, metric, out_dir / f"{metric}.png")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
