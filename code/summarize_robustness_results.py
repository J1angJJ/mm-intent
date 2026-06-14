from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT


MetricRow = dict[str, str | float]


def load_metric(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_row(model_name: str, suite: str, experiment: str, metrics_path: Path) -> MetricRow:
    metrics = load_metric(metrics_path)
    final = metrics["final_metrics"]
    best = metrics.get("best_checkpoint", {})
    return {
        "model": model_name,
        "suite": suite,
        "experiment": experiment,
        "best_epoch": float(best.get("epoch", -1)),
        "val_acc": float(best.get("val_acc", 0.0)),
        "joint_acc": float(final["test_joint_acc"]),
        "intent_acc": float(final["test_intent_acc"]),
        "scene_acc": float(final["test_scene_acc"]),
    }


def collect_model_rows(model_name: str) -> list[MetricRow]:
    rows: list[MetricRow] = []

    main_path = MODEL_OUTPUT_ROOT / "feature_suite" / model_name / "main" / "metrics.json"
    if main_path.exists():
        rows.append(final_row(model_name, "main", "main", main_path))

    for suite, root_name in (
        ("missing", "missing_experiments"),
        ("noise", "noise_experiments"),
    ):
        root = MODEL_OUTPUT_ROOT / root_name / model_name
        for metrics_path in sorted(root.glob("*/metrics.json")):
            rows.append(final_row(model_name, suite, metrics_path.parent.name, metrics_path))

    return rows


def write_csv(rows: Iterable[MetricRow], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "suite", "experiment", "best_epoch", "val_acc", "joint_acc", "intent_acc", "scene_acc"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[MetricRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Robustness Summary",
        "",
        "| model | suite | experiment | joint | intent | scene |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {suite} | {experiment} | {joint_acc:.4f} | {intent_acc:.4f} | {scene_acc:.4f} |".format(
                **row
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_suite(rows: list[MetricRow], suite: str, output_path: Path) -> None:
    suite_rows = [row for row in rows if row["suite"] == suite]
    if not suite_rows:
        return

    experiments = sorted({str(row["experiment"]) for row in suite_rows})
    models = sorted({str(row["model"]) for row in suite_rows})
    x_positions = list(range(len(experiments)))
    width = 0.8 / max(len(models), 1)

    fig_width = max(10, len(experiments) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    for idx, model in enumerate(models):
        values = []
        by_exp = {str(row["experiment"]): row for row in suite_rows if row["model"] == model}
        for experiment in experiments:
            values.append(float(by_exp.get(experiment, {}).get("joint_acc", 0.0)))
        offsets = [x + (idx - (len(models) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=model)

    ax.set_title(f"{suite.title()} robustness (joint accuracy)")
    ax.set_ylabel("Joint accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize robustness metrics into CSV/Markdown/PNG.")
    parser.add_argument("--models", nargs="+", default=["improved", "hand_geometry"])
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "summary"))
    args = parser.parse_args()

    rows: list[MetricRow] = []
    for model_name in args.models:
        rows.extend(collect_model_rows(model_name))

    output_dir = Path(args.output_dir)
    write_csv(rows, output_dir / "robustness_summary.csv")
    write_markdown(rows, output_dir / "robustness_summary.md")
    plot_suite(rows, "main", output_dir / "robustness_main.png")
    plot_suite(rows, "missing", output_dir / "robustness_missing.png")
    plot_suite(rows, "noise", output_dir / "robustness_noise.png")
    print(f"[summary] rows={len(rows)} output_dir={output_dir}")


if __name__ == "__main__":
    main()
