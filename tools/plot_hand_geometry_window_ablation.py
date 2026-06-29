from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "outputs" / "hand_geometry_window_ablation" / "window_ablation_summary.csv"
FALLBACK_SUMMARY = (
    ROOT
    / ".tmp_review"
    / "outputs"
    / "hand_geometry_window_ablation"
    / "window_ablation_summary.csv"
)
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "hand_geometry_window_ablation.png"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Hand Geometry temporal window ablation results.")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    summary_path = Path(args.summary) if args.summary else DEFAULT_SUMMARY
    if not summary_path.exists() and FALLBACK_SUMMARY.exists():
        summary_path = FALLBACK_SUMMARY
    if not summary_path.exists():
        raise SystemExit(f"Summary CSV not found: {summary_path}")

    rows = load_rows(summary_path)
    labels = [row["spec"].replace("_", "\n") for row in rows]
    joint = [float(row["test_joint_acc"]) for row in rows]
    intent = [float(row["test_intent_acc"]) for row in rows]
    scene = [float(row["test_scene_acc"]) for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = range(len(rows))
    width = 0.24
    colors = {
        "joint": "#2f80ed",
        "intent": "#f2994a",
        "scene": "#219653",
    }
    ax.bar([i - width for i in x], joint, width=width, label="joint_acc", color=colors["joint"])
    ax.bar(list(x), intent, width=width, label="intent_acc", color=colors["intent"])
    ax.bar([i + width for i in x], scene, width=width, label="scene_acc", color=colors["scene"])

    for i, value in enumerate(joint):
        ax.text(i - width, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0.975, 1.002)
    ax.set_ylabel("Accuracy")
    ax.set_title("Hand Geometry Temporal Window Ablation", fontsize=13, pad=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
