from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT


FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def metric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def find_row(rows: list[dict[str, str]], experiment: str) -> dict[str, str]:
    for row in rows:
        if row["experiment"] == experiment:
            return row
    raise KeyError(experiment)


def save_main_results() -> None:
    rows = read_rows(MODEL_OUTPUT_ROOT / "experiment_summary_mptasks.csv")
    baseline = find_row(rows, "baseline_real_scene_perceiver_io_mptasks")
    improved = find_row(rows, "improved_real_scene_anchor2_perceiver_io_mptasks")
    labels = ["Joint", "Intent", "Scene"]
    baseline_values = [metric(baseline, f"test_{name.lower()}_acc") for name in labels]
    improved_values = [metric(improved, f"test_{name.lower()}_acc") for name in labels]

    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([i - width / 2 for i in x], baseline_values, width=width, label="Baseline", color="#8da0cb")
    ax.bar([i + width / 2 for i in x], improved_values, width=width, label="Improved", color="#66c2a5")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0.9, 1.01)
    ax.set_ylabel("Accuracy")
    ax.set_title("Main Results with MediaPipe Tasks")
    ax.legend(frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "main_results_mptasks.png", dpi=180)
    plt.close(fig)


def save_single_missing() -> None:
    baseline_rows = read_rows(MODEL_OUTPUT_ROOT / "experiment_summary_missing_baseline.csv")
    improved_rows = read_rows(MODEL_OUTPUT_ROOT / "experiment_summary_missing_improved.csv")
    names = ["no_imu", "no_audio", "no_gesture", "no_text", "no_scene"]
    display = ["No IMU", "No Audio", "No Gesture", "No Text", "No Scene"]
    baseline_values = [metric(find_row(baseline_rows, name), "test_joint_acc") for name in names]
    improved_values = [metric(find_row(improved_rows, name), "test_joint_acc") for name in names]

    x = range(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.bar([i - width / 2 for i in x], baseline_values, width=width, label="Baseline", color="#fc8d62")
    ax.bar([i + width / 2 for i in x], improved_values, width=width, label="Improved", color="#66c2a5")
    ax.set_xticks(list(x), display)
    ax.set_ylim(0.35, 1.03)
    ax.set_ylabel("Joint Accuracy")
    ax.set_title("Single-Modality Missing Experiments")
    ax.legend(frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "single_modality_missing.png", dpi=180)
    plt.close(fig)


def save_noise_curves() -> None:
    baseline_rows = read_rows(MODEL_OUTPUT_ROOT / "experiment_summary_noise_baseline.csv")
    improved_rows = read_rows(MODEL_OUTPUT_ROOT / "experiment_summary_noise_improved.csv")
    modalities = ["imu", "gesture", "audio", "text", "scene"]
    levels = [20, 40, 60]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, rows, title in [
        (axes[0], baseline_rows, "Baseline"),
        (axes[1], improved_rows, "Improved"),
    ]:
        for modality in modalities:
            values = [
                metric(find_row(rows, f"{modality}_noise_{level}"), "test_joint_acc")
                for level in levels
            ]
            ax.plot(levels, values, marker="o", linewidth=2, label=modality.title())
        ax.set_title(title)
        ax.set_xlabel("Noise Level (%)")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(levels)
    axes[0].set_ylabel("Joint Accuracy")
    axes[0].set_ylim(0.70, 1.01)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle("Single-Modality Noise Robustness")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "noise_robustness.png", dpi=180)
    plt.close(fig)


def save_generalization() -> None:
    rows = read_rows(MODEL_OUTPUT_ROOT / "experiment_summary_generalization.csv")
    selected = [
        "generalization/improved_seed7_default_test",
        "generalization/improved_seed42_default_test",
        "generalization/improved_seed123_default_test",
        "generalization/improved_date_20260131_test_seed42",
        "generalization/improved_date_20260227_test_seed42",
        "generalization/improved_date_20260301_test_seed42",
        "generalization/improved_date_20260306_test_seed42",
    ]
    labels = ["Seed 7", "Seed 42", "Seed 123", "Date 0131", "Date 0227", "Date 0301", "Date 0306"]
    values = [metric(find_row(rows, name), "test_joint_acc") for name in selected]
    colors = ["#8da0cb", "#8da0cb", "#8da0cb", "#a6d854", "#a6d854", "#a6d854", "#a6d854"]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0.90, 1.00)
    ax.set_ylabel("Joint Accuracy")
    ax.set_title("Generalization Checks")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "generalization_checks.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_main_results()
    save_single_missing()
    save_noise_curves()
    save_generalization()
    print(f"[saved] {FIGURE_DIR}")


if __name__ == "__main__":
    main()
