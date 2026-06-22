from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT


FIGURE_DIR = PROJECT_ROOT / "docs_wjy" / "figures"

BASELINE = "Given Baseline"
IMPROVED = "Given Improved Baseline"
HAND = "Ours: Hand Geometry"
FACTORIZED = "Ours: Factorized Heads"
FACTORIZED_NO_SCENE = "Factorized Heads (No Explicit Scene)"

COLORS = {
    BASELINE: "#4C78A8",
    IMPROVED: "#F58518",
    HAND: "#54A24B",
    FACTORIZED: "#B279A2",
    FACTORIZED_NO_SCENE: "#E45756",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_metrics(path: Path) -> dict:
    return load_json(path)["final_metrics"]


def model_metrics() -> dict[str, dict]:
    factorized = load_json(MODEL_OUTPUT_ROOT / "factorized_head_fusion" / "factorized_head_fusion.json")[
        "best_by_scenario"
    ]["full"]
    return {
        BASELINE: final_metrics(MODEL_OUTPUT_ROOT / "baseline_real_scene_perceiver_io_mptasks" / "metrics.json"),
        IMPROVED: final_metrics(MODEL_OUTPUT_ROOT / "improved_real_scene_anchor2_perceiver_io_mptasks" / "metrics.json"),
        HAND: final_metrics(MODEL_OUTPUT_ROOT / "feature_suite" / "hand_geometry" / "main" / "metrics.json"),
        FACTORIZED: {
            "test_joint_acc": factorized["joint_acc"],
            "test_intent_acc": factorized["intent_acc"],
            "test_scene_acc": factorized["scene_acc"],
        },
    }


def robustness_metric(model: str, suite: str, experiment: str, metric: str = "test_joint_acc") -> float:
    path = MODEL_OUTPUT_ROOT / f"{suite}_experiments" / model / experiment / "metrics.json"
    return float(final_metrics(path)[metric])


def factorized_robustness(experiment: str) -> dict:
    return load_json(MODEL_OUTPUT_ROOT / "factorized_robustness" / experiment / "factorized_head_fusion.json")[
        "best_by_scenario"
    ][experiment]


def generalization_rows() -> tuple[list[str], dict[str, list[float]]]:
    labels = ["Seed 7", "Seed 42", "Seed 123", "2026-01-31", "2026-02-27", "2026-03-01", "2026-03-06"]
    keys = [
        "seed7_default_test",
        "seed42_default_test",
        "seed123_default_test",
        "date_20260131_test_seed42",
        "date_20260227_test_seed42",
        "date_20260301_test_seed42",
        "date_20260306_test_seed42",
    ]
    values = {IMPROVED: [], HAND: [], FACTORIZED: [], FACTORIZED_NO_SCENE: []}
    for key in keys:
        values[IMPROVED].append(
            float(
                final_metrics(MODEL_OUTPUT_ROOT / "generalization" / f"improved_{key}" / "metrics.json")[
                    "test_joint_acc"
                ]
            )
        )
        values[HAND].append(
            float(
                final_metrics(MODEL_OUTPUT_ROOT / "generalization" / f"hand_geometry_{key}" / "metrics.json")[
                    "test_joint_acc"
                ]
            )
        )
        factorized = load_json(
            MODEL_OUTPUT_ROOT / "factorized_generalization" / key / "factorized_head_fusion.json"
        )["best_by_scenario"]
        values[FACTORIZED].append(float(factorized["full"]["joint_acc"]))
        values[FACTORIZED_NO_SCENE].append(float(factorized["no_scene"]["joint_acc"]))
    return labels, values


def save(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(
    filename: str,
    title: str,
    categories: list[str],
    series: dict[str, list[float]],
    ylim: tuple[float, float],
    ylabel: str = "Accuracy",
    rotate: int = 0,
) -> None:
    x = np.arange(len(categories))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(8.0, len(categories) * 1.55), 4.9))
    for index, (name, values) in enumerate(series.items()):
        offsets = x + (index - (len(series) - 1) / 2) * width
        bars = ax.bar(offsets, values, width=width, label=name, color=COLORS.get(name))
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (ylim[1] - ylim[0]) * 0.012,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=rotate, ha="right" if rotate else "center")
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, filename)


def plot_main_results() -> None:
    metrics = model_metrics()
    categories = list(metrics)
    series = {
        "Joint": [metrics[name]["test_joint_acc"] for name in categories],
        "Intent": [metrics[name]["test_intent_acc"] for name in categories],
        "Scene": [metrics[name]["test_scene_acc"] for name in categories],
    }
    # Use metric-specific colors because the bars encode metrics, not models.
    x = np.arange(len(categories))
    width = 0.23
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for index, ((name, values), color) in enumerate(zip(series.items(), ["#4C78A8", "#F58518", "#54A24B"])):
        bars = ax.bar(x + (index - 1) * width, values, width, label=name, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}", ha="center", fontsize=8)
    ax.set_ylim(0.95, 1.005)
    ax.set_ylabel("Accuracy")
    ax.set_title("Main Test Results (Current Environment)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=8, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "main_results_methods.png")


def plot_pairwise_main() -> None:
    metrics = model_metrics()
    labels = ["Joint", "Intent", "Scene"]
    grouped_bars(
        "main_results_mptasks.png",
        "Given Baseline vs Given Improved Baseline",
        labels,
        {
            BASELINE: [metrics[BASELINE][f"test_{name.lower()}_acc"] for name in labels],
            IMPROVED: [metrics[IMPROVED][f"test_{name.lower()}_acc"] for name in labels],
        },
        (0.94, 1.005),
    )
    grouped_bars(
        "main_results_hand_geometry.png",
        "Given Improved Baseline vs Hand Geometry",
        labels,
        {
            IMPROVED: [metrics[IMPROVED][f"test_{name.lower()}_acc"] for name in labels],
            HAND: [metrics[HAND][f"test_{name.lower()}_acc"] for name in labels],
        },
        (0.96, 1.005),
    )
    grouped_bars(
        "hand_geometry_robustness_main.png",
        "Full-Modality Main Results",
        labels,
        {
            IMPROVED: [metrics[IMPROVED][f"test_{name.lower()}_acc"] for name in labels],
            HAND: [metrics[HAND][f"test_{name.lower()}_acc"] for name in labels],
        },
        (0.96, 1.005),
    )


def plot_hand_robustness() -> None:
    missing = ["no_text", "no_audio_text", "no_imu_text", "no_gesture_text", "no_scene"]
    grouped_bars(
        "hand_geometry_robustness_missing.png",
        "Missing-Modality Robustness (Retrained per Condition)",
        missing,
        {
            IMPROVED: [robustness_metric("improved", "missing", name) for name in missing],
            HAND: [robustness_metric("hand_geometry", "missing", name) for name in missing],
        },
        (0.35, 1.02),
        ylabel="Joint accuracy",
        rotate=20,
    )

    noise = ["gesture_noise_60", "text_noise_60", "audio_noise_60", "imu_noise_60", "scene_noise_60"]
    grouped_bars(
        "hand_geometry_robustness_noise.png",
        "60% Modality-Noise Robustness (Retrained per Condition)",
        noise,
        {
            IMPROVED: [robustness_metric("improved", "noise", name) for name in noise],
            HAND: [robustness_metric("hand_geometry", "noise", name) for name in noise],
        },
        (0.94, 1.005),
        ylabel="Joint accuracy",
        rotate=20,
    )


def plot_baseline_robustness() -> None:
    missing = ["no_imu", "no_audio", "no_gesture", "no_text", "no_scene"]
    grouped_bars(
        "single_modality_missing.png",
        "Single-Modality Missing Experiments (Retrained)",
        ["No IMU", "No Audio", "No Gesture", "No Text", "No Scene"],
        {
            BASELINE: [robustness_metric("baseline", "missing", name) for name in missing],
            IMPROVED: [robustness_metric("improved", "missing", name) for name in missing],
        },
        (0.20, 1.02),
        ylabel="Joint accuracy",
        rotate=10,
    )

    modalities = ["imu", "gesture", "audio", "text", "scene"]
    levels = [20, 40, 60]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    for ax, model, title in [(axes[0], "baseline", BASELINE), (axes[1], "improved", IMPROVED)]:
        for modality in modalities:
            values = [robustness_metric(model, "noise", f"{modality}_noise_{level}") for level in levels]
            ax.plot(levels, values, marker="o", linewidth=2, label=modality.title())
        ax.set_title(title)
        ax.set_xlabel("Noise level (%)")
        ax.set_xticks(levels)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Joint accuracy")
    axes[0].set_ylim(0.80, 1.005)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle("Single-Modality Noise Robustness (Retrained)")
    fig.tight_layout()
    save(fig, "noise_robustness.png")


def plot_generalization() -> None:
    labels, values = generalization_rows()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for name in [IMPROVED, HAND, FACTORIZED, FACTORIZED_NO_SCENE]:
        style = "o--" if name == FACTORIZED_NO_SCENE else "o-"
        ax.plot(x, values[name], style, linewidth=2, label=name, color=COLORS[name])
    ax.axvline(2.5, color="#888", linestyle=":", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.89, 1.005)
    ax.set_ylabel("Joint accuracy")
    ax.set_title("Generalization Analysis (Current Environment)")
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "factorized_generalization.png")

    grouped_bars(
        "generalization_checks.png",
        "Generalization Checks",
        labels,
        {IMPROVED: values[IMPROVED], HAND: values[HAND]},
        (0.89, 1.005),
        ylabel="Joint accuracy",
        rotate=20,
    )


def plot_method_positioning() -> None:
    _, values = generalization_rows()
    metrics = model_metrics()
    factorized_default = load_json(MODEL_OUTPUT_ROOT / "factorized_head_fusion" / "factorized_head_fusion.json")[
        "best_by_scenario"
    ]
    categories = ["Full", "3-seed mean", "4-date mean", "No explicit scene"]
    series = {
        IMPROVED: [
            metrics[IMPROVED]["test_joint_acc"],
            float(np.mean(values[IMPROVED][:3])),
            float(np.mean(values[IMPROVED][3:])),
            robustness_metric("improved", "missing", "no_scene"),
        ],
        HAND: [
            metrics[HAND]["test_joint_acc"],
            float(np.mean(values[HAND][:3])),
            float(np.mean(values[HAND][3:])),
            robustness_metric("hand_geometry", "missing", "no_scene"),
        ],
        FACTORIZED: [
            metrics[FACTORIZED]["test_joint_acc"],
            float(np.mean(values[FACTORIZED][:3])),
            float(np.mean(values[FACTORIZED][3:])),
            factorized_default["no_scene"]["joint_acc"],
        ],
    }
    grouped_bars(
        "method_positioning.png",
        "Method Positioning Across Evaluation Settings",
        categories,
        series,
        (0.78, 1.005),
        ylabel="Joint accuracy",
        rotate=8,
    )


def copy_model_artifacts() -> None:
    source = MODEL_OUTPUT_ROOT / "feature_suite" / "hand_geometry" / "main"
    shutil.copy2(source / "confusion_matrix.png", FIGURE_DIR / "hand_geometry_confusion_matrix.png")
    shutil.copy2(source / "loss_curve.png", FIGURE_DIR / "hand_geometry_loss_curve.png")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_main_results()
    plot_pairwise_main()
    plot_hand_robustness()
    plot_baseline_robustness()
    plot_generalization()
    plot_method_positioning()
    copy_model_artifacts()
    generated = sorted(path.name for path in FIGURE_DIR.glob("*.png"))
    print(f"[saved] {FIGURE_DIR}")
    print(f"[figures] {len(generated)}")
    for name in generated:
        print(f"  {name}")


if __name__ == "__main__":
    main()
