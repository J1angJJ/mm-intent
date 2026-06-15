from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT


FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"
SUMMARY_DIR = MODEL_OUTPUT_ROOT / "summary"

GIVEN_BASELINE = "Given Baseline"
GIVEN_IMPROVED = "Given Improved Baseline"
OURS_GEOMETRY = "Ours: Hand Geometry"
OURS_FACTORIZED = "Ours: Factorized Heads"
OURS_FACTORIZED_NO_SCENE = "Ours: Factorized Heads (No Explicit Scene)"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_metrics(path: Path) -> dict:
    return load_json(path)["final_metrics"]


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches="tight")
    fig.savefig(SUMMARY_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_main_results() -> None:
    baseline = final_metrics(MODEL_OUTPUT_ROOT / "baseline_real_scene_perceiver_io_mptasks" / "metrics.json")
    improved = final_metrics(MODEL_OUTPUT_ROOT / "improved_real_scene_anchor2_perceiver_io_mptasks" / "metrics.json")
    geometry = final_metrics(MODEL_OUTPUT_ROOT / "feature_suite" / "hand_geometry" / "main" / "metrics.json")
    factorized = load_json(MODEL_OUTPUT_ROOT / "factorized_head_fusion" / "factorized_head_fusion.json")[
        "best_by_scenario"
    ]["full"]

    models = [GIVEN_BASELINE, GIVEN_IMPROVED, OURS_GEOMETRY, OURS_FACTORIZED]
    series = {
        "joint_acc": [baseline["test_joint_acc"], improved["test_joint_acc"], geometry["test_joint_acc"], factorized["joint_acc"]],
        "intent_acc": [baseline["test_intent_acc"], improved["test_intent_acc"], geometry["test_intent_acc"], factorized["intent_acc"]],
        "scene_acc": [baseline["test_scene_acc"], improved["test_scene_acc"], geometry["test_scene_acc"], factorized["scene_acc"]],
    }
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    x = np.arange(len(models))
    width = 0.23
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for index, ((label, values), color) in enumerate(zip(series.items(), colors)):
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0015,
                f"{bar.get_height():.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylim(0.90, 1.01)
    ax.set_ylabel("Accuracy")
    ax.set_title("Main Test Results")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=8, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "main_results_methods.png")


def generalization_rows() -> tuple[list[str], list[float], list[float], list[float], list[float]]:
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
    improved_values: list[float] = []
    geometry_values: list[float] = []
    factorized_values: list[float] = []
    no_scene_values: list[float] = []
    for key in keys:
        improved_values.append(
            final_metrics(MODEL_OUTPUT_ROOT / "generalization" / f"improved_{key}" / "metrics.json")["test_joint_acc"]
        )
        geometry_values.append(
            final_metrics(MODEL_OUTPUT_ROOT / "generalization" / f"hand_geometry_{key}" / "metrics.json")["test_joint_acc"]
        )
        factorized = load_json(MODEL_OUTPUT_ROOT / "factorized_generalization" / key / "factorized_head_fusion.json")[
            "best_by_scenario"
        ]
        factorized_values.append(factorized["full"]["joint_acc"])
        no_scene_values.append(factorized["no_scene"]["joint_acc"])
    return labels, improved_values, geometry_values, factorized_values, no_scene_values


def plot_generalization() -> None:
    labels, improved, geometry, factorized, no_scene = generalization_rows()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(x, improved, "o-", label=GIVEN_IMPROVED, linewidth=2)
    ax.plot(x, geometry, "o-", label=OURS_GEOMETRY, linewidth=2)
    ax.plot(x, factorized, "o-", label=OURS_FACTORIZED, linewidth=2)
    ax.plot(x, no_scene, "o--", label=OURS_FACTORIZED_NO_SCENE, linewidth=2)
    ax.axvline(2.5, color="#888", linestyle=":", alpha=0.7)
    ax.text(1, 1.001, "Random seeds", ha="center", va="top", fontsize=10, color="#555")
    ax.text(4.5, 1.001, "Date holdout", ha="center", va="top", fontsize=10, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.90, 1.005)
    ax.set_ylabel("Joint accuracy")
    ax.set_title("Generalization Analysis")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "factorized_generalization.png")


def plot_method_positioning() -> None:
    labels, improved, geometry, factorized, no_scene = generalization_rows()
    del labels
    improved_no_scene = load_json(MODEL_OUTPUT_ROOT / "diagnostics" / "teacher_improved_no_scene" / "analysis_summary.json")[
        "joint_acc"
    ]
    factorized_default = load_json(MODEL_OUTPUT_ROOT / "factorized_head_fusion" / "factorized_head_fusion.json")[
        "best_by_scenario"
    ]
    categories = ["Full", "3-seed mean", "4-date mean", "No explicit scene\n(test-time)"]
    values = {
        GIVEN_IMPROVED: [0.9838709677, float(np.mean(improved[:3])), float(np.mean(improved[3:])), improved_no_scene],
        OURS_GEOMETRY: [0.9946236559, float(np.mean(geometry[:3])), float(np.mean(geometry[3:])), np.nan],
        OURS_FACTORIZED: [
            factorized_default["full"]["joint_acc"],
            float(np.mean(factorized[:3])),
            float(np.mean(factorized[3:])),
            factorized_default["no_scene"]["joint_acc"],
        ],
    }
    x = np.arange(len(categories))
    width = 0.24
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for index, ((name, model_values), color) in enumerate(zip(values.items(), colors)):
        offsets = x + (index - 1) * width
        bars = ax.bar(offsets, np.nan_to_num(model_values, nan=0.0), width, label=name, color=color)
        for bar, value in zip(bars, model_values):
            if np.isnan(value):
                bar.set_visible(False)
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.0015,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylim(0.92, 1.005)
    ax.set_ylabel("Joint accuracy")
    ax.set_title("Method Positioning Across Evaluation Settings")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "method_positioning.png")


def main() -> None:
    plot_main_results()
    plot_generalization()
    plot_method_positioning()
    print(f"[saved] {FIGURE_DIR}")


if __name__ == "__main__":
    main()
