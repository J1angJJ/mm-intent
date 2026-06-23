from __future__ import annotations

import csv
import itertools
import json
import shutil
from pathlib import Path
from typing import Iterable

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

MODALITIES = ("imu", "gesture", "audio", "text", "scene")
NOISE_LEVELS = (20, 40, 60)

MISSING_GROUPS = [
    tuple(group)
    for size in (1, 2)
    for group in itertools.combinations(MODALITIES, size)
]
MISSING_EXPERIMENTS = ["no_" + "_".join(group) for group in MISSING_GROUPS]
SINGLE_MISSING = ["no_" + modality for modality in MODALITIES]

MODEL_OUTPUT_NAMES = {
    BASELINE: "baseline",
    IMPROVED: "improved",
    HAND: "hand_geometry",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


METRIC_ALIASES = {
    "test_joint_acc": (
        "test_joint_acc",
        "joint_accuracy",
        "joint_acc",
    ),
    "test_intent_acc": (
        "test_intent_acc",
        "intent_accuracy",
        "intent_acc",
    ),
    "test_scene_acc": (
        "test_scene_acc",
        "scene_accuracy",
        "scene_acc",
    ),
}


def normalize_metrics(payload: dict) -> dict:
    """Normalize training and independent-test metric schemas."""
    raw = payload.get("final_metrics", payload)
    if not isinstance(raw, dict):
        raise TypeError("Metric payload is not a JSON object.")

    normalized = dict(raw)
    for canonical, aliases in METRIC_ALIASES.items():
        if canonical in normalized:
            continue
        for alias in aliases:
            if alias in raw:
                normalized[canonical] = raw[alias]
                break
    return normalized


def final_metrics(path: Path) -> dict:
    return normalize_metrics(load_json(path))


def metric_from_directory(
    directory: Path,
    metric: str,
    *,
    prefer_independent: bool = True,
) -> float:
    filenames = (
        ("independent_test_metrics.json", "metrics.json")
        if prefer_independent
        else ("metrics.json", "independent_test_metrics.json")
    )

    checked: list[str] = []
    errors: list[str] = []
    for filename in filenames:
        path = directory / filename
        checked.append(str(path))
        if not path.exists():
            continue

        metrics = final_metrics(path)
        if metric in metrics:
            return float(metrics[metric])

        available = ", ".join(sorted(metrics))
        errors.append(
            f"{metric!r} absent from {path}; available keys: {available}"
        )

    if errors:
        raise KeyError(" | ".join(errors))
    raise FileNotFoundError(
        "No usable metric file was found. Checked: " + "; ".join(checked)
    )


def main_model_directory(model_name: str) -> Path:
    formal = {
        "baseline": MODEL_OUTPUT_ROOT / "baseline_raw_end_to_end",
        "improved": MODEL_OUTPUT_ROOT / "improved_raw_end_to_end",
        "hand_geometry": MODEL_OUTPUT_ROOT / "hand_geometry_raw_end_to_end",
    }[model_name]
    if formal.exists():
        return formal

    legacy = {
        "baseline": (
            MODEL_OUTPUT_ROOT
            / "baseline_real_scene_perceiver_io_mptasks"
        ),
        "improved": (
            MODEL_OUTPUT_ROOT
            / "improved_real_scene_anchor2_perceiver_io_mptasks"
        ),
        "hand_geometry": (
            MODEL_OUTPUT_ROOT
            / "feature_suite"
            / "hand_geometry"
            / "main"
        ),
    }[model_name]
    return legacy


def factorized_full_metrics() -> dict:
    path = (
        MODEL_OUTPUT_ROOT
        / "factorized_head_fusion"
        / "factorized_head_fusion.json"
    )
    payload = load_json(path)["best_by_scenario"]["full"]
    return {
        "test_joint_acc": float(payload["joint_acc"]),
        "test_intent_acc": float(payload["intent_acc"]),
        "test_scene_acc": float(payload["scene_acc"]),
    }


def model_metrics() -> dict[str, dict]:
    result: dict[str, dict] = {}
    for display_name, model_name in MODEL_OUTPUT_NAMES.items():
        directory = main_model_directory(model_name)
        result[display_name] = {
            metric: metric_from_directory(directory, metric)
            for metric in (
                "test_joint_acc",
                "test_intent_acc",
                "test_scene_acc",
            )
        }
    result[FACTORIZED] = factorized_full_metrics()
    return result


def robustness_metric(
    model: str,
    suite: str,
    experiment: str,
    metric: str = "test_joint_acc",
) -> float:
    """Read the authoritative independent-test robustness metric."""
    roots_by_suite = {
        "missing": (
            "raw_missing_experiments",
            "missing_experiments",
        ),
        "noise": (
            "raw_noise_experiments",
            "noise_experiments",
        ),
    }
    if suite not in roots_by_suite:
        raise ValueError(f"Unsupported robustness suite: {suite}")

    errors: list[str] = []
    for root_name in roots_by_suite[suite]:
        directory = MODEL_OUTPUT_ROOT / root_name / model / experiment
        if not directory.exists():
            errors.append(f"missing directory: {directory}")
            continue
        try:
            return metric_from_directory(
                directory,
                metric,
                prefer_independent=True,
            )
        except (FileNotFoundError, KeyError) as exc:
            errors.append(str(exc))

    raise RuntimeError(
        f"Unable to load {metric} for {suite}/{model}/{experiment}. "
        + " | ".join(errors)
    )


def factorized_scenario_metrics(scenario: str) -> dict:
    candidates = (
        MODEL_OUTPUT_ROOT
        / "factorized_robustness"
        / scenario
        / "factorized_head_fusion.json",
        MODEL_OUTPUT_ROOT
        / "factorized_head_fusion"
        / "factorized_head_fusion.json",
    )

    errors: list[str] = []
    for path in candidates:
        if not path.exists():
            errors.append(f"missing file: {path}")
            continue
        payload = load_json(path).get("best_by_scenario", {})
        if scenario in payload:
            return payload[scenario]
        available = ", ".join(sorted(payload)) or "<none>"
        errors.append(
            f"{path} has scenarios [{available}], not {scenario!r}"
        )

    raise RuntimeError(
        f"Unable to load Factorized-Heads scenario {scenario!r}. "
        + " | ".join(errors)
    )


def factorized_scenario_metric(
    scenario: str,
    metric: str = "joint_acc",
) -> float:
    payload = factorized_scenario_metrics(scenario)
    if metric not in payload:
        available = ", ".join(sorted(payload))
        raise KeyError(
            f"{scenario!r} lacks {metric!r}; available: {available}"
        )
    return float(payload[metric])


def generalization_rows() -> tuple[list[str], dict[str, list[float]]]:
    labels = [
        "Seed 7",
        "Seed 42",
        "Seed 123",
        "2026-01-31",
        "2026-02-27",
        "2026-03-01",
        "2026-03-06",
    ]
    keys = [
        "seed7_default_test",
        "seed42_default_test",
        "seed123_default_test",
        "date_20260131_test_seed42",
        "date_20260227_test_seed42",
        "date_20260301_test_seed42",
        "date_20260306_test_seed42",
    ]
    values = {
        IMPROVED: [],
        HAND: [],
        FACTORIZED: [],
        FACTORIZED_NO_SCENE: [],
    }

    for key in keys:
        for display_name, prefix in (
            (IMPROVED, "improved"),
            (HAND, "hand_geometry"),
        ):
            directory = (
                MODEL_OUTPUT_ROOT
                / "generalization"
                / f"{prefix}_{key}"
            )
            values[display_name].append(
                metric_from_directory(
                    directory,
                    "test_joint_acc",
                    prefer_independent=False,
                )
            )

        factorized_path = (
            MODEL_OUTPUT_ROOT
            / "factorized_generalization"
            / key
            / "factorized_head_fusion.json"
        )
        factorized = load_json(factorized_path)["best_by_scenario"]
        values[FACTORIZED].append(
            float(factorized["full"]["joint_acc"])
        )
        values[FACTORIZED_NO_SCENE].append(
            float(factorized["no_scene"]["joint_acc"])
        )

    return labels, values



# ---------------------------------------------------------------------------
# Plot specification
#
# The following functions intentionally reproduce the 13 plot layouts used
# in docs/figures. Only the values are read from the current outputs.
# ---------------------------------------------------------------------------

EXPECTED_FIGURES = (
    "factorized_generalization.png",
    "generalization_checks.png",
    "hand_geometry_confusion_matrix.png",
    "hand_geometry_loss_curve.png",
    "hand_geometry_robustness_main.png",
    "hand_geometry_robustness_missing.png",
    "hand_geometry_robustness_noise.png",
    "main_results_hand_geometry.png",
    "main_results_methods.png",
    "main_results_mptasks.png",
    "method_positioning.png",
    "noise_robustness.png",
    "single_modality_missing.png",
)

# Same model/metric palettes used by the reference docs figures.
DOCS_BLUE = "#4C78A8"
DOCS_ORANGE = "#F58518"
DOCS_GREEN = "#54A24B"
DOCS_PURPLE = "#B279A2"
DOCS_RED = "#E45756"

MATPLOTLIB_BLUE = "#1F77B4"
MATPLOTLIB_ORANGE = "#FF7F0E"

MPTASK_BASELINE = "#8DA0CB"
MPTASK_IMPROVED = "#66C2A5"
MISSING_BASELINE = "#FC8D62"
MISSING_IMPROVED = "#66C2A5"

MODALITY_COLORS = {
    "imu": "#1F77B4",
    "gesture": "#FF7F0E",
    "audio": "#2CA02C",
    "text": "#D62728",
    "scene": "#9467BD",
}

# Pixel dimensions of the reference docs images at 200 dpi.
FIGURE_SIZES = {
    "factorized_generalization.png": (10.70, 4.59),
    "generalization_checks.png": (8.275, 4.135),
    "hand_geometry_confusion_matrix.png": (10.25, 8.83),
    "hand_geometry_loss_curve.png": (11.85, 7.33),
    "hand_geometry_robustness_main.png": (9.00, 4.32),
    "hand_geometry_robustness_missing.png": (9.00, 4.32),
    "hand_geometry_robustness_noise.png": (9.00, 4.32),
    "main_results_hand_geometry.png": (8.275, 4.32),
    "main_results_methods.png": (10.25, 4.59),
    "main_results_mptasks.png": (6.48, 3.78),
    "method_positioning.png": (9.35, 4.58),
    "noise_robustness.png": (10.80, 3.96),
    "single_modality_missing.png": (7.56, 4.05),
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
)


def _new_figure(filename: str):
    return plt.subplots(figsize=FIGURE_SIZES[filename])


def _grid_y(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.45)


def _docs_ylim(
    values: Iterable[float],
    default_lower: float,
    upper: float = 1.01,
    margin: float = 0.01,
) -> tuple[float, float]:
    finite = np.asarray(
        [float(value) for value in values if np.isfinite(value)],
        dtype=float,
    )
    if finite.size == 0:
        return default_lower, upper

    lower = min(default_lower, float(finite.min()) - margin)
    lower = max(0.0, lower)
    upper_value = max(upper, float(finite.max()) + margin)
    return lower, upper_value


def _annotate_bars(
    ax: plt.Axes,
    bars,
    values: Iterable[float],
    decimals: int,
    fontsize: int = 8,
) -> None:
    y0, y1 = ax.get_ylim()
    offset = (y1 - y0) * 0.012
    for bar, value in zip(bars, values):
        value = float(value)
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
        )


def _save_reference_style(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / filename,
        dpi=200,
        facecolor="white",
    )
    plt.close(fig)


def _clear_old_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def plot_main_results_methods() -> None:
    filename = "main_results_methods.png"
    metrics = model_metrics()
    methods = [BASELINE, IMPROVED, HAND, FACTORIZED]
    metric_names = ("joint_acc", "intent_acc", "scene_acc")
    metric_keys = (
        "test_joint_acc",
        "test_intent_acc",
        "test_scene_acc",
    )
    metric_colors = (DOCS_BLUE, DOCS_ORANGE, DOCS_GREEN)

    x = np.arange(len(methods))
    width = 0.23
    values_by_metric = [
        [metrics[method][key] for method in methods]
        for key in metric_keys
    ]
    all_values = [
        value
        for values in values_by_metric
        for value in values
    ]

    fig, ax = _new_figure(filename)
    ax.set_ylim(*_docs_ylim(all_values, 0.90, 1.01, 0.005))

    for index, (name, color, values) in enumerate(
        zip(metric_names, metric_colors, values_by_metric)
    ):
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=name,
            color=color,
        )
        _annotate_bars(ax, bars, values, decimals=4, fontsize=8)

    ax.set_title("Main Test Results")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    _grid_y(ax)
    ax.legend(loc="lower right")
    _save_reference_style(fig, filename)


def plot_main_results_hand_geometry() -> None:
    filename = "main_results_hand_geometry.png"
    metrics = model_metrics()
    methods = [BASELINE, IMPROVED, HAND]
    display_methods = [
        "Baseline",
        "Improved baseline",
        "Ours: Hand geometry",
    ]
    metric_names = ("joint_acc", "intent_acc", "scene_acc")
    metric_keys = (
        "test_joint_acc",
        "test_intent_acc",
        "test_scene_acc",
    )
    metric_colors = (DOCS_BLUE, DOCS_ORANGE, DOCS_GREEN)

    x = np.arange(len(methods))
    width = 0.24
    values_by_metric = [
        [metrics[method][key] for method in methods]
        for key in metric_keys
    ]
    all_values = [
        value
        for values in values_by_metric
        for value in values
    ]

    fig, ax = _new_figure(filename)
    ax.set_ylim(*_docs_ylim(all_values, 0.90, 1.01, 0.005))

    for index, (name, color, values) in enumerate(
        zip(metric_names, metric_colors, values_by_metric)
    ):
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=name,
            color=color,
        )
        _annotate_bars(ax, bars, values, decimals=4, fontsize=8)

    ax.set_title("Main Test Results")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(display_methods)
    _grid_y(ax)
    ax.legend(loc="lower right")
    _save_reference_style(fig, filename)


def plot_main_results_mptasks() -> None:
    filename = "main_results_mptasks.png"
    metrics = model_metrics()
    categories = ["Joint", "Intent", "Scene"]
    baseline_values = [
        metrics[BASELINE]["test_joint_acc"],
        metrics[BASELINE]["test_intent_acc"],
        metrics[BASELINE]["test_scene_acc"],
    ]
    improved_values = [
        metrics[IMPROVED]["test_joint_acc"],
        metrics[IMPROVED]["test_intent_acc"],
        metrics[IMPROVED]["test_scene_acc"],
    ]

    x = np.arange(len(categories))
    width = 0.36
    all_values = baseline_values + improved_values

    fig, ax = _new_figure(filename)
    ax.set_ylim(*_docs_ylim(all_values, 0.90, 1.01, 0.005))
    bars_a = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Baseline",
        color=MPTASK_BASELINE,
    )
    bars_b = ax.bar(
        x + width / 2,
        improved_values,
        width,
        label="Improved",
        color=MPTASK_IMPROVED,
    )
    _annotate_bars(ax, bars_a, baseline_values, decimals=3, fontsize=9)
    _annotate_bars(ax, bars_b, improved_values, decimals=3, fontsize=9)

    ax.set_title("Main Results with MediaPipe Tasks")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper left", frameon=False)
    _save_reference_style(fig, filename)


def plot_hand_geometry_robustness_main() -> None:
    filename = "hand_geometry_robustness_main.png"
    hand_value = model_metrics()[HAND]["test_joint_acc"]

    fig, ax = _new_figure(filename)
    ax.bar(
        [0],
        [hand_value],
        width=0.8,
        label="Ours: Hand geometry",
        color=MATPLOTLIB_BLUE,
    )
    ax.set_ylim(0.0, max(1.02, hand_value + 0.02))
    ax.set_title("Main robustness (joint accuracy)")
    ax.set_ylabel("Joint accuracy")
    ax.set_xticks([0])
    ax.set_xticklabels(["main"], rotation=45, ha="right")
    _grid_y(ax)
    ax.legend(loc="upper right")
    _save_reference_style(fig, filename)


def _sorted_missing_experiments() -> list[str]:
    return sorted(MISSING_EXPERIMENTS)


def _sorted_noise_experiments() -> list[str]:
    return sorted(
        f"{modality}_noise_{level}"
        for modality in MODALITIES
        for level in NOISE_LEVELS
    )


def plot_hand_geometry_robustness_missing() -> None:
    filename = "hand_geometry_robustness_missing.png"
    experiments = _sorted_missing_experiments()
    hand_values = [
        robustness_metric("hand_geometry", "missing", name)
        for name in experiments
    ]
    improved_values = [
        robustness_metric("improved", "missing", name)
        for name in experiments
    ]

    x = np.arange(len(experiments))
    width = 0.40

    fig, ax = _new_figure(filename)
    ax.bar(
        x - width / 2,
        hand_values,
        width,
        label="Ours: Hand Geometry",
        color=MATPLOTLIB_BLUE,
    )
    ax.bar(
        x + width / 2,
        improved_values,
        width,
        label="Given Improved Baseline",
        color=MATPLOTLIB_ORANGE,
    )
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Missing robustness (retrained under perturbation)")
    ax.set_ylabel("Joint accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    _grid_y(ax)
    ax.legend(loc="upper right")
    _save_reference_style(fig, filename)


def plot_hand_geometry_robustness_noise() -> None:
    filename = "hand_geometry_robustness_noise.png"
    experiments = _sorted_noise_experiments()
    hand_values = [
        robustness_metric("hand_geometry", "noise", name)
        for name in experiments
    ]
    improved_values = [
        robustness_metric("improved", "noise", name)
        for name in experiments
    ]

    x = np.arange(len(experiments))
    width = 0.40

    fig, ax = _new_figure(filename)
    ax.bar(
        x - width / 2,
        hand_values,
        width,
        label="Ours: Hand Geometry",
        color=MATPLOTLIB_BLUE,
    )
    ax.bar(
        x + width / 2,
        improved_values,
        width,
        label="Given Improved Baseline",
        color=MATPLOTLIB_ORANGE,
    )
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Noise robustness (retrained under perturbation)")
    ax.set_ylabel("Joint accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha="right")
    _grid_y(ax)
    ax.legend(loc="upper right")
    _save_reference_style(fig, filename)


def plot_noise_robustness() -> None:
    filename = "noise_robustness.png"
    levels = list(NOISE_LEVELS)
    models = (
        ("baseline", "Given Baseline"),
        ("improved", "Given Improved Baseline"),
    )

    values: dict[str, dict[str, list[float]]] = {}
    all_values: list[float] = []
    for model_name, display_name in models:
        values[display_name] = {}
        for modality in MODALITIES:
            series = [
                robustness_metric(
                    model_name,
                    "noise",
                    f"{modality}_noise_{level}",
                )
                for level in levels
            ]
            values[display_name][modality] = series
            all_values.extend(series)

    lower, upper = _docs_ylim(
        all_values,
        default_lower=0.80,
        upper=1.005,
        margin=0.012,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZES[filename],
        sharey=True,
    )
    for ax, (_, display_name) in zip(axes, models):
        for modality in MODALITIES:
            label = "IMU" if modality == "imu" else modality.title()
            ax.plot(
                levels,
                values[display_name][modality],
                marker="o",
                linewidth=2,
                label=label,
                color=MODALITY_COLORS[modality],
            )
        ax.set_title(display_name)
        ax.set_xlabel("Noise level (%)")
        ax.set_xticks(levels)
        ax.set_ylim(lower, upper)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Joint accuracy")
    axes[1].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    fig.suptitle("Single-Modality Noise Robustness (Retrained)")
    fig.tight_layout(rect=(0, 0, 0.92, 0.94))
    fig.savefig(
        FIGURE_DIR / filename,
        dpi=200,
        facecolor="white",
    )
    plt.close(fig)


def plot_single_modality_missing() -> None:
    filename = "single_modality_missing.png"
    experiments = [
        "no_imu",
        "no_audio",
        "no_gesture",
        "no_text",
        "no_scene",
    ]
    categories = [
        "No IMU",
        "No Audio",
        "No Gesture",
        "No Text",
        "No Scene",
    ]
    baseline_values = [
        robustness_metric("baseline", "missing", name)
        for name in experiments
    ]
    improved_values = [
        robustness_metric("improved", "missing", name)
        for name in experiments
    ]
    all_values = baseline_values + improved_values

    x = np.arange(len(categories))
    width = 0.36

    fig, ax = _new_figure(filename)
    lower, upper = _docs_ylim(
        all_values,
        default_lower=0.35,
        upper=1.03,
        margin=0.02,
    )
    ax.set_ylim(lower, upper)

    bars_a = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Baseline",
        color=MISSING_BASELINE,
    )
    bars_b = ax.bar(
        x + width / 2,
        improved_values,
        width,
        label="Improved",
        color=MISSING_IMPROVED,
    )
    _annotate_bars(ax, bars_a, baseline_values, decimals=3, fontsize=8)
    _annotate_bars(ax, bars_b, improved_values, decimals=3, fontsize=8)

    ax.set_title("Single-Modality Missing Experiments")
    ax.set_ylabel("Joint Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper right", frameon=False)
    _save_reference_style(fig, filename)


def plot_generalization_checks() -> None:
    filename = "generalization_checks.png"
    labels, values = generalization_rows()
    improved_values = values[IMPROVED]
    hand_values = values[HAND]
    all_values = improved_values + hand_values

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = _new_figure(filename)
    ax.set_ylim(
        *_docs_ylim(
            all_values,
            default_lower=0.89,
            upper=1.005,
            margin=0.008,
        )
    )
    bars_a = ax.bar(
        x - width / 2,
        improved_values,
        width,
        label=IMPROVED,
        color=DOCS_ORANGE,
    )
    bars_b = ax.bar(
        x + width / 2,
        hand_values,
        width,
        label=HAND,
        color=DOCS_GREEN,
    )
    _annotate_bars(ax, bars_a, improved_values, decimals=4, fontsize=8)
    _annotate_bars(ax, bars_b, hand_values, decimals=4, fontsize=8)

    ax.set_title("Generalization Checks")
    ax.set_ylabel("Joint accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _grid_y(ax)
    ax.legend(loc="upper right", frameon=False)
    _save_reference_style(fig, filename)


def plot_factorized_generalization() -> None:
    filename = "factorized_generalization.png"
    labels, values = generalization_rows()
    x = np.arange(len(labels))
    series_order = (
        IMPROVED,
        HAND,
        FACTORIZED,
        FACTORIZED_NO_SCENE,
    )
    colors = {
        IMPROVED: DOCS_ORANGE,
        HAND: DOCS_GREEN,
        FACTORIZED: DOCS_PURPLE,
        FACTORIZED_NO_SCENE: DOCS_RED,
    }
    all_values = [
        value
        for name in series_order
        for value in values[name]
    ]

    fig, ax = _new_figure(filename)
    ax.set_ylim(
        *_docs_ylim(
            all_values,
            default_lower=0.89,
            upper=1.005,
            margin=0.008,
        )
    )

    for name in series_order:
        ax.plot(
            x,
            values[name],
            marker="o",
            linewidth=2,
            linestyle="--" if name == FACTORIZED_NO_SCENE else "-",
            color=colors[name],
            label=name,
        )

    ax.axvline(2.5, color="#888888", linestyle=":", alpha=0.7)
    ax.set_title("Generalization Analysis (Current Environment)")
    ax.set_ylabel("Joint accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    _grid_y(ax)
    ax.legend(loc="lower left", frameon=False)
    _save_reference_style(fig, filename)


def plot_method_positioning() -> None:
    filename = "method_positioning.png"
    _, generalization = generalization_rows()
    metrics = model_metrics()

    categories = [
        "Full",
        "3-seed mean",
        "4-date mean",
        "No explicit scene\n(test-time)",
    ]
    improved_values = [
        metrics[IMPROVED]["test_joint_acc"],
        float(np.mean(generalization[IMPROVED][:3])),
        float(np.mean(generalization[IMPROVED][3:])),
        robustness_metric("improved", "missing", "no_scene"),
    ]
    hand_values = [
        metrics[HAND]["test_joint_acc"],
        float(np.mean(generalization[HAND][:3])),
        float(np.mean(generalization[HAND][3:])),
        np.nan,
    ]
    factorized_values = [
        metrics[FACTORIZED]["test_joint_acc"],
        float(np.mean(generalization[FACTORIZED][:3])),
        float(np.mean(generalization[FACTORIZED][3:])),
        factorized_scenario_metric("no_scene", "joint_acc"),
    ]
    all_values = [
        value
        for values in (
            improved_values,
            hand_values,
            factorized_values,
        )
        for value in values
        if np.isfinite(value)
    ]

    x = np.arange(len(categories))
    width = 0.24

    fig, ax = _new_figure(filename)
    ax.set_ylim(
        *_docs_ylim(
            all_values,
            default_lower=0.92,
            upper=1.005,
            margin=0.006,
        )
    )
    bars_a = ax.bar(
        x - width,
        improved_values,
        width,
        label=IMPROVED,
        color=DOCS_BLUE,
    )
    bars_b = ax.bar(
        x,
        np.nan_to_num(hand_values, nan=0.0),
        width,
        label=HAND,
        color=DOCS_ORANGE,
    )
    # Hide the intentionally absent Hand Geometry no-scene bar.
    bars_b[-1].set_visible(False)
    bars_c = ax.bar(
        x + width,
        factorized_values,
        width,
        label=FACTORIZED,
        color=DOCS_GREEN,
    )

    _annotate_bars(ax, bars_a, improved_values, decimals=4, fontsize=8)
    _annotate_bars(ax, bars_b[:-1], hand_values[:-1], decimals=4, fontsize=8)
    _annotate_bars(ax, bars_c, factorized_values, decimals=4, fontsize=8)

    ax.set_title("Method Positioning Across Evaluation Settings")
    ax.set_ylabel("Joint accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    _grid_y(ax)
    ax.legend(loc="lower left")
    _save_reference_style(fig, filename)


def copy_model_artifacts() -> None:
    candidates = (
        MODEL_OUTPUT_ROOT / "hand_geometry_raw_end_to_end",
        (
            MODEL_OUTPUT_ROOT
            / "feature_suite"
            / "hand_geometry"
            / "main"
        ),
    )
    artifact_map = (
        ("confusion_matrix.png", "hand_geometry_confusion_matrix.png"),
        ("loss_curve.png", "hand_geometry_loss_curve.png"),
    )
    for source_name, destination_name in artifact_map:
        for directory in candidates:
            source = directory / source_name
            if source.exists():
                shutil.copy2(source, FIGURE_DIR / destination_name)
                break
        else:
            raise FileNotFoundError(
                f"Unable to find current model artifact: {source_name}"
            )


def write_main_summary() -> None:
    metrics = model_metrics()
    path = MODEL_OUTPUT_ROOT / "main_results_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "joint_acc", "intent_acc", "scene_acc"]
        )
        for name, payload in metrics.items():
            writer.writerow(
                [
                    name,
                    f"{payload['test_joint_acc']:.8f}",
                    f"{payload['test_intent_acc']:.8f}",
                    f"{payload['test_scene_acc']:.8f}",
                ]
            )


def main() -> None:
    _clear_old_figures()

    plot_main_results_methods()
    plot_main_results_hand_geometry()
    plot_main_results_mptasks()
    plot_hand_geometry_robustness_main()
    plot_hand_geometry_robustness_missing()
    plot_hand_geometry_robustness_noise()
    plot_noise_robustness()
    plot_single_modality_missing()
    plot_generalization_checks()
    plot_factorized_generalization()
    plot_method_positioning()
    copy_model_artifacts()
    write_main_summary()

    generated = sorted(
        path.name for path in FIGURE_DIR.glob("*.png")
    )
    missing = sorted(set(EXPECTED_FIGURES) - set(generated))
    extra = sorted(set(generated) - set(EXPECTED_FIGURES))

    print(f"[saved] {FIGURE_DIR}")
    print(f"[figures] {len(generated)}")
    for name in generated:
        print(f"  {name}")

    if missing:
        raise RuntimeError(
            "Missing expected figures: " + ", ".join(missing)
        )
    if extra:
        raise RuntimeError(
            "Unexpected figures remain: " + ", ".join(extra)
        )


if __name__ == "__main__":
    main()
