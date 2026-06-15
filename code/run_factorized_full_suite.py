from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

from project_paths import MODEL_OUTPUT_ROOT, PROCESSED_DATA_DIR, PROJECT_ROOT
from run_generalization_experiments import DATE_GROUPS, default_test_video_names, video_names_for_prefix


MODALITIES = ("imu", "gesture", "audio", "text", "scene")
NOISE_LEVELS = (0.2, 0.4, 0.6)


def tee_process(command: list[str], log_path: Path) -> None:
    print("[run]", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def run_or_print(name: str, command: list[str], log_path: Path, execute: bool) -> None:
    print(f"[job] {name}")
    print(" ".join(command))
    print(f"log: {log_path}")
    if execute:
        tee_process(command, log_path)


def evaluator_base(args: argparse.Namespace, geometry_output: Path, scene_output: Path, analysis_dir: Path) -> list[str]:
    return [
        sys.executable,
        "code/evaluate_factorized_head_fusion.py",
        "--geometry-output-dir",
        str(geometry_output),
        "--scene-output-dir",
        str(scene_output),
        "--geometry-feature-dir",
        str(Path(args.geometry_feature_dir).resolve()),
        "--scene-gesture-feature-dir",
        str(Path(args.scene_gesture_feature_dir).resolve()),
        "--geometry-feature-dim",
        "96",
        "--scene-gesture-feature-dim",
        "768",
        "--analysis-dir",
        str(analysis_dir),
    ]


def generalization_specs(seeds: list[int]) -> list[tuple[str, int, list[str]]]:
    specs: list[tuple[str, int, list[str]]] = []
    default_test = default_test_video_names()
    for seed in seeds:
        specs.append((f"seed{seed}_default_test", seed, default_test))
    for group_name, prefix in DATE_GROUPS.items():
        specs.append((f"{group_name}_test_seed42", 42, video_names_for_prefix(prefix)))
    return specs


def write_summary() -> None:
    rows: list[dict[str, object]] = []
    for suite, root in (
        ("robustness", MODEL_OUTPUT_ROOT / "factorized_robustness"),
        ("generalization", MODEL_OUTPUT_ROOT / "factorized_generalization"),
    ):
        for path in sorted(root.glob("*/factorized_head_fusion.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for scenario, metrics in payload.get("best_by_scenario", {}).items():
                rows.append(
                    {
                        "suite": suite,
                        "experiment": path.parent.name,
                        "scenario": scenario,
                        "intent_weight": metrics["intent_weight"],
                        "scene_weight": metrics["scene_weight"],
                        "joint_acc": metrics["joint_acc"],
                        "intent_acc": metrics["intent_acc"],
                        "scene_acc": metrics["scene_acc"],
                    }
                )
    if not rows:
        return
    output_path = MODEL_OUTPUT_ROOT / "factorized_full_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] rows={len(rows)} path={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full factorized-head robustness and generalization suite.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument(
        "--geometry-feature-dir",
        default=str(PROCESSED_DATA_DIR / "hand_geometry_features"),
    )
    parser.add_argument(
        "--scene-gesture-feature-dir",
        default=str(PROCESSED_DATA_DIR / "strong_gesture_features"),
    )
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-generalization-training", action="store_true")
    parser.add_argument("--skip-generalization-eval", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    jobs: list[tuple[str, list[str], Path]] = []
    geometry_main = MODEL_OUTPUT_ROOT / "feature_suite" / "hand_geometry" / "main"
    improved_main = MODEL_OUTPUT_ROOT / "improved_real_scene_anchor2_perceiver_io_mptasks"

    if not args.skip_robustness:
        robustness_root = MODEL_OUTPUT_ROOT / "factorized_robustness"
        missing_groups = [
            group
            for size in (1, 2)
            for group in itertools.combinations(MODALITIES, size)
        ]
        scenario_specs: list[tuple[str, tuple[str, ...], str, float]] = [("full", (), "", 0.0)]
        scenario_specs.extend(("no_" + "_".join(group), group, "", 0.0) for group in missing_groups)
        scenario_specs.extend(
            (f"{modality}_noise_{int(level * 100)}", (), modality, level)
            for modality in MODALITIES
            for level in NOISE_LEVELS
        )
        for scenario, missing, noise_modality, noise_level in scenario_specs:
            analysis_dir = robustness_root / scenario
            command = evaluator_base(args, geometry_main, improved_main, analysis_dir)
            command.extend(["--scenario-name", scenario])
            if missing:
                command.extend(["--missing-modalities", *missing])
            if noise_modality:
                command.extend(["--noise-modality", noise_modality, "--noise-level", str(noise_level)])
            jobs.append(
                (
                    f"robustness_{scenario}",
                    command,
                    PROJECT_ROOT / "logs_factorized_full" / f"robustness_{scenario}.txt",
                )
            )

    specs = generalization_specs(args.seeds)
    if not args.skip_generalization_training:
        geometry_train = [
            sys.executable,
            "code/run_generalization_experiments.py",
            "--model",
            "improved",
            "--output-model-name",
            "hand_geometry",
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--gesture-feature-dir",
            str(Path(args.geometry_feature_dir).resolve()),
            "--gesture-feature-dim",
            "96",
            "--skip-existing",
            "--execute",
        ]
        improved_train = [
            sys.executable,
            "code/run_generalization_experiments.py",
            "--model",
            "improved",
            "--output-model-name",
            "improved",
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--skip-existing",
            "--execute",
        ]
        jobs.append(
            (
                "train_generalization_geometry",
                geometry_train,
                PROJECT_ROOT / "logs_factorized_full" / "train_generalization_geometry.txt",
            )
        )
        jobs.append(
            (
                "ensure_generalization_improved",
                improved_train,
                PROJECT_ROOT / "logs_factorized_full" / "train_generalization_improved.txt",
            )
        )

    if not args.skip_generalization_eval:
        for suffix, _seed, test_videos in specs:
            geometry_output = MODEL_OUTPUT_ROOT / "generalization" / f"hand_geometry_{suffix}"
            scene_output = MODEL_OUTPUT_ROOT / "generalization" / f"improved_{suffix}"
            analysis_dir = MODEL_OUTPUT_ROOT / "factorized_generalization" / suffix
            command = evaluator_base(args, geometry_output, scene_output, analysis_dir)
            command.extend(["--scenarios", "full", "no_scene", "--test-video-names", *test_videos])
            jobs.append(
                (
                    f"generalization_{suffix}",
                    command,
                    PROJECT_ROOT / "logs_factorized_full" / f"generalization_{suffix}.txt",
                )
            )

    print(f"[factorized-full-suite] jobs={len(jobs)} execute={args.execute}")
    for index, (name, command, log_path) in enumerate(jobs, start=1):
        print(f"[{index:02d}/{len(jobs):02d}]")
        run_or_print(name, command, log_path, args.execute)

    if args.execute:
        write_summary()
    else:
        print("[dry-run] add --execute to run the full suite")


if __name__ == "__main__":
    main()
