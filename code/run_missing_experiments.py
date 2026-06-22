from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT
from raw_pipeline import default_raw_missing_cache_dir


MODALITIES = ("imu", "gesture", "audio", "text", "scene")
CHECKPOINT_NAMES = {
    "baseline": "baseline_real_scene_perceiver_io.pt",
    "improved": "improved_real_scene_anchor2.pt",
}


def experiment_groups(max_missing: int) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    for size in range(1, max_missing + 1):
        groups.extend(tuple(group) for group in itertools.combinations(MODALITIES, size))
    return groups


def experiment_output_dir(args: argparse.Namespace, group: Sequence[str]) -> Path:
    root_name = "raw_missing_experiments" if args.input_mode == "raw" else "missing_experiments"
    return MODEL_OUTPUT_ROOT / root_name / args.output_model_name / ("no_" + "_".join(group))


def raw_cache_dir(args: argparse.Namespace, group: Sequence[str]) -> Path:
    if args.raw_cache_root:
        suffix = "_hand_geometry" if args.gesture_representation == "hand_geometry" else ""
        return Path(args.raw_cache_root) / (
            "no_" + "_".join(group) + f"_seed{args.noise_seed}{suffix}"
        )
    return default_raw_missing_cache_dir(group, args.noise_seed, args.gesture_representation)


def clean_raw_cache_dir(args: argparse.Namespace) -> Path:
    if args.base_feature_dir:
        return Path(args.base_feature_dir)
    suffix = "_hand_geometry" if args.gesture_representation == "hand_geometry" else ""
    return MODEL_OUTPUT_ROOT / "raw_feature_cache" / f"clean_seed{args.noise_seed}{suffix}"


def append_model_options(command: list[str], args: argparse.Namespace) -> None:
    option_map = (
        ("text_feature_dir", "--text-feature-dir"),
        ("gesture_feature_dir", "--gesture-feature-dir"),
        ("audio_feature_dir", "--audio-feature-dir"),
        ("imu_feature_dir", "--imu-feature-dir"),
        ("gesture_feature_dim", "--gesture-feature-dim"),
        ("audio_feature_dim", "--audio-feature-dim"),
        ("text_feature_dim", "--text-feature-dim"),
        ("imu_feature_dim", "--imu-feature-dim"),
        ("consistency_weight", "--consistency-weight"),
        ("consistency_mask_prob", "--consistency-mask-prob"),
        ("consistency_noise_std", "--consistency-noise-std"),
        ("consistency_temperature", "--consistency-temperature"),
        ("margin_loss_weight", "--margin-loss-weight"),
        ("margin_value", "--margin-value"),
        ("margin_intent_confusion_weight", "--margin-intent-confusion-weight"),
        ("margin_scene_confusion_weight", "--margin-scene-confusion-weight"),
        ("missing_distill_weight", "--missing-distill-weight"),
        ("missing_distill_temperature", "--missing-distill-temperature"),
        ("missing_distill_intent_weight", "--missing-distill-intent-weight"),
        ("missing_distill_scene_weight", "--missing-distill-scene-weight"),
        ("focal_loss_gamma", "--focal-loss-gamma"),
        ("fallback_max_gate", "--fallback-max-gate"),
        ("fallback_aux_weight", "--fallback-aux-weight"),
        ("supcon_loss_weight", "--supcon-loss-weight"),
        ("supcon_temperature", "--supcon-temperature"),
        ("supcon_target", "--supcon-target"),
    )
    for attribute, option in option_map:
        value = getattr(args, attribute)
        if value is not None:
            command.extend([option, str(value)])

    if args.consistency_modalities:
        command.extend(["--consistency-modalities", *args.consistency_modalities])
    if args.missing_distill_modalities:
        command.extend(["--missing-distill-modalities", *args.missing_distill_modalities])
    if args.no_missing_distill_force_mask:
        command.append("--no-missing-distill-force-mask")
    for modality, probability in args.missing_distill_prob:
        command.extend(["--missing-distill-prob", modality, str(probability)])
    if args.no_focal_loss_apply_aux:
        command.append("--no-focal-loss-apply-aux")


def append_data_options(command: list[str], args: argparse.Namespace) -> None:
    for attribute, option in (
        ("dataset_dir", "--dataset-dir"),
        ("hololens_dir", "--hololens-dir"),
        ("fisheye_dir", "--fisheye-dir"),
    ):
        value = getattr(args, attribute)
        if value:
            command.extend([option, value])


def build_train_command(args: argparse.Namespace, group: Sequence[str]) -> list[str]:
    output_dir = experiment_output_dir(args, group)
    command = [
        sys.executable,
        "code/train.py",
        "--model",
        args.model,
        "--input-mode",
        args.input_mode,
        "--gesture-representation",
        args.gesture_representation,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--noise-seed",
        str(args.noise_seed),
        "--missing-modalities",
        *group,
    ]
    if args.input_mode == "raw":
        command.extend(
            [
                "--raw-cache-dir",
                str(raw_cache_dir(args, group)),
                "--base-feature-dir",
                str(clean_raw_cache_dir(args)),
            ]
        )
    if args.force_preprocess:
        command.append("--force-preprocess")
    if args.preprocess_dry_run:
        command.append("--preprocess-dry-run")
    if args.skip_test_eval or args.independent_test:
        # Independent inference below avoids the expensive duplicate test/subset analysis in train.py.
        command.append("--skip-test-eval")
    if args.skip_feature_check:
        command.append("--skip-feature-check")
    append_data_options(command, args)
    append_model_options(command, args)
    return command


def build_test_command(args: argparse.Namespace, group: Sequence[str]) -> list[str]:
    output_dir = experiment_output_dir(args, group)
    command = [
        sys.executable,
        "code/test.py",
        "--model",
        args.model,
        "--input-mode",
        args.input_mode,
        "--gesture-representation",
        args.gesture_representation,
        "--output-dir",
        str(output_dir),
        "--noise-seed",
        str(args.noise_seed),
        "--missing-modalities",
        *group,
    ]
    if args.input_mode == "raw":
        command.extend(
            [
                "--raw-cache-dir",
                str(raw_cache_dir(args, group)),
                "--base-feature-dir",
                str(clean_raw_cache_dir(args)),
            ]
        )
    elif args.processed_data_dir:
        command.extend(["--processed-data-dir", args.processed_data_dir])
    if args.force_preprocess:
        command.append("--force-preprocess")
    append_data_options(command, args)
    for attribute, option in (
        ("text_feature_dir", "--text-feature-dir"),
        ("gesture_feature_dir", "--gesture-feature-dir"),
        ("audio_feature_dir", "--audio-feature-dir"),
        ("imu_feature_dir", "--imu-feature-dir"),
        ("gesture_feature_dim", "--gesture-feature-dim"),
        ("audio_feature_dim", "--audio-feature-dim"),
        ("text_feature_dim", "--text-feature-dim"),
        ("imu_feature_dim", "--imu-feature-dim"),
    ):
        value = getattr(args, attribute)
        if value is not None:
            command.extend([option, str(value)])
    return command


def training_complete(args: argparse.Namespace, group: Sequence[str]) -> bool:
    output_dir = experiment_output_dir(args, group)
    required = (
        output_dir / "metrics.json",
        output_dir / CHECKPOINT_NAMES[args.model],
        output_dir / "scalers.pkl",
        output_dir / "label_encoder.pkl",
    )
    return all(path.exists() for path in required)


def add_advanced_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--consistency-weight", type=float)
    parser.add_argument("--consistency-mask-prob", type=float)
    parser.add_argument("--consistency-noise-std", type=float)
    parser.add_argument("--consistency-temperature", type=float)
    parser.add_argument("--consistency-modalities", nargs="*", default=[])
    parser.add_argument("--margin-loss-weight", type=float)
    parser.add_argument("--margin-value", type=float)
    parser.add_argument("--margin-intent-confusion-weight", type=float)
    parser.add_argument("--margin-scene-confusion-weight", type=float)
    parser.add_argument("--missing-distill-weight", type=float)
    parser.add_argument("--missing-distill-temperature", type=float)
    parser.add_argument("--missing-distill-intent-weight", type=float)
    parser.add_argument("--missing-distill-scene-weight", type=float)
    parser.add_argument("--missing-distill-modalities", nargs="*", default=[])
    parser.add_argument("--no-missing-distill-force-mask", action="store_true")
    parser.add_argument("--missing-distill-prob", nargs=2, action="append", default=[])
    parser.add_argument("--focal-loss-gamma", type=float)
    parser.add_argument("--no-focal-loss-apply-aux", action="store_true")
    parser.add_argument("--fallback-max-gate", type=float)
    parser.add_argument("--fallback-aux-weight", type=float)
    parser.add_argument("--supcon-loss-weight", type=float)
    parser.add_argument("--supcon-temperature", type=float)
    parser.add_argument("--supcon-target", choices=("joint", "intent", "scene"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all single- and double-modality missing experiments."
    )
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument("--output-model-name")
    parser.add_argument("--input-mode", choices=("raw", "features"), default="raw")
    parser.add_argument("--gesture-representation", choices=("clip", "hand_geometry"), default="clip")
    parser.add_argument("--raw-cache-root", default=str(MODEL_OUTPUT_ROOT / "raw_feature_cache"))
    parser.add_argument("--base-feature-dir")
    parser.add_argument("--processed-data-dir")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--hololens-dir")
    parser.add_argument("--fisheye-dir")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--max-missing", type=int, choices=(1, 2), default=2)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--preprocess-dry-run", action="store_true")
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--skip-feature-check", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--independent-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run code/test.py after each training run (default: enabled).",
    )
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--text-feature-dir")
    parser.add_argument("--gesture-feature-dir")
    parser.add_argument("--audio-feature-dir")
    parser.add_argument("--imu-feature-dir")
    parser.add_argument("--gesture-feature-dim", type=int)
    parser.add_argument("--audio-feature-dim", type=int)
    parser.add_argument("--text-feature-dim", type=int)
    parser.add_argument("--imu-feature-dim", type=int)
    add_advanced_model_arguments(parser)
    parser.add_argument("--execute", action="store_true", help="Actually run commands; default prints them.")
    args = parser.parse_args()

    args.missing_distill_prob = [
        (modality, float(probability))
        for modality, probability in args.missing_distill_prob
    ]
    if args.output_model_name is None:
        args.output_model_name = args.model
    if args.test_only and not args.independent_test:
        parser.error("--test-only requires --independent-test")
    if args.input_mode == "raw" and not clean_raw_cache_dir(args).exists():
        parser.error(f"Clean raw cache does not exist: {clean_raw_cache_dir(args)}")

    groups = experiment_groups(args.max_missing)
    print(
        f"[missing] model={args.model} output_model_name={args.output_model_name} "
        f"input_mode={args.input_mode} experiments={len(groups)} "
        f"independent_test={args.independent_test} execute={args.execute}"
    )
    for index, group in enumerate(groups, start=1):
        output_dir = experiment_output_dir(args, group)
        prefix = f"[{index:02d}/{len(groups):02d}]"

        complete = training_complete(args, group)
        if args.test_only:
            if not complete:
                message = f"{prefix} missing training artifacts: {output_dir}"
                if args.execute:
                    raise FileNotFoundError(message)
                print(message)
        elif args.skip_existing and complete:
            print(f"{prefix} [skip-existing-train] {output_dir}")
        else:
            train_command = build_train_command(args, group)
            print(prefix, "[train]", " ".join(train_command), flush=True)
            if args.execute:
                subprocess.run(train_command, cwd=PROJECT_ROOT, check=True)

        if not args.independent_test or args.preprocess_dry_run:
            continue
        independent_metrics = output_dir / "independent_test_metrics.json"
        if args.skip_existing and independent_metrics.exists():
            print(f"{prefix} [skip-existing-test] {independent_metrics}")
            continue
        test_command = build_test_command(args, group)
        print(prefix, "[test]", " ".join(test_command), flush=True)
        if args.execute:
            subprocess.run(test_command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
