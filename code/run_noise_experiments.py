from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT


MODALITIES = ("imu", "gesture", "audio", "text", "scene")
NOISE_LEVELS = (0.2, 0.4, 0.6)


def experiment_output_dir(args: argparse.Namespace, modality: str, level: float) -> Path:
    percent = int(round(level * 100))
    experiment_root = "raw_noise_experiments" if args.noise_space == "raw" else "noise_experiments"
    return MODEL_OUTPUT_ROOT / experiment_root / args.output_model_name / f"{modality}_noise_{percent}"


def build_command(args: argparse.Namespace, modality: str, level: float) -> list[str]:
    percent = int(round(level * 100))
    output_dir = experiment_output_dir(args, modality, level)
    command = [
        sys.executable,
        "code/train.py",
        "--model",
        args.model,
        "--input-mode",
        args.input_mode,
        "--noise-space",
        args.noise_space,
        "--gesture-representation",
        args.gesture_representation,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--noise-modality",
        modality,
        "--noise-level",
        str(level),
    ]
    if args.input_mode == "raw" and args.raw_cache_root:
        suffix = "_hand_geometry" if args.gesture_representation == "hand_geometry" else ""
        raw_cache_dir = Path(args.raw_cache_root) / f"{modality}_noise_{percent}_seed{args.noise_seed}{suffix}"
        command.extend(["--raw-cache-dir", str(raw_cache_dir)])
    if args.base_feature_dir:
        command.extend(["--base-feature-dir", args.base_feature_dir])
    command.extend(["--noise-seed", str(args.noise_seed)])
    if args.force_preprocess:
        command.append("--force-preprocess")
    if args.preprocess_dry_run:
        command.append("--preprocess-dry-run")
    if args.dataset_dir:
        command.extend(["--dataset-dir", args.dataset_dir])
    if args.hololens_dir:
        command.extend(["--hololens-dir", args.hololens_dir])
    if args.fisheye_dir:
        command.extend(["--fisheye-dir", args.fisheye_dir])
    if args.skip_test_eval:
        command.append("--skip-test-eval")
    if args.skip_feature_check:
        command.append("--skip-feature-check")
    if args.text_feature_dir:
        command.extend(["--text-feature-dir", args.text_feature_dir])
    if args.gesture_feature_dir:
        command.extend(["--gesture-feature-dir", args.gesture_feature_dir])
    if args.audio_feature_dir:
        command.extend(["--audio-feature-dir", args.audio_feature_dir])
    if args.imu_feature_dir:
        command.extend(["--imu-feature-dir", args.imu_feature_dir])
    if args.gesture_feature_dim is not None:
        command.extend(["--gesture-feature-dim", str(args.gesture_feature_dim)])
    if args.audio_feature_dim is not None:
        command.extend(["--audio-feature-dim", str(args.audio_feature_dim)])
    if args.text_feature_dim is not None:
        command.extend(["--text-feature-dim", str(args.text_feature_dim)])
    if args.imu_feature_dim is not None:
        command.extend(["--imu-feature-dim", str(args.imu_feature_dim)])
    if args.consistency_weight is not None:
        command.extend(["--consistency-weight", str(args.consistency_weight)])
    if args.consistency_mask_prob is not None:
        command.extend(["--consistency-mask-prob", str(args.consistency_mask_prob)])
    if args.consistency_noise_std is not None:
        command.extend(["--consistency-noise-std", str(args.consistency_noise_std)])
    if args.consistency_temperature is not None:
        command.extend(["--consistency-temperature", str(args.consistency_temperature)])
    if args.consistency_modalities:
        command.extend(["--consistency-modalities", *args.consistency_modalities])
    if args.margin_loss_weight is not None:
        command.extend(["--margin-loss-weight", str(args.margin_loss_weight)])
    if args.margin_value is not None:
        command.extend(["--margin-value", str(args.margin_value)])
    if args.margin_intent_confusion_weight is not None:
        command.extend(["--margin-intent-confusion-weight", str(args.margin_intent_confusion_weight)])
    if args.margin_scene_confusion_weight is not None:
        command.extend(["--margin-scene-confusion-weight", str(args.margin_scene_confusion_weight)])
    if args.missing_distill_weight is not None:
        command.extend(["--missing-distill-weight", str(args.missing_distill_weight)])
    if args.missing_distill_temperature is not None:
        command.extend(["--missing-distill-temperature", str(args.missing_distill_temperature)])
    if args.missing_distill_intent_weight is not None:
        command.extend(["--missing-distill-intent-weight", str(args.missing_distill_intent_weight)])
    if args.missing_distill_scene_weight is not None:
        command.extend(["--missing-distill-scene-weight", str(args.missing_distill_scene_weight)])
    if args.missing_distill_modalities:
        command.extend(["--missing-distill-modalities", *args.missing_distill_modalities])
    if args.no_missing_distill_force_mask:
        command.append("--no-missing-distill-force-mask")
    for modality, probability in args.missing_distill_prob:
        command.extend(["--missing-distill-prob", modality, str(probability)])
    if args.focal_loss_gamma is not None:
        command.extend(["--focal-loss-gamma", str(args.focal_loss_gamma)])
    if args.no_focal_loss_apply_aux:
        command.append("--no-focal-loss-apply-aux")
    if args.fallback_max_gate is not None:
        command.extend(["--fallback-max-gate", str(args.fallback_max_gate)])
    if args.fallback_aux_weight is not None:
        command.extend(["--fallback-aux-weight", str(args.fallback_aux_weight)])
    if args.supcon_loss_weight is not None:
        command.extend(["--supcon-loss-weight", str(args.supcon_loss_weight)])
    if args.supcon_temperature is not None:
        command.extend(["--supcon-temperature", str(args.supcon_temperature)])
    if args.supcon_target is not None:
        command.extend(["--supcon-target", args.supcon_target])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-modality noise experiments.")
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument("--input-mode", choices=("raw", "features"), default="raw")
    parser.add_argument("--noise-space", choices=("raw", "feature"))
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--gesture-representation", choices=("clip", "hand_geometry"), default="clip")
    parser.add_argument("--raw-cache-root", default=str(MODEL_OUTPUT_ROOT / "raw_feature_cache"))
    parser.add_argument(
        "--base-feature-dir",
        default=str(PROJECT_ROOT / "dataset" / "AR_Data_Process3.0" / "data_full"),
    )
    parser.add_argument("--dataset-dir")
    parser.add_argument("--hololens-dir")
    parser.add_argument("--fisheye-dir")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--preprocess-dry-run", action="store_true")
    parser.add_argument("--output-model-name", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--skip-feature-check", action="store_true")
    parser.add_argument("--text-feature-dir")
    parser.add_argument("--gesture-feature-dir")
    parser.add_argument("--audio-feature-dir")
    parser.add_argument("--imu-feature-dir")
    parser.add_argument("--gesture-feature-dim", type=int)
    parser.add_argument("--audio-feature-dim", type=int)
    parser.add_argument("--text-feature-dim", type=int)
    parser.add_argument("--imu-feature-dim", type=int)
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
    parser.add_argument("--missing-distill-prob", nargs=2, action="append", metavar=("MODALITY", "PROB"), default=[])
    parser.add_argument("--focal-loss-gamma", type=float)
    parser.add_argument("--no-focal-loss-apply-aux", action="store_true")
    parser.add_argument("--fallback-max-gate", type=float)
    parser.add_argument("--fallback-aux-weight", type=float)
    parser.add_argument("--supcon-loss-weight", type=float)
    parser.add_argument("--supcon-temperature", type=float)
    parser.add_argument("--supcon-target", choices=("joint", "intent", "scene"))
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only prints them.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip conditions with a completed metrics.json.")
    args = parser.parse_args()
    if args.noise_space is None:
        args.noise_space = "raw" if args.input_mode == "raw" else "feature"
    if args.noise_space == "raw" and args.input_mode != "raw":
        parser.error("--noise-space raw requires --input-mode raw")
    args.missing_distill_prob = [
        (modality, float(probability))
        for modality, probability in args.missing_distill_prob
    ]
    if args.output_model_name is None:
        args.output_model_name = args.model

    jobs = [(modality, level) for modality in MODALITIES for level in NOISE_LEVELS]
    print(
        f"[noise] model={args.model} output_model_name={args.output_model_name} "
        f"experiments={len(jobs)} execute={args.execute}"
    )
    for index, (modality, level) in enumerate(jobs, start=1):
        output_dir = experiment_output_dir(args, modality, level)
        if args.skip_existing and (output_dir / "metrics.json").exists():
            print(f"[{index:02d}/{len(jobs):02d}] [skip-existing] {output_dir}")
            continue
        command = build_command(args, modality, level)
        print(f"[{index:02d}/{len(jobs):02d}]", " ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
