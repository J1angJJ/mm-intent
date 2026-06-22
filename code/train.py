from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from project_paths import PROCESSED_DATA_DIR, PROJECT_ROOT


FEATURE_PATTERNS = {
    "timestamp": "features_timestamp_*.npy",
    "gesture": "strong_gesture_features/strong_gesture_features_*.npy",
    "audio": "audio_features/audio_features_*.npy",
    "text": "text_features/text_features_*.npy",
    "imu": "imu_features/imu_features_*.npy",
}


def count_pattern(pattern: str) -> int:
    return len(list(PROCESSED_DATA_DIR.glob(pattern)))


def check_features(expected_count: int) -> bool:
    print("[feature-check]")
    ok = True
    for name, pattern in FEATURE_PATTERNS.items():
        count = count_pattern(pattern)
        print(f"  {name:9s} {count}/{expected_count}")
        ok = ok and count >= expected_count
    return ok


def feature_commands() -> Sequence[Sequence[str]]:
    return (
        (sys.executable, "code/feature_extraction/get_timestamp.py"),
        (sys.executable, "code/feature_extraction/strong_gesture2.0.py"),
        (sys.executable, "code/feature_extraction/mfcc.py"),
        (sys.executable, "code/feature_extraction/ASR.py"),
        (sys.executable, "code/feature_extraction/imu.py"),
    )


def run_commands(commands: Iterable[Sequence[str]], env: dict[str, str]) -> None:
    for command in commands:
        print("[run]", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.seed is not None:
        env["SMART_AR_RANDOM_SEED"] = str(args.seed)
    if args.val_split is not None:
        env["SMART_AR_VAL_SPLIT"] = str(args.val_split)
    if args.test_video_names:
        env["SMART_AR_TEST_VIDEO_NAMES"] = ",".join(args.test_video_names)
    if args.output_dir:
        env["SMART_AR_MODEL_OUTPUT_DIR"] = str(Path(args.output_dir).resolve())
    if args.gesture_feature_dir:
        env["MM_INTENT_GESTURE_FEATURE_DIR"] = str(Path(args.gesture_feature_dir).resolve())
    if args.audio_feature_dir:
        env["MM_INTENT_AUDIO_FEATURE_DIR"] = str(Path(args.audio_feature_dir).resolve())
    if args.text_feature_dir:
        env["MM_INTENT_TEXT_FEATURE_DIR"] = str(Path(args.text_feature_dir).resolve())
    if args.imu_feature_dir:
        env["MM_INTENT_IMU_FEATURE_DIR"] = str(Path(args.imu_feature_dir).resolve())
    if args.gesture_feature_dim is not None:
        env["MM_INTENT_GESTURE_FEAT_DIM"] = str(args.gesture_feature_dim)
    if args.audio_feature_dim is not None:
        env["MM_INTENT_AUDIO_FEAT_DIM"] = str(args.audio_feature_dim)
    if args.text_feature_dim is not None:
        env["MM_INTENT_TEXT_FEAT_DIM"] = str(args.text_feature_dim)
    if args.imu_feature_dim is not None:
        env["MM_INTENT_IMU_FEAT_DIM"] = str(args.imu_feature_dim)
    if args.epochs is not None:
        env["BASELINE_SCENE_EPOCHS"] = str(args.epochs)
        env["IMPROVED_REAL_SCENE_A2_EPOCHS"] = str(args.epochs)
    if args.patience is not None:
        env["BASELINE_SCENE_PATIENCE"] = str(args.patience)
        env["IMPROVED_REAL_SCENE_A2_PATIENCE"] = str(args.patience)
    if args.batch_size is not None:
        env["BASELINE_SCENE_BATCH_SIZE"] = str(args.batch_size)
        env["IMPROVED_REAL_SCENE_A2_BATCH_SIZE"] = str(args.batch_size)
    if args.missing_modalities:
        env["SMART_AR_MISSING_MODALITIES"] = ",".join(args.missing_modalities)
    if args.noise_modality:
        env["SMART_AR_NOISE_MODALITY"] = args.noise_modality
        env["SMART_AR_NOISE_LEVEL"] = str(args.noise_level)
    if args.consistency_weight is not None:
        env["IMPROVED_REAL_SCENE_A2_CONSISTENCY_WEIGHT"] = str(args.consistency_weight)
    if args.consistency_mask_prob is not None:
        env["IMPROVED_REAL_SCENE_A2_CONSISTENCY_MASK_PROB"] = str(args.consistency_mask_prob)
    if args.consistency_noise_std is not None:
        env["IMPROVED_REAL_SCENE_A2_CONSISTENCY_NOISE_STD"] = str(args.consistency_noise_std)
    if args.consistency_temperature is not None:
        env["IMPROVED_REAL_SCENE_A2_CONSISTENCY_TEMPERATURE"] = str(args.consistency_temperature)
    if args.consistency_modalities:
        env["IMPROVED_REAL_SCENE_A2_CONSISTENCY_MODALITIES"] = ",".join(args.consistency_modalities)
    if args.margin_loss_weight is not None:
        env["IMPROVED_REAL_SCENE_A2_MARGIN_LOSS_WEIGHT"] = str(args.margin_loss_weight)
    if args.margin_value is not None:
        env["IMPROVED_REAL_SCENE_A2_MARGIN_VALUE"] = str(args.margin_value)
    if args.margin_intent_confusion_weight is not None:
        env["IMPROVED_REAL_SCENE_A2_MARGIN_INTENT_CONFUSION_WEIGHT"] = str(args.margin_intent_confusion_weight)
    if args.margin_scene_confusion_weight is not None:
        env["IMPROVED_REAL_SCENE_A2_MARGIN_SCENE_CONFUSION_WEIGHT"] = str(args.margin_scene_confusion_weight)
    if args.missing_distill_weight is not None:
        env["IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_WEIGHT"] = str(args.missing_distill_weight)
    if args.missing_distill_temperature is not None:
        env["IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_TEMPERATURE"] = str(args.missing_distill_temperature)
    if args.missing_distill_intent_weight is not None:
        env["IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_INTENT_WEIGHT"] = str(args.missing_distill_intent_weight)
    if args.missing_distill_scene_weight is not None:
        env["IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_SCENE_WEIGHT"] = str(args.missing_distill_scene_weight)
    if args.missing_distill_modalities:
        env["IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_MODALITIES"] = ",".join(args.missing_distill_modalities)
    if args.missing_distill_force_mask is not None:
        env["IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_FORCE_MASK"] = "1" if args.missing_distill_force_mask else "0"
    for modality, value in args.missing_distill_probs:
        env[f"IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_{modality.upper()}_PROB"] = str(value)
    if args.focal_loss_gamma is not None:
        env["IMPROVED_REAL_SCENE_A2_FOCAL_LOSS_GAMMA"] = str(args.focal_loss_gamma)
    if args.no_focal_loss_apply_aux:
        env["IMPROVED_REAL_SCENE_A2_FOCAL_LOSS_APPLY_AUX"] = "0"
    if args.fallback_max_gate is not None:
        env["IMPROVED_REAL_SCENE_A2_FALLBACK_MAX_GATE"] = str(args.fallback_max_gate)
    if args.fallback_aux_weight is not None:
        env["IMPROVED_REAL_SCENE_A2_FALLBACK_AUX_WEIGHT"] = str(args.fallback_aux_weight)
    if args.supcon_loss_weight is not None:
        env["IMPROVED_REAL_SCENE_A2_SUPCON_LOSS_WEIGHT"] = str(args.supcon_loss_weight)
    if args.supcon_temperature is not None:
        env["IMPROVED_REAL_SCENE_A2_SUPCON_TEMPERATURE"] = str(args.supcon_temperature)
    if args.supcon_target is not None:
        env["IMPROVED_REAL_SCENE_A2_SUPCON_TARGET"] = args.supcon_target
    if args.skip_test_eval:
        env["SMART_AR_SKIP_TEST_EVAL"] = "1"
    return env


def train_command(model: str) -> Sequence[str]:
    if model == "baseline":
        return (sys.executable, "code/baseline_real_scene.py")
    if model == "improved":
        return (sys.executable, "code/train_and_test.py")
    raise ValueError(f"Unsupported model: {model}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end training entry for the MM-Intent project.")
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument("--expected-count", type=int, default=39)
    parser.add_argument("--extract-features", action="store_true", help="Run feature extraction if cached features are incomplete.")
    parser.add_argument("--skip-feature-check", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val-split", type=float)
    parser.add_argument("--test-video-names", nargs="*", default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--gesture-feature-dir")
    parser.add_argument("--audio-feature-dir")
    parser.add_argument("--text-feature-dir")
    parser.add_argument("--imu-feature-dir")
    parser.add_argument("--gesture-feature-dim", type=int)
    parser.add_argument("--audio-feature-dim", type=int)
    parser.add_argument("--text-feature-dim", type=int)
    parser.add_argument("--imu-feature-dim", type=int)
    parser.add_argument("--missing-modalities", nargs="*", default=[])
    parser.add_argument("--noise-modality", choices=("imu", "gesture", "audio", "text", "scene"))
    parser.add_argument("--noise-level", type=float, default=0.0)
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
    parser.add_argument("--missing-distill-force-mask", dest="missing_distill_force_mask", action="store_true", default=None)
    parser.add_argument("--no-missing-distill-force-mask", dest="missing_distill_force_mask", action="store_false")
    parser.add_argument(
        "--missing-distill-prob",
        nargs=2,
        action="append",
        metavar=("MODALITY", "PROB"),
        default=[],
        help="Per-modality missing distillation drop probability, e.g. --missing-distill-prob text 0.35",
    )
    parser.add_argument("--focal-loss-gamma", type=float)
    parser.add_argument("--no-focal-loss-apply-aux", action="store_true")
    parser.add_argument("--fallback-max-gate", type=float)
    parser.add_argument("--fallback-aux-weight", type=float)
    parser.add_argument("--supcon-loss-weight", type=float)
    parser.add_argument("--supcon-temperature", type=float)
    parser.add_argument("--supcon-target", choices=("joint", "intent", "scene"))
    parser.add_argument("--skip-test-eval", action="store_true")
    args = parser.parse_args()
    args.missing_distill_probs = [
        (modality, float(value))
        for modality, value in args.missing_distill_prob
    ]

    env = build_env(args)
    if not args.skip_feature_check:
        features_ready = check_features(args.expected_count)
        if not features_ready:
            if not args.extract_features:
                raise SystemExit("Cached features are incomplete. Re-run with --extract-features or generate features first.")
            run_commands(feature_commands(), env)
            if not check_features(args.expected_count):
                raise SystemExit("Feature extraction finished but cached features are still incomplete.")

    print(f"[train] model={args.model}")
    if args.missing_modalities:
        print(f"[train] missing_modalities={','.join(args.missing_modalities)}")
    if args.noise_modality:
        print(f"[train] noise={args.noise_modality}:{args.noise_level}")
    run_commands((train_command(args.model),), env)


if __name__ == "__main__":
    main()
