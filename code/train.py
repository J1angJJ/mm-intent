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
    if args.epochs is not None:
        env["BASELINE_SCENE_EPOCHS"] = str(args.epochs)
        env["IMPROVED_REAL_SCENE_A2_EPOCHS"] = str(args.epochs)
    if args.patience is not None:
        env["BASELINE_SCENE_PATIENCE"] = str(args.patience)
        env["IMPROVED_REAL_SCENE_A2_PATIENCE"] = str(args.patience)
    if args.missing_modalities:
        env["SMART_AR_MISSING_MODALITIES"] = ",".join(args.missing_modalities)
    if args.noise_modality:
        env["SMART_AR_NOISE_MODALITY"] = args.noise_modality
        env["SMART_AR_NOISE_LEVEL"] = str(args.noise_level)
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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val-split", type=float)
    parser.add_argument("--test-video-names", nargs="*", default=[])
    parser.add_argument("--output-dir")
    parser.add_argument("--missing-modalities", nargs="*", default=[])
    parser.add_argument("--noise-modality", choices=("imu", "gesture", "audio", "text", "scene"))
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--skip-test-eval", action="store_true")
    args = parser.parse_args()

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
