from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from project_paths import MODEL_OUTPUT_ROOT, PROCESSED_DATA_DIR, PROJECT_ROOT
from raw_pipeline import (
    COURSE_VIDEO_NAMES,
    default_raw_cache_dir,
    default_raw_missing_cache_dir,
    prepare_raw_features,
)


FEATURE_FILE_PATTERNS = {
    "timestamp": "features_timestamp_*.npy",
    "gesture": "strong_gesture_features_*.npy",
    "audio": "audio_features_*.npy",
    "text": "text_features_*.npy",
    "imu": "imu_features_*.npy",
}


def count_pattern(directory: Path, pattern: str) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def resolve_feature_directories(
    *,
    processed_data_dir: Path,
    gesture_representation: str,
    gesture_feature_dir: str | None,
    audio_feature_dir: str | None,
    text_feature_dir: str | None,
    imu_feature_dir: str | None,
) -> dict[str, Path]:
    default_gesture_subdir = (
        "hand_geometry_features"
        if gesture_representation == "hand_geometry"
        else "strong_gesture_features"
    )
    return {
        "timestamp": processed_data_dir,
        "gesture": (
            Path(gesture_feature_dir).resolve()
            if gesture_feature_dir
            else (processed_data_dir / default_gesture_subdir).resolve()
        ),
        "audio": (
            Path(audio_feature_dir).resolve()
            if audio_feature_dir
            else (processed_data_dir / "audio_features").resolve()
        ),
        "text": (
            Path(text_feature_dir).resolve()
            if text_feature_dir
            else (processed_data_dir / "text_features").resolve()
        ),
        "imu": (
            Path(imu_feature_dir).resolve()
            if imu_feature_dir
            else (processed_data_dir / "imu_features").resolve()
        ),
    }


def check_features(
    *,
    processed_data_dir: Path,
    expected_count: int,
    gesture_representation: str,
    gesture_feature_dir: str | None = None,
    audio_feature_dir: str | None = None,
    text_feature_dir: str | None = None,
    imu_feature_dir: str | None = None,
) -> bool:
    """Validate cached feature-mode inputs.

    Raw-mode validation is performed by raw_pipeline.prepare_raw_features(),
    which understands raw-noise, raw-missing and Hand Geometry caches.
    Feature-mode missing experiments still require the source files because
    labels, timestamps and tensor shapes are read before the selected
    modalities are zeroed by the model loader.
    """
    directories = resolve_feature_directories(
        processed_data_dir=processed_data_dir,
        gesture_representation=gesture_representation,
        gesture_feature_dir=gesture_feature_dir,
        audio_feature_dir=audio_feature_dir,
        text_feature_dir=text_feature_dir,
        imu_feature_dir=imu_feature_dir,
    )

    print(f"[feature-check] processed_dir={processed_data_dir}")
    ok = True
    for name in ("timestamp", "gesture", "audio", "text", "imu"):
        directory = directories[name]
        pattern = FEATURE_FILE_PATTERNS[name]
        count = count_pattern(directory, pattern)
        print(f"  {name:9s} {count}/{expected_count}  dir={directory}")
        ok = ok and count >= expected_count
    return ok


def feature_commands(
    gesture_representation: str,
    processed_data_dir: Path,
) -> Sequence[Sequence[str]]:
    commands: list[Sequence[str]] = [
        (sys.executable, "code/feature_extraction/get_timestamp.py"),
        (sys.executable, "code/feature_extraction/strong_gesture2.0.py"),
        (sys.executable, "code/feature_extraction/mfcc.py"),
        (sys.executable, "code/feature_extraction/ASR.py"),
        (sys.executable, "code/feature_extraction/imu.py"),
    ]
    if gesture_representation == "hand_geometry":
        commands.append(
            (
                sys.executable,
                "code/feature_extraction/extract_hand_geometry_features.py",
                "--output-dir",
                str((processed_data_dir / "hand_geometry_features").resolve()),
            )
        )
    return tuple(commands)


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
    if args.processed_data_dir:
        env["MM_INTENT_PROCESSED_DATA_DIR"] = str(Path(args.processed_data_dir).resolve())
    if args.dataset_dir:
        env["MM_INTENT_DATASET_DIR"] = str(Path(args.dataset_dir).resolve())
    if args.hololens_dir:
        env["MM_INTENT_HOLOLENS_DIR"] = str(Path(args.hololens_dir).resolve())
    if args.fisheye_dir:
        env["MM_INTENT_FISHEYE_DIR"] = str(Path(args.fisheye_dir).resolve())
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
    if args.missing_modalities:
        env["SMART_AR_MISSING_MODALITIES"] = ",".join(args.missing_modalities)
    if args.noise_modality:
        if args.noise_space == "raw":
            env["MM_INTENT_RAW_NOISE_MODALITY"] = args.noise_modality
            env["MM_INTENT_RAW_NOISE_LEVEL"] = str(args.noise_level)
            env["MM_INTENT_RAW_NOISE_SEED"] = str(args.noise_seed)
            env.pop("SMART_AR_NOISE_MODALITY", None)
            env["SMART_AR_NOISE_LEVEL"] = "0.0"
        else:
            env["SMART_AR_NOISE_MODALITY"] = args.noise_modality
            env["SMART_AR_NOISE_LEVEL"] = str(args.noise_level)
            env["SMART_AR_NOISE_SEED"] = str(args.noise_seed)
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


def default_raw_cache_for_request(
    *,
    noise_modality: str,
    noise_level: float,
    noise_seed: int,
    gesture_representation: str,
) -> Path:
    path = default_raw_cache_dir(noise_modality, noise_level, noise_seed)
    if gesture_representation == "hand_geometry":
        return path.with_name(path.name + "_hand_geometry")
    return path


def train_command(model: str) -> Sequence[str]:
    if model == "baseline":
        return (sys.executable, "code/baseline_real_scene.py")
    if model == "improved":
        return (sys.executable, "code/train_and_test.py")
    raise ValueError(f"Unsupported model: {model}")


def resolve_model_output_dir(model: str, env: dict[str, str]) -> Path:
    configured = env.get("SMART_AR_MODEL_OUTPUT_DIR")
    if configured:
        return Path(configured).resolve()
    default_name = (
        "baseline_real_scene_perceiver_io"
        if model == "baseline"
        else "improved_real_scene_anchor2_perceiver_io"
    )
    return (MODEL_OUTPUT_ROOT / default_name).resolve()


def update_metrics_with_entry_timing(
    metrics_path: Path,
    *,
    input_mode: str,
    feature_preparation_seconds: float,
    training_command_seconds: float,
    end_to_end_training_seconds: float,
) -> None:
    if not metrics_path.exists():
        print(f"[timing-warning] metrics file not found: {metrics_path}")
        return

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[timing-warning] unable to update {metrics_path}: {exc}")
        return

    timing = dict(payload.get("timing", {}))
    train_sample_count = int(payload.get("splits", {}).get("train_samples", 0) or 0)
    timing.update(
        {
            "input_mode": input_mode,
            "feature_preparation_seconds": float(feature_preparation_seconds),
            "training_command_seconds": float(training_command_seconds),
            "end_to_end_training_seconds": float(end_to_end_training_seconds),
            "avg_end_to_end_training_seconds_per_unique_sample": (
                float(end_to_end_training_seconds / train_sample_count)
                if train_sample_count > 0
                else 0.0
            ),
            "avg_end_to_end_training_seconds_per_unique_sample_definition": (
                "end_to_end_training_seconds / train_sample_count"
            ),
        }
    )
    payload["timing"] = timing
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    entry_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="End-to-end training entry for the MM-Intent project.")
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument(
        "--input-mode",
        choices=("features", "raw"),
        default="features",
        help="Use cached features or run the integrated raw-data preprocessing pipeline.",
    )
    parser.add_argument("--expected-count", type=int, default=39)
    parser.add_argument("--extract-features", action="store_true", help="Run feature extraction if cached features are incomplete.")
    parser.add_argument("--processed-data-dir")
    parser.add_argument("--raw-cache-dir")
    parser.add_argument(
        "--base-feature-dir",
        help=("Complete clean cache reused by raw-noise/raw-missing preprocessing "
              "and by the loader for alignment-only fallbacks."),
    )
    parser.add_argument("--gesture-representation", choices=("clip", "hand_geometry"), default="clip")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--hololens-dir")
    parser.add_argument("--fisheye-dir")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--preprocess-dry-run", action="store_true")
    parser.add_argument("--skip-feature-check", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
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
    parser.add_argument("--noise-space", choices=("feature", "raw"))
    parser.add_argument("--noise-seed", type=int, default=42)
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
    if args.noise_space is None:
        args.noise_space = "raw" if args.input_mode == "raw" else "feature"
    if args.noise_space == "raw" and args.input_mode != "raw":
        parser.error("--noise-space raw requires --input-mode raw")
    if args.noise_level > 0.0 and not args.noise_modality:
        parser.error("--noise-modality is required when --noise-level is positive")
    if not 0.0 <= args.noise_level <= 1.0:
        parser.error("--noise-level must be in [0, 1]")

    raw_cache_dir: Path | None = None
    base_feature_dir: Path | None = None
    if args.input_mode == "raw":
        raw_cache_dir = (
            Path(args.raw_cache_dir).resolve()
            if args.raw_cache_dir
            else default_raw_missing_cache_dir(
                args.missing_modalities,
                args.noise_seed,
                args.gesture_representation,
            )
            if args.missing_modalities
            else default_raw_cache_for_request(
                noise_modality=args.noise_modality or "",
                noise_level=args.noise_level,
                noise_seed=args.noise_seed,
                gesture_representation=args.gesture_representation,
            )
        )
        args.processed_data_dir = str(raw_cache_dir)
        if args.gesture_representation == "hand_geometry":
            args.gesture_feature_dir = str(
                raw_cache_dir / "hand_geometry_features"
            )
            args.gesture_feature_dim = 96
        if args.base_feature_dir:
            base_feature_dir = Path(args.base_feature_dir).resolve()
        elif args.missing_modalities:
            suffix = (
                "_hand_geometry"
                if args.gesture_representation == "hand_geometry"
                else ""
            )
            base_feature_dir = (
                PROJECT_ROOT
                / "outputs"
                / "raw_feature_cache"
                / f"clean_seed{args.noise_seed}{suffix}"
            ).resolve()
        elif args.noise_modality and args.noise_level > 0.0:
            dataset_root = (
                Path(args.dataset_dir).resolve()
                if args.dataset_dir
                else PROJECT_ROOT / "dataset"
            )
            base_feature_dir = (
                dataset_root / "AR_Data_Process3.0" / "data_full"
            ).resolve()
    if (
        args.input_mode == "features"
        and args.gesture_representation == "hand_geometry"
    ):
        feature_root = (
            Path(args.processed_data_dir).resolve()
            if args.processed_data_dir
            else Path(PROCESSED_DATA_DIR).resolve()
        )
        if not args.gesture_feature_dir:
            args.gesture_feature_dir = str(
                feature_root / "hand_geometry_features"
            )
        if args.gesture_feature_dim is None:
            args.gesture_feature_dim = 96

    args.missing_distill_probs = [
        (modality, float(value))
        for modality, value in args.missing_distill_prob
    ]

    env = build_env(args)
    if base_feature_dir is not None:
        env["MM_INTENT_BASE_FEATURE_DIR"] = str(base_feature_dir.resolve())
    processed_data_dir = Path(
        env.get("MM_INTENT_PROCESSED_DATA_DIR", str(PROCESSED_DATA_DIR))
    ).resolve()
    feature_preparation_start = time.perf_counter()
    if args.input_mode == "raw":
        assert raw_cache_dir is not None
        env["MM_INTENT_SCENE_CACHE_DIR"] = str(raw_cache_dir / "scene_features")
        raw_manifest = prepare_raw_features(
            output_dir=raw_cache_dir,
            video_names=COURSE_VIDEO_NAMES,
            noise_modality=args.noise_modality or "",
            noise_level=args.noise_level,
            noise_seed=args.noise_seed,
            base_env=env,
            force=args.force_preprocess,
            dry_run=args.preprocess_dry_run,
            base_feature_dir=base_feature_dir,
            gesture_representation=args.gesture_representation,
            missing_modalities=args.missing_modalities,
        )
        scene_cache_dir = raw_manifest.get("scene_cache_dir")
        if scene_cache_dir:
            env["MM_INTENT_SCENE_CACHE_DIR"] = str(
                Path(str(scene_cache_dir)).resolve()
            )
        if args.preprocess_dry_run:
            print("[train] preprocessing dry-run complete; training was not started.")
            return
    elif not args.skip_feature_check:
        feature_check_kwargs = {
            "processed_data_dir": processed_data_dir,
            "expected_count": args.expected_count,
            "gesture_representation": args.gesture_representation,
            "gesture_feature_dir": args.gesture_feature_dir,
            "audio_feature_dir": args.audio_feature_dir,
            "text_feature_dir": args.text_feature_dir,
            "imu_feature_dir": args.imu_feature_dir,
        }
        features_ready = check_features(**feature_check_kwargs)
        if not features_ready:
            if not args.extract_features:
                raise SystemExit(
                    "Cached features are incomplete. Re-run with "
                    "--extract-features or generate features first."
                )
            run_commands(
                feature_commands(
                    args.gesture_representation,
                    processed_data_dir,
                ),
                env,
            )
            if not check_features(**feature_check_kwargs):
                raise SystemExit(
                    "Feature extraction finished but cached features "
                    "are still incomplete."
                )

    feature_preparation_seconds = time.perf_counter() - feature_preparation_start
    print(f"[train] model={args.model} input_mode={args.input_mode}")
    if args.missing_modalities:
        print(f"[train] missing_modalities={','.join(args.missing_modalities)}")
    if args.noise_modality:
        print(f"[train] noise={args.noise_modality}:{args.noise_level} space={args.noise_space}")

    training_command_start = time.perf_counter()
    run_commands((train_command(args.model),), env)
    training_command_seconds = time.perf_counter() - training_command_start
    end_to_end_training_seconds = time.perf_counter() - entry_start

    output_dir = resolve_model_output_dir(args.model, env)
    metrics_path = output_dir / "metrics.json"
    update_metrics_with_entry_timing(
        metrics_path,
        input_mode=args.input_mode,
        feature_preparation_seconds=feature_preparation_seconds,
        training_command_seconds=training_command_seconds,
        end_to_end_training_seconds=end_to_end_training_seconds,
    )
    print(
        "[train-timing] "
        f"feature_preparation={feature_preparation_seconds:.3f}s "
        f"training_command={training_command_seconds:.3f}s "
        f"end_to_end={end_to_end_training_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
