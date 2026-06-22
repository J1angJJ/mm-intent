from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from raw_data_utils import RAW_MODALITIES, raw_noise_definition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_TEST_VIDEO_NAMES = (
    "interaction_20260306_072344.mp4",
    "interaction_20260227_122606.mp4",
    "interaction_20260227_122952.mp4",
    "interaction_20260227_123354.mp4",
    "interaction_20260227_124559.mp4",
    "interaction_20260227_123745.mp4",
    "interaction_20260306_082346.mp4",
    "interaction_20260306_083107.mp4",
    "interaction_20260306_083434.mp4",
    "interaction_20260306_084406.mp4",
    "interaction_20260306_084853.mp4",
    "interaction_20260306_085830.mp4",
    "interaction_20260306_090441.mp4",
)
COURSE_TRAIN_VIDEO_NAMES = (
    "interaction_20260131_120024.mp4",
    "interaction_20260227_132951.mp4",
    "interaction_20260227_133408.mp4",
    "interaction_20260131_114156.mp4",
    "interaction_20260131_115150.mp4",
    "interaction_20260131_114852.mp4",
    "interaction_20260301_073041.mp4",
    "interaction_20260301_064753.mp4",
    "interaction_20260306_072721.mp4",
    "interaction_20260301_071948.mp4",
    "interaction_20260131_121548.mp4",
    "interaction_20260301_073435.mp4",
    "interaction_20260301_072503.mp4",
    "interaction_20260131_071552.mp4",
    "interaction_20260131_072412.mp4",
    "interaction_20260131_084300.mp4",
    "interaction_20260131_084732.mp4",
    "interaction_20260131_085207.mp4",
    "interaction_20260131_085611.mp4",
    "interaction_20260131_090139.mp4",
    "interaction_20260131_065459.mp4",
    "interaction_20260131_070722.mp4",
    "interaction_20260131_090541.mp4",
    "interaction_20260131_090917.mp4",
    "interaction_20260131_091249.mp4",
    "interaction_20260131_091657.mp4",
)
COURSE_VIDEO_NAMES = COURSE_TRAIN_VIDEO_NAMES + COURSE_TEST_VIDEO_NAMES
FEATURE_PATTERNS = {
    "timestamp": "features_timestamp_{base}.npy",
    "gesture": "strong_gesture_features/strong_gesture_features_{base}.npy",
    "audio": "audio_features/audio_features_{base}.npy",
    "text": "text_features/text_features_{base}.npy",
    "imu": "imu_features/imu_features_{base}.npy",
}
EXTRACTION_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("code/feature_extraction/get_timestamp.py",),
    ("code/feature_extraction/strong_gesture2.0.py",),
    ("code/feature_extraction/mfcc.py",),
    ("code/feature_extraction/ASR.py",),
    ("code/feature_extraction/imu.py",),
)
MODALITY_EXTRACTION_COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "gesture": (("code/feature_extraction/strong_gesture2.0.py",),),
    "audio": (("code/feature_extraction/mfcc.py",),),
    "text": (("code/feature_extraction/ASR.py",),),
    "imu": (("code/feature_extraction/imu.py",),),
    "scene": (),
}


def default_raw_cache_dir(noise_modality: str, noise_level: float, noise_seed: int) -> Path:
    if noise_modality and noise_level > 0.0:
        condition = f"{noise_modality}_noise_{int(round(noise_level * 100))}"
    else:
        condition = "clean"
    return PROJECT_ROOT / "outputs" / "raw_feature_cache" / f"{condition}_seed{noise_seed}"


def expected_feature_paths(
    output_dir: Path,
    video_names: Sequence[str],
    gesture_representation: str = "clip",
) -> list[Path]:
    paths: list[Path] = []
    for video_name in video_names:
        base = Path(video_name).stem
        paths.extend(output_dir / pattern.format(base=base) for pattern in FEATURE_PATTERNS.values())
        if gesture_representation == "hand_geometry":
            paths.append(output_dir / "hand_geometry_features" / f"strong_gesture_features_{base}.npy")
    return paths


def missing_feature_paths(
    output_dir: Path,
    video_names: Sequence[str],
    gesture_representation: str = "clip",
) -> list[Path]:
    return [
        path
        for path in expected_feature_paths(output_dir, video_names, gesture_representation)
        if not path.exists()
    ]


def build_manifest(
    output_dir: Path,
    video_names: Sequence[str],
    noise_modality: str,
    noise_level: float,
    noise_seed: int,
    env: dict[str, str],
    base_feature_dir: Path | None,
    gesture_representation: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_mode": "raw",
        "raw_inputs": {
            "dataset_dir": env.get("MM_INTENT_DATASET_DIR", str(PROJECT_ROOT / "dataset")),
            "hololens_dir": env.get("MM_INTENT_HOLOLENS_DIR", str(PROJECT_ROOT / "dataset" / "HoloLens")),
            "fisheye_dir": env.get("MM_INTENT_FISHEYE_DIR", str(PROJECT_ROOT / "dataset" / "fisheye")),
        },
        "processed_output_dir": str(output_dir),
        "unchanged_feature_cache": str(base_feature_dir) if base_feature_dir else None,
        "video_names": list(video_names),
        "gesture_representation": gesture_representation,
        "raw_noise": {
            "modality": noise_modality,
            "level": noise_level,
            "seed": noise_seed,
            "definition": raw_noise_definition(noise_modality, noise_level),
        },
        "scene_processing": "Frames are decoded and ViT-encoded lazily during train/test; the scene cache is condition-specific.",
    }


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_feature_directory(source_dir: Path, destination_dir: Path) -> None:
    if not source_dir.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.npy"):
        _link_or_copy(source, destination_dir / source.name)


def seed_unchanged_features(
    output_dir: Path,
    base_feature_dir: Path,
    video_names: Sequence[str],
    target_modality: str,
    gesture_representation: str,
) -> None:
    if not base_feature_dir.exists():
        raise FileNotFoundError(f"Clean feature cache does not exist: {base_feature_dir}")

    print(
        f"[raw-preprocess] reuse unchanged raw-derived features from {base_feature_dir}; "
        f"re-extract target={target_modality}"
    )
    for video_name in video_names:
        base = Path(video_name).stem
        timestamp_source = base_feature_dir / f"features_timestamp_{base}.npy"
        if not timestamp_source.exists():
            raise FileNotFoundError(f"Missing clean timestamp cache: {timestamp_source}")
        _link_or_copy(timestamp_source, output_dir / timestamp_source.name)

        metadata_source = base_feature_dir / f"metadata_strong_gesture_{base}.npy"
        if target_modality != "gesture" or gesture_representation == "hand_geometry":
            if not metadata_source.exists():
                raise FileNotFoundError(f"Missing clean gesture metadata: {metadata_source}")
            _link_or_copy(metadata_source, output_dir / metadata_source.name)

        for modality, pattern in FEATURE_PATTERNS.items():
            if modality == "timestamp":
                continue
            if modality == target_modality and not (
                modality == "gesture" and gesture_representation == "hand_geometry"
            ):
                continue
            relative_path = Path(pattern.format(base=base))
            source = base_feature_dir / relative_path
            if not source.exists():
                raise FileNotFoundError(f"Missing clean {modality} feature: {source}")
            _link_or_copy(source, output_dir / relative_path)

        if gesture_representation == "hand_geometry" and target_modality != "gesture":
            relative_path = Path("hand_geometry_features") / f"strong_gesture_features_{base}.npy"
            source = base_feature_dir / relative_path
            if not source.exists():
                raise FileNotFoundError(f"Missing clean hand-geometry feature: {source}")
            _link_or_copy(source, output_dir / relative_path)


def _manifest_matches(
    manifest: dict[str, object],
    video_names: Sequence[str],
    noise_modality: str,
    noise_level: float,
    noise_seed: int,
    gesture_representation: str,
) -> bool:
    raw_noise = manifest.get("raw_noise", {})
    if not isinstance(raw_noise, dict):
        return False
    cached_names = manifest.get("video_names", [])
    return (
        set(cached_names if isinstance(cached_names, list) else []) >= set(video_names)
        and raw_noise.get("modality") == noise_modality
        and float(raw_noise.get("level", -1.0)) == float(noise_level)
        and int(raw_noise.get("seed", -1)) == int(noise_seed)
        and manifest.get("gesture_representation", "clip") == gesture_representation
    )


def run_commands(commands: Iterable[Sequence[str]], env: dict[str, str], dry_run: bool) -> None:
    for command in commands:
        full_command = (sys.executable, *command)
        print("[raw-preprocess]", " ".join(full_command), flush=True)
        if not dry_run:
            subprocess.run(full_command, cwd=PROJECT_ROOT, env=env, check=True)


def prepare_raw_features(
    *,
    output_dir: Path,
    video_names: Sequence[str],
    noise_modality: str = "",
    noise_level: float = 0.0,
    noise_seed: int = 42,
    base_env: dict[str, str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    base_feature_dir: Path | None = None,
    gesture_representation: str = "clip",
) -> dict[str, object]:
    if noise_modality and noise_modality not in RAW_MODALITIES:
        raise ValueError(f"Unknown raw-noise modality: {noise_modality}")
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError(f"Raw-noise level must be in [0, 1], got {noise_level}")
    if noise_level > 0.0 and not noise_modality:
        raise ValueError("A raw-noise modality is required when noise level is positive.")
    if not video_names:
        raise ValueError("At least one video name is required for raw preprocessing.")
    if gesture_representation not in ("clip", "hand_geometry"):
        raise ValueError(f"Unknown gesture representation: {gesture_representation}")

    output_dir = output_dir.resolve()
    base_feature_dir = base_feature_dir.resolve() if base_feature_dir else None
    manifest_path = output_dir / "raw_preprocessing_manifest.json"
    existing_manifest: dict[str, object] | None = None
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _manifest_matches(
            existing_manifest,
            video_names,
            noise_modality,
            noise_level,
            noise_seed,
            gesture_representation,
        ) and not force:
            raise RuntimeError(
                f"Raw cache manifest does not match the requested videos/noise condition: {manifest_path}. "
                "Choose another --raw-cache-dir or use --force-preprocess."
            )

    missing_before = missing_feature_paths(output_dir, video_names, gesture_representation)
    if not force and not missing_before and existing_manifest is not None:
        print(f"[raw-preprocess] reuse complete cache: {output_dir}")
        return existing_manifest

    env = dict(base_env or os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env["MM_INTENT_PROCESSED_DATA_DIR"] = str(output_dir)
    env["MM_INTENT_VIDEO_NAMES"] = ",".join(video_names)
    env["MM_INTENT_RAW_NOISE_MODALITY"] = noise_modality
    env["MM_INTENT_RAW_NOISE_LEVEL"] = str(noise_level)
    env["MM_INTENT_RAW_NOISE_SEED"] = str(noise_seed)
    env["MM_INTENT_SCENE_CACHE_DIR"] = str(output_dir / "scene_features")

    print(
        f"[raw-preprocess] videos={len(video_names)} output={output_dir} "
        f"noise={noise_modality or 'none'}:{noise_level} seed={noise_seed} force={int(force)}"
    )
    commands = EXTRACTION_COMMANDS
    reuse_target = noise_modality if noise_modality and noise_level > 0.0 else ""
    if not reuse_target and gesture_representation == "hand_geometry" and base_feature_dir is not None:
        reuse_target = "gesture"
    if reuse_target and base_feature_dir is not None:
        if not dry_run:
            seed_unchanged_features(
                output_dir,
                base_feature_dir,
                video_names,
                reuse_target,
                gesture_representation,
            )
        else:
            print(
                f"[raw-preprocess] would reuse unchanged features from {base_feature_dir} "
                f"and re-extract only {reuse_target}"
            )
        commands = MODALITY_EXTRACTION_COMMANDS[reuse_target]
        if noise_modality != "scene":
            clean_scene_cache = Path(
                env.get(
                    "MM_INTENT_CLEAN_SCENE_CACHE_DIR",
                    str(PROJECT_ROOT / "outputs" / "raw_feature_cache" / "clean_seed42" / "scene_features"),
                )
            ).resolve()
            link_feature_directory(clean_scene_cache, output_dir / "scene_features")
    if gesture_representation == "hand_geometry" and reuse_target == "gesture":
        commands = (
            (
                "code/feature_extraction/extract_hand_geometry_features.py",
                "--output-dir",
                str(output_dir / "hand_geometry_features"),
            ),
        )
    elif gesture_representation == "hand_geometry" and not reuse_target:
        commands = (
            *commands,
            (
                "code/feature_extraction/extract_hand_geometry_features.py",
                "--output-dir",
                str(output_dir / "hand_geometry_features"),
            ),
        )

    run_commands(commands, env, dry_run=dry_run)
    manifest = build_manifest(
        output_dir,
        video_names,
        noise_modality,
        noise_level,
        noise_seed,
        env,
        base_feature_dir,
        gesture_representation,
    )
    if dry_run:
        return manifest

    missing_after = missing_feature_paths(output_dir, video_names, gesture_representation)
    if missing_after:
        examples = "\n".join(f"  - {path}" for path in missing_after[:10])
        raise RuntimeError(
            f"Raw preprocessing finished with {len(missing_after)} missing feature files.\n{examples}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[raw-preprocess] manifest={manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw-data preprocessing entry for MM-Intent.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-names", nargs="+", required=True)
    parser.add_argument("--noise-modality", choices=RAW_MODALITIES, default="")
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--dataset-dir")
    parser.add_argument("--hololens-dir")
    parser.add_argument("--fisheye-dir")
    parser.add_argument("--base-feature-dir")
    parser.add_argument("--gesture-representation", choices=("clip", "hand_geometry"), default="clip")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    for argument_name, environment_name in (
        ("dataset_dir", "MM_INTENT_DATASET_DIR"),
        ("hololens_dir", "MM_INTENT_HOLOLENS_DIR"),
        ("fisheye_dir", "MM_INTENT_FISHEYE_DIR"),
    ):
        value = getattr(args, argument_name)
        if value:
            env[environment_name] = str(Path(value).resolve())

    prepare_raw_features(
        output_dir=Path(args.output_dir),
        video_names=args.video_names,
        noise_modality=args.noise_modality,
        noise_level=args.noise_level,
        noise_seed=args.noise_seed,
        base_env=env,
        force=args.force,
        dry_run=args.dry_run,
        base_feature_dir=Path(args.base_feature_dir) if args.base_feature_dir else None,
        gesture_representation=args.gesture_representation,
    )


if __name__ == "__main__":
    main()
