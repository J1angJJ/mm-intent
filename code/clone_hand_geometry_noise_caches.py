from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from raw_pipeline import (
    COURSE_VIDEO_NAMES,
    FEATURE_PATTERNS,
    _link_or_copy,
    build_manifest,
    link_feature_directory,
    missing_feature_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def clone_condition(
    *,
    source: Path,
    destination: Path,
    clean_features: Path,
    video_names: list[str],
    modality: str,
    level: float,
    seed: int,
) -> None:
    source_manifest = source / "raw_preprocessing_manifest.json"
    if not source_manifest.exists():
        raise FileNotFoundError(f"Source raw-noise cache is incomplete: {source_manifest}")
    destination.mkdir(parents=True, exist_ok=True)

    for video_name in video_names:
        base = Path(video_name).stem
        for pattern in FEATURE_PATTERNS.values():
            relative = Path(pattern.format(base=base))
            source_path = source / relative
            if not source_path.exists():
                raise FileNotFoundError(f"Missing source feature: {source_path}")
            _link_or_copy(source_path, destination / relative)

        metadata = source / f"metadata_strong_gesture_{base}.npy"
        if not metadata.exists():
            metadata = clean_features / f"metadata_strong_gesture_{base}.npy"
        _link_or_copy(metadata, destination / metadata.name)

        hand_relative = Path("hand_geometry_features") / f"strong_gesture_features_{base}.npy"
        hand_source = clean_features / hand_relative
        if not hand_source.exists():
            raise FileNotFoundError(f"Missing clean hand-geometry feature: {hand_source}")
        _link_or_copy(hand_source, destination / hand_relative)

    link_feature_directory(source / "scene_features", destination / "scene_features")
    missing = missing_feature_paths(destination, video_names, "hand_geometry")
    if missing:
        raise RuntimeError(f"Cloned hand-geometry cache is incomplete: {missing[:5]}")

    manifest = build_manifest(
        destination,
        video_names,
        modality,
        level,
        seed,
        os.environ.copy(),
        source,
        "hand_geometry",
    )
    manifest["cache_reuse"] = {
        "source_raw_noise_cache": str(source),
        "reason": "The same raw perturbation is shared across model variants; only the gesture representation differs.",
    }
    (destination / "raw_preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[cloned] {source.name} -> {destination.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse completed raw-noise caches for Hand Geometry without re-extracting unchanged modalities."
    )
    parser.add_argument("--levels", nargs="+", type=float, default=(0.2, 0.4, 0.6))
    parser.add_argument("--modalities", nargs="+", default=("imu", "audio", "text", "scene"))
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "outputs" / "raw_feature_cache"))
    parser.add_argument(
        "--clean-feature-dir",
        default=str(PROJECT_ROOT / "dataset" / "AR_Data_Process3.0" / "data_full"),
    )
    parser.add_argument("--video-names", nargs="*", default=list(COURSE_VIDEO_NAMES))
    args = parser.parse_args()

    cache_root = Path(args.cache_root).resolve()
    clean_features = Path(args.clean_feature_dir).resolve()
    for modality in args.modalities:
        if modality == "gesture":
            raise ValueError("Gesture requires raw Hand Geometry re-extraction; use precompute_gesture_noise_levels.py.")
        for level in args.levels:
            percent = int(round(level * 100))
            source = cache_root / f"{modality}_noise_{percent}_seed{args.noise_seed}"
            destination = cache_root / f"{modality}_noise_{percent}_seed{args.noise_seed}_hand_geometry"
            clone_condition(
                source=source,
                destination=destination,
                clean_features=clean_features,
                video_names=args.video_names,
                modality=modality,
                level=level,
                seed=args.noise_seed,
            )


if __name__ == "__main__":
    main()
