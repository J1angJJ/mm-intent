from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from project_paths import PROCESSED_DATA_DIR


DEFAULT_BASE_DIR = PROCESSED_DATA_DIR / "strong_gesture_features"
DEFAULT_GEOMETRY_DIR = PROCESSED_DATA_DIR / "hand_geometry_features"
DEFAULT_OUTPUT_DIR = PROCESSED_DATA_DIR / "gesture_clip_hand_geometry_features"
FEATURE_PATTERN = "strong_gesture_features_*.npy"


def as_array(payload: dict[str, Any], key: str, path: Path) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"{path} missing key: {key}")
    return np.asarray(payload[key])


def check_optional_equal(base: dict[str, Any], extra: dict[str, Any], key: str, filename: str) -> None:
    if key not in base or key not in extra:
        return
    base_values = np.asarray(base[key], dtype=object)
    extra_values = np.asarray(extra[key], dtype=object)
    if base_values.shape != extra_values.shape or not np.array_equal(base_values, extra_values):
        raise ValueError(f"{filename}: metadata mismatch for key '{key}'")


def fuse_one(base_path: Path, geometry_path: Path, output_path: Path, overwrite: bool) -> tuple[int, int]:
    if output_path.exists() and not overwrite:
        payload = np.load(output_path, allow_pickle=True).item()
        features = as_array(payload, "features", output_path)
        return int(features.shape[0]), int(features.shape[-1])

    base_payload = np.load(base_path, allow_pickle=True).item()
    geometry_payload = np.load(geometry_path, allow_pickle=True).item()

    base_features = as_array(base_payload, "features", base_path).astype(np.float32, copy=False)
    geometry_features = as_array(geometry_payload, "features", geometry_path).astype(np.float32, copy=False)

    if base_features.ndim != 3 or geometry_features.ndim != 3:
        raise ValueError(
            f"{base_path.name}: expected 3D features, got {base_features.shape} and {geometry_features.shape}"
        )
    if base_features.shape[:2] != geometry_features.shape[:2]:
        raise ValueError(
            f"{base_path.name}: feature alignment mismatch, "
            f"base={base_features.shape}, geometry={geometry_features.shape}"
        )

    for key in ("labels", "video_names", "approx_timestamps"):
        check_optional_equal(base_payload, geometry_payload, key, base_path.name)

    fused_features = np.concatenate([base_features, geometry_features], axis=-1).astype(np.float32)
    fused_payload = dict(base_payload)
    fused_payload["features"] = fused_features
    fused_payload["feature_components"] = {
        "clip_gesture_dim": int(base_features.shape[-1]),
        "hand_geometry_dim": int(geometry_features.shape[-1]),
        "fused_dim": int(fused_features.shape[-1]),
    }
    fused_payload["source_feature_dirs"] = {
        "clip_gesture": str(base_path.parent),
        "hand_geometry": str(geometry_path.parent),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, fused_payload)
    return int(fused_features.shape[0]), int(fused_features.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate CLIP gesture features with hand geometry features.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--geometry-dir", default=str(DEFAULT_GEOMETRY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    geometry_dir = Path(args.geometry_dir)
    output_dir = Path(args.output_dir)

    base_files = sorted(base_dir.glob(FEATURE_PATTERN))
    if not base_files:
        raise SystemExit(f"No base gesture feature files found under: {base_dir}")

    total_segments = 0
    fused_dim = None
    for base_path in base_files:
        geometry_path = geometry_dir / base_path.name
        if not geometry_path.exists():
            raise FileNotFoundError(f"Missing geometry feature file: {geometry_path}")
        output_path = output_dir / base_path.name
        segments, dim = fuse_one(base_path, geometry_path, output_path, overwrite=args.overwrite)
        total_segments += segments
        fused_dim = dim
        print(f"[saved] {output_path.name} segments={segments} dim={dim}")

    print(
        f"[done] files={len(base_files)} total_segments={total_segments} "
        f"fused_dim={fused_dim} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
