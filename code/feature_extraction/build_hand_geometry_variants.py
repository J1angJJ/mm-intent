from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from project_paths import PROCESSED_DATA_DIR


FEATURE_DIM = 96
META = slice(0, 9)
REL_LANDMARKS = slice(9, 72)
FINGERTIP_DIST = slice(72, 77)
PINCH_DIST = slice(77, 81)


def first_order_delta(features: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(features)
    delta[:, 1:, :] = features[:, 1:, :] - features[:, :-1, :]
    return delta


def second_order_delta(features: np.ndarray) -> np.ndarray:
    velocity = first_order_delta(features)
    acceleration = np.zeros_like(features)
    acceleration[:, 1:, :] = velocity[:, 1:, :] - velocity[:, :-1, :]
    return acceleration


def keep_slices(features: np.ndarray, slices: list[slice]) -> np.ndarray:
    output = np.zeros_like(features)
    for group_slice in slices:
        output[:, :, group_slice] = features[:, :, group_slice]
    return output


def variant_original(features: np.ndarray) -> np.ndarray:
    return features


def variant_landmark_only(features: np.ndarray) -> np.ndarray:
    return keep_slices(features, [REL_LANDMARKS])


def variant_no_meta(features: np.ndarray) -> np.ndarray:
    output = features.copy()
    output[:, :, META] = 0.0
    return output


def variant_no_fingertip(features: np.ndarray) -> np.ndarray:
    output = features.copy()
    output[:, :, FINGERTIP_DIST] = 0.0
    return output


def variant_no_pinch(features: np.ndarray) -> np.ndarray:
    output = features.copy()
    output[:, :, PINCH_DIST] = 0.0
    return output


def variant_no_distances(features: np.ndarray) -> np.ndarray:
    output = features.copy()
    output[:, :, FINGERTIP_DIST] = 0.0
    output[:, :, PINCH_DIST] = 0.0
    return output


def variant_delta(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, first_order_delta(features)], axis=2).astype(np.float32)


def variant_delta_accel(features: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [features, first_order_delta(features), second_order_delta(features)],
        axis=2,
    ).astype(np.float32)


VARIANTS: dict[str, tuple[int, Callable[[np.ndarray], np.ndarray], str]] = {
    "original": (96, variant_original, "Original hand geometry feature."),
    "landmark_only": (96, variant_landmark_only, "Only wrist-relative 21x3 landmarks; meta and distance groups are zeroed."),
    "no_meta": (96, variant_no_meta, "Remove detection/box/meta geometry group."),
    "no_fingertip": (96, variant_no_fingertip, "Remove fingertip-to-wrist distance group."),
    "no_pinch": (96, variant_no_pinch, "Remove thumb-to-fingertip pinch distance group."),
    "no_distances": (96, variant_no_distances, "Remove fingertip and pinch distance groups."),
    "delta": (192, variant_delta, "Concatenate first-order temporal delta to original geometry."),
    "delta_accel": (288, variant_delta_accel, "Concatenate first- and second-order temporal deltas."),
}


def convert_file(input_path: Path, output_path: Path, transform: Callable[[np.ndarray], np.ndarray], overwrite: bool) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        payload = np.load(output_path, allow_pickle=True).item()
        return {
            "file": output_path.name,
            "status": "exists",
            "features_shape": list(np.asarray(payload["features"]).shape),
        }

    payload = np.load(input_path, allow_pickle=True).item()
    features = np.asarray(payload["features"], dtype=np.float32)
    output_payload = dict(payload)
    output_payload["features"] = transform(features).astype(np.float32)
    np.save(output_path, output_payload)
    return {
        "file": output_path.name,
        "status": "written",
        "input_shape": list(features.shape),
        "features_shape": list(output_payload["features"].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper ablation variants from cached hand-geometry features.")
    parser.add_argument("--input-dir", default=str(PROCESSED_DATA_DIR / "hand_geometry_features"))
    parser.add_argument("--output-root", default=str(PROCESSED_DATA_DIR / "hand_geometry_variants"))
    parser.add_argument("--variants", nargs="*", default=["landmark_only", "no_meta", "no_fingertip", "no_pinch", "no_distances", "delta"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    input_files = sorted(input_dir.glob("strong_gesture_features_*.npy"))
    if not input_files:
        raise SystemExit(f"No hand-geometry files found under: {input_dir}")

    for variant_name in args.variants:
        if variant_name not in VARIANTS:
            raise SystemExit(f"Unknown variant: {variant_name}. Choices: {', '.join(VARIANTS)}")
        feature_dim, transform, description = VARIANTS[variant_name]
        output_dir = output_root / variant_name
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "variant": variant_name,
            "feature_dim": feature_dim,
            "description": description,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "files": [],
        }
        print(f"[variant] {variant_name} dim={feature_dim} -> {output_dir}")
        for input_path in input_files:
            output_path = output_dir / input_path.name
            record = convert_file(input_path, output_path, transform, args.overwrite)
            manifest["files"].append(record)
            print(f"  {record['status']:7s} {record['file']} {record['features_shape']}")
        with (output_dir / "variant_manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
