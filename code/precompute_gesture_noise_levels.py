from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from raw_data_utils import add_image_pixel_noise_at_level
from raw_pipeline import (
    COURSE_VIDEO_NAMES,
    build_manifest,
    missing_feature_paths,
    seed_unchanged_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def level_name(level: float) -> str:
    return str(int(round(level * 100)))


def condition_dir(output_root: Path, level: float, representation: str, seed: int) -> Path:
    suffix = "_hand_geometry" if representation == "hand_geometry" else ""
    return output_root / f"gesture_noise_{level_name(level)}_seed{seed}{suffix}"


def link_scene_cache(clean_scene_cache: Path, destination: Path) -> None:
    if not clean_scene_cache.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for source in clean_scene_cache.glob("*.npy"):
        target = destination / source.name
        if target.exists():
            continue
        try:
            os.link(source, target)
        except OSError:
            import shutil

            shutil.copy2(source, target)


def video_center_ms(avi_name: str, timestamp_value: str) -> float:
    time_part = avi_name.split("_")[1] + "_" + avi_name.split("_")[2].split(".")[0]
    avi_start = datetime.strptime(time_part, "%Y%m%d_%H%M%S%f") - timedelta(hours=8)
    target = datetime.fromisoformat(str(timestamp_value).replace("Z", "+00:00")).replace(tzinfo=None)
    return (target - avi_start).total_seconds() * 1000.0


def prepare_condition_roots(
    output_root: Path,
    base_feature_dir: Path,
    clean_scene_cache: Path,
    video_names: list[str],
    levels: list[float],
    representation: str,
    seed: int,
) -> dict[float, Path]:
    roots: dict[float, Path] = {}
    for level in levels:
        root = condition_dir(output_root, level, representation, seed)
        root.mkdir(parents=True, exist_ok=True)
        seed_unchanged_features(root, base_feature_dir, video_names, "gesture", representation)
        link_scene_cache(clean_scene_cache, root / "scene_features")
        roots[level] = root
    return roots


def clip_frame_features(
    module: ModuleType,
    image: Image.Image,
    video_name: str,
    msec: float,
    levels: list[float],
    seed: int,
) -> dict[float, np.ndarray]:
    crops = []
    for level in levels:
        noisy = add_image_pixel_noise_at_level(
            image,
            "gesture",
            f"{video_name}|{msec:.3f}",
            level,
            seed,
        )
        crops.append(module.crop_hand(noisy, msec).convert("RGB"))

    inputs = module.clip_processor(images=crops, return_tensors="pt").to(module.device)
    with torch.no_grad():
        outputs = module.clip_vision(**inputs)
    features = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
    return {level: features[index] for index, level in enumerate(levels)}


def geometry_frame_features(
    module: ModuleType,
    detector_kind: str,
    detector,
    image: Image.Image,
    video_name: str,
    msec: float,
    levels: list[float],
    seed: int,
) -> dict[float, np.ndarray]:
    result: dict[float, np.ndarray] = {}
    for level in levels:
        noisy = add_image_pixel_noise_at_level(
            image,
            "gesture",
            f"{video_name}|{msec:.3f}",
            level,
            seed,
        )
        frame = cv2.cvtColor(np.asarray(noisy), cv2.COLOR_RGB2BGR)
        landmarks = module.detect_landmarks(detector_kind, detector, frame)
        result[level] = module.hand_feature_from_landmarks(landmarks)
    return result


def output_path(root: Path, video_name: str, representation: str) -> Path:
    subdir = "hand_geometry_features" if representation == "hand_geometry" else "strong_gesture_features"
    return root / subdir / f"strong_gesture_features_{Path(video_name).stem}.npy"


def valid_feature_file(path: Path, expected_dim: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = np.load(path, allow_pickle=True).item()
        features = np.asarray(payload["features"])
        return features.ndim == 3 and features.shape[1:] == (10, expected_dim) and len(features) > 0
    except Exception:
        return False


def process_video(
    *,
    avi_name: str,
    video_name: str,
    video_path: Path,
    metadata_path: Path,
    roots: dict[float, Path],
    representation: str,
    seed: int,
    module: ModuleType,
    detector_kind: str | None,
    detector,
    force: bool,
) -> None:
    expected_dim = 96 if representation == "hand_geometry" else 768
    needed_levels = [
        level
        for level, root in roots.items()
        if force or not valid_feature_file(output_path(root, video_name, representation), expected_dim)
    ]
    if not needed_levels:
        print(f"[skip] complete: {video_name}")
        return

    metadata = np.load(metadata_path, allow_pickle=True).item()
    timestamps = metadata["approx_timestamps"]
    labels = metadata["labels"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_ms = frame_count / fps * 1000.0

    level_features: dict[float, list[np.ndarray]] = {level: [] for level in needed_levels}
    valid_labels: list[int] = []
    valid_timestamps: list[str] = []
    for index, timestamp_value in tqdm(
        list(enumerate(timestamps)),
        total=len(timestamps),
        desc=f"{representation}:{video_name}",
    ):
        center_ms = video_center_ms(avi_name, str(timestamp_value))
        offsets = np.linspace(center_ms - 750.0, center_ms + 750.0, 10)
        if offsets[0] < 0.0 or offsets[-1] > total_ms:
            continue

        sequences: dict[float, list[np.ndarray]] = {level: [] for level in needed_levels}
        valid = True
        for msec in offsets:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(msec))
            ok, frame = cap.read()
            if not ok:
                valid = False
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if representation == "clip":
                frame_features = clip_frame_features(
                    module,
                    image,
                    avi_name,
                    float(msec),
                    needed_levels,
                    seed,
                )
            else:
                assert detector_kind is not None
                frame_features = geometry_frame_features(
                    module,
                    detector_kind,
                    detector,
                    image,
                    avi_name,
                    float(msec),
                    needed_levels,
                    seed,
                )
            for level in needed_levels:
                sequences[level].append(frame_features[level])

        if not valid:
            continue
        for level in needed_levels:
            level_features[level].append(np.stack(sequences[level], axis=0).astype(np.float32))
        valid_labels.append(int(labels[index]))
        valid_timestamps.append(str(timestamp_value))
    cap.release()

    if not valid_labels:
        raise RuntimeError(f"No valid gesture samples extracted for {video_name}")
    for level in needed_levels:
        path = output_path(roots[level], video_name, representation)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features": np.stack(level_features[level], axis=0).astype(np.float32),
            "labels": np.asarray(valid_labels),
            "video_names": np.asarray([video_name] * len(valid_labels)),
            "approx_timestamps": valid_timestamps,
        }
        np.save(path, payload)
        if representation == "clip":
            metadata_output = roots[level] / f"metadata_strong_gesture_{Path(video_name).stem}.npy"
            np.save(metadata_output, {key: value for key, value in payload.items() if key != "features"})
        print(f"[saved] level={level_name(level)} {video_name} {payload['features'].shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode gesture frames once and precompute multiple raw-noise levels."
    )
    parser.add_argument("--representation", choices=("clip", "hand_geometry"), default="clip")
    parser.add_argument("--levels", nargs="+", type=float, default=(0.2, 0.4, 0.6))
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs" / "raw_feature_cache"))
    parser.add_argument(
        "--base-feature-dir",
        default=str(PROJECT_ROOT / "dataset" / "AR_Data_Process3.0" / "data_full"),
    )
    parser.add_argument(
        "--clean-scene-cache",
        default=str(PROJECT_ROOT / "outputs" / "raw_feature_cache" / "clean_seed42" / "scene_features"),
    )
    parser.add_argument("--fisheye-dir", default=str(PROJECT_ROOT / "dataset" / "fisheye"))
    parser.add_argument("--video-names", nargs="*", default=list(COURSE_VIDEO_NAMES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    levels = sorted(set(args.levels))
    if any(level <= 0.0 or level > 1.0 for level in levels):
        parser.error("Every noise level must be in (0, 1].")
    video_names = args.video_names[: args.limit] if args.limit > 0 else args.video_names
    base_feature_dir = Path(args.base_feature_dir).resolve()
    output_root = Path(args.output_root).resolve()
    clean_scene_cache = Path(args.clean_scene_cache).resolve()
    fisheye_dir = Path(args.fisheye_dir).resolve()

    if args.representation == "clip":
        module = load_module(
            PROJECT_ROOT / "code" / "feature_extraction" / "strong_gesture2.0.py",
            "mm_intent_strong_gesture_batch",
        )
        mapping = module.AVI_TO_MP4_MAP
        detector_kind = None
        detector = None
    else:
        module = load_module(
            PROJECT_ROOT / "code" / "feature_extraction" / "extract_hand_geometry_features.py",
            "mm_intent_hand_geometry_batch",
        )
        mapping = module.load_avi_to_mp4_map()
        detector_kind, detector = module.create_landmark_detector(
            PROJECT_ROOT / "models" / "hand_landmarker.task"
        )

    selected = set(video_names)
    items = [(avi, mp4) for avi, mp4 in mapping.items() if mp4 in selected]
    unknown = sorted(selected - {mp4 for _avi, mp4 in items})
    if unknown:
        raise ValueError(f"Unknown video names: {unknown}")

    roots = prepare_condition_roots(
        output_root,
        base_feature_dir,
        clean_scene_cache,
        video_names,
        levels,
        args.representation,
        args.noise_seed,
    )
    print(
        f"[gesture-batch] representation={args.representation} videos={len(items)} "
        f"levels={levels} seed={args.noise_seed}"
    )
    for avi_name, video_name in items:
        metadata_path = base_feature_dir / f"metadata_strong_gesture_{Path(video_name).stem}.npy"
        process_video(
            avi_name=avi_name,
            video_name=video_name,
            video_path=fisheye_dir / avi_name,
            metadata_path=metadata_path,
            roots=roots,
            representation=args.representation,
            seed=args.noise_seed,
            module=module,
            detector_kind=detector_kind,
            detector=detector,
            force=args.force,
        )

    env = os.environ.copy()
    for level, root in roots.items():
        missing = missing_feature_paths(root, video_names, args.representation)
        if missing:
            examples = "\n".join(f"  - {path}" for path in missing[:10])
            raise RuntimeError(f"Noise cache level {level} is incomplete ({len(missing)} files).\n{examples}")
        manifest = build_manifest(
            root,
            video_names,
            "gesture",
            level,
            args.noise_seed,
            env,
            base_feature_dir,
            args.representation,
        )
        (root / "raw_preprocessing_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[complete] {root}")


if __name__ == "__main__":
    main()
