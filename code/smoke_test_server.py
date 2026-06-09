from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

import baseline_real_scene as base
from project_paths import DATASET_DIR, FISHEYE_DIR, HOLOLENS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from real_scene_utils import MP4_TO_AVI_MAP, get_scene_backbone, resolve_avi_path


def summarize_dir(path: Path, pattern: str) -> None:
    files = sorted(path.glob(pattern)) if path.exists() else []
    total_bytes = sum(file.stat().st_size for file in files if file.is_file())
    print(f"{path}: {len(files)} files, {total_bytes / 1024**3:.2f} GiB")
    for file in files[:5]:
        print(f"  - {file.name} ({file.stat().st_size / 1024**2:.1f} MiB)")
    if len(files) > 5:
        print(f"  ... {len(files) - 5} more")


def check_video(path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[fail] cannot open video: {path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    print(
        f"[video] {path.name}: {width}x{height}, "
        f"fps={fps:.2f}, frames={frame_count}, duration={duration:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-backbone",
        action="store_true",
        help="Load the ViT scene backbone. This may download model weights on first run.",
    )
    args = parser.parse_args()

    print("[paths]")
    print(f"project_root       {PROJECT_ROOT}")
    print(f"dataset_dir        {DATASET_DIR}")
    print(f"fisheye_dir        {FISHEYE_DIR}")
    print(f"hololens_dir       {HOLOLENS_DIR}")
    print(f"processed_data_dir {PROCESSED_DATA_DIR}")
    print(f"output_dir         {base.MODEL_OUTPUT_DIR}")

    print("[device]")
    print(f"torch              {torch.__version__}")
    print(f"cuda_available     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu                {torch.cuda.get_device_name(0)}")
        print(f"capability         {torch.cuda.get_device_capability(0)}")

    print("[dataset]")
    summarize_dir(FISHEYE_DIR, "*.avi")
    summarize_dir(HOLOLENS_DIR, "*.mp4")

    first_fisheye = next(iter(sorted(FISHEYE_DIR.glob("*.avi"))), None)
    first_hololens = next(iter(sorted(HOLOLENS_DIR.glob("*.mp4"))), None)
    if first_fisheye is not None:
        check_video(first_fisheye)
    if first_hololens is not None:
        check_video(first_hololens)

    print("[mapping]")
    available_mapped = []
    missing_mapped = []
    for mp4_name, avi_name in MP4_TO_AVI_MAP.items():
        avi_path = FISHEYE_DIR / avi_name
        if avi_path.exists():
            available_mapped.append(mp4_name)
        else:
            missing_mapped.append(mp4_name)
    print(f"mapped fisheye present {len(available_mapped)} / {len(MP4_TO_AVI_MAP)}")
    print(f"first present          {available_mapped[:5]}")
    print(f"first missing          {missing_mapped[:5]}")
    if first_hololens is not None and first_hololens.name in MP4_TO_AVI_MAP:
        print(f"resolved avi for {first_hololens.name}: {resolve_avi_path(first_hololens.name)}")

    print("[features]")
    feature_dirs = [
        base.STRONG_GESTURE_DIR,
        base.AUDIO_FEAT_DIR,
        base.TEXT_FEAT_DIR,
        base.IMU_FEAT_DIR,
    ]
    for feature_dir in feature_dirs:
        count = len(list(feature_dir.glob("*.npy"))) if feature_dir.exists() else 0
        print(f"{feature_dir}: {count} npy files")

    if args.check_backbone:
        print("[backbone]")
        processor, model = get_scene_backbone()
        print(f"processor {processor.__class__.__name__}")
        print(f"model     {model.__class__.__name__}")

    print("[ok] smoke test finished")


if __name__ == "__main__":
    main()
