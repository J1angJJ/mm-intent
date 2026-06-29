from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from project_paths import FISHEYE_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from video_selection import filter_avi_mp4_items


SEQ_LEN = 10
HALF_WINDOW_MS = 750
FEATURE_DIM = 96
DEFAULT_TASK_MODEL_URLS = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
)
EXPECTED_TASK_MODEL_SHA256 = {
    DEFAULT_TASK_MODEL_URLS[0]: "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
}
MIN_TASK_MODEL_BYTES = 1024 * 1024


def load_avi_to_mp4_map() -> dict[str, str]:
    source_path = Path(__file__).with_name("strong_gesture2.0.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "AVI_TO_MP4_MAP":
                    return ast.literal_eval(node.value)
    raise RuntimeError("Cannot find AVI_TO_MP4_MAP in strong_gesture2.0.py")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_task_model(task_path: Path, model_urls: Sequence[str]) -> Path | None:
    task_path = task_path.resolve()
    task_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = task_path.with_suffix(task_path.suffix + ".download")
    errors: list[str] = []
    for url in model_urls:
        try:
            print(f"[hand] downloading MediaPipe hand landmarker: {url}")
            with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as file:
                shutil.copyfileobj(response, file)
            size = tmp_path.stat().st_size
            if size < MIN_TASK_MODEL_BYTES:
                raise RuntimeError(f"downloaded model is too small: {size} bytes")
            sha256 = file_sha256(tmp_path)
            expected_sha256 = EXPECTED_TASK_MODEL_SHA256.get(url)
            if expected_sha256 and sha256.lower() != expected_sha256:
                raise RuntimeError(f"unexpected model sha256: {sha256}")
            tmp_path.replace(task_path)
            print(f"[hand] saved MediaPipe hand landmarker: {task_path} ({size} bytes, sha256={sha256})")
            return task_path
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if tmp_path.exists():
                tmp_path.unlink()
    print("[warn] MediaPipe Tasks model download failed:")
    for error in errors:
        print(f"       {error}")
    return None


def legacy_hands_module():
    import mediapipe as mp

    try:
        return mp.solutions.hands
    except AttributeError:
        try:
            from mediapipe.python.solutions import hands as mp_hands

            return mp_hands
        except ModuleNotFoundError:
            return None


def create_landmark_detector(task_path: Path, *, auto_download: bool = True, model_urls: Sequence[str] = DEFAULT_TASK_MODEL_URLS):
    import mediapipe as mp

    task_path = task_path.resolve()
    if not task_path.exists() and auto_download:
        download_task_model(task_path, model_urls)

    task_error: Exception | None = None
    if task_path.exists():
        try:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3,
            )
            print(f"[hand] MediaPipe Tasks: {task_path}")
            return "tasks", mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            task_error = exc
            print(f"[warn] MediaPipe Tasks failed, fallback to legacy: {exc}")
    else:
        print(f"[warn] MediaPipe Tasks model not found: {task_path}")

    mp_hands = legacy_hands_module()
    if mp_hands is None:
        details = [
            "No MediaPipe hand detector is available.",
            f"Tasks model path does not exist or cannot be used: {task_path}.",
            "The installed mediapipe package does not expose legacy Hands via mp.solutions or mediapipe.python.solutions.",
        ]
        if task_error is not None:
            details.append(f"Tasks error: {task_error}")
        details.append("Fix by allowing the default model download, or pass --task-model path/to/hand_landmarker.task.")
        raise RuntimeError(" ".join(details))

    try:
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
        print("[hand] MediaPipe legacy Hands")
        return "legacy", hands
    except Exception as exc:
        raise RuntimeError(f"No MediaPipe hand detector is available: {exc}") from exc


def detect_landmarks(detector_kind: str, detector, frame_bgr: np.ndarray):
    import cv2
    import mediapipe as mp

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if detector_kind == "tasks":
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)
        return result.hand_landmarks or []
    result = detector.process(frame_rgb)
    if not result.multi_hand_landmarks:
        return []
    return [hand.landmark for hand in result.multi_hand_landmarks]


def close_landmark_detector(detector) -> None:
    close = getattr(detector, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            print(f"[warn] MediaPipe detector close failed: {exc}")


def hand_feature_from_landmarks(hand_landmarks) -> np.ndarray:
    if not hand_landmarks:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    # Pick the hand with the largest 2D bounding box.
    best = None
    best_area = -1.0
    for hand in hand_landmarks:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
        width = float(coords[:, 0].max() - coords[:, 0].min())
        height = float(coords[:, 1].max() - coords[:, 1].min())
        area = width * height
        if area > best_area:
            best_area = area
            best = coords
    assert best is not None
    coords = best
    wrist = coords[0].copy()
    width = float(coords[:, 0].max() - coords[:, 0].min())
    height = float(coords[:, 1].max() - coords[:, 1].min())
    scale = max(width, height, 1e-4)
    rel = coords.copy()
    rel[:, 0] = (rel[:, 0] - wrist[0]) / scale
    rel[:, 1] = (rel[:, 1] - wrist[1]) / scale
    rel[:, 2] = rel[:, 2] - wrist[2]

    center = coords[:, :2].mean(axis=0)
    fingertips = coords[[4, 8, 12, 16, 20], :2]
    wrist_xy = coords[0, :2]
    fingertip_dist = np.linalg.norm(fingertips - wrist_xy[None, :], axis=1) / scale
    pinch_pairs = [(4, 8), (4, 12), (4, 16), (4, 20)]
    pinch_dist = np.array(
        [np.linalg.norm(coords[a, :2] - coords[b, :2]) / scale for a, b in pinch_pairs],
        dtype=np.float32,
    )
    palm_width = np.linalg.norm(coords[5, :2] - coords[17, :2]) / scale
    hand_height = np.linalg.norm(coords[0, :2] - coords[12, :2]) / scale
    aspect = width / max(height, 1e-4)
    meta = np.array(
        [
            1.0,
            center[0],
            center[1],
            width,
            height,
            aspect,
            palm_width,
            hand_height,
            float(len(hand_landmarks)),
        ],
        dtype=np.float32,
    )
    feature = np.concatenate([meta, rel.reshape(-1), fingertip_dist.astype(np.float32), pinch_dist])
    padded = np.zeros(FEATURE_DIM, dtype=np.float32)
    padded[: min(FEATURE_DIM, len(feature))] = feature[:FEATURE_DIM]
    return padded


def get_avi_sync_ms(avi_path: Path, utc_target: datetime) -> float | None:
    import cv2

    avi_fn = avi_path.name
    time_part = avi_fn.split("_")[1] + "_" + avi_fn.split("_")[2].split(".")[0]
    avi_utc_start = datetime.strptime(time_part, "%Y%m%d_%H%M%S%f") - timedelta(hours=8)
    diff_ms = (utc_target - avi_utc_start).total_seconds() * 1000
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return None
    dur_ms = frames / fps * 1000
    return diff_ms if 0 <= diff_ms <= dur_ms else None


def extract_sequence(video_path: Path, center_ms: float, detector_kind: str, detector) -> np.ndarray | None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0:
        cap.release()
        return None
    total_ms = frames / fps * 1000
    start_ms = center_ms - HALF_WINDOW_MS
    end_ms = center_ms + HALF_WINDOW_MS
    if start_ms < 0 or end_ms > total_ms:
        cap.release()
        return None

    features = []
    for msec in np.linspace(start_ms, end_ms, SEQ_LEN):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(msec))
        ok, frame = cap.read()
        if not ok:
            break
        landmarks = detect_landmarks(detector_kind, detector, frame)
        features.append(hand_feature_from_landmarks(landmarks))
    cap.release()
    if len(features) != SEQ_LEN:
        return None
    return np.stack(features, axis=0).astype(np.float32)


def main() -> None:
    global HALF_WINDOW_MS, SEQ_LEN

    parser = argparse.ArgumentParser(description="Extract MediaPipe hand geometry time-series features.")
    parser.add_argument("--output-dir", default=str(PROCESSED_DATA_DIR / "hand_geometry_features"))
    parser.add_argument(
        "--task-model",
        default=os.getenv("MM_INTENT_HAND_LANDMARKER_TASK", str(PROJECT_ROOT / "models" / "hand_landmarker.task")),
    )
    parser.add_argument(
        "--no-download-task-model",
        action="store_true",
        help="Do not download the default MediaPipe Tasks hand model when --task-model is missing.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--half-window-ms", type=int, default=HALF_WINDOW_MS)
    args = parser.parse_args()
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be positive.")
    if args.half_window_ms <= 0:
        raise SystemExit("--half-window-ms must be positive.")
    SEQ_LEN = args.seq_len
    HALF_WINDOW_MS = args.half_window_ms

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_kind, detector = create_landmark_detector(Path(args.task_model), auto_download=not args.no_download_task_model)
    mapping = load_avi_to_mp4_map()
    items = filter_avi_mp4_items(mapping.items())
    if args.limit > 0:
        items = items[: args.limit]

    print(
        f"[hand-geometry] videos={len(items)} output={output_dir} "
        f"seq_len={SEQ_LEN} half_window_ms={HALF_WINDOW_MS} dim={FEATURE_DIM}"
    )
    try:
        for avi_name, mp4_name in items:
            mp4_base = Path(mp4_name).stem
            metadata_path = PROCESSED_DATA_DIR / f"metadata_strong_gesture_{mp4_base}.npy"
            video_path = FISHEYE_DIR / avi_name
            if not metadata_path.exists() or not video_path.exists():
                print(f"[skip] missing metadata/video: {mp4_name}")
                continue
            metadata = np.load(metadata_path, allow_pickle=True).item()
            timestamps = metadata["approx_timestamps"]
            labels = metadata["labels"]
            valid_feats, valid_labels, valid_tss = [], [], []
            debug: dict[str, object] = {}
            print(f"\n>>> {avi_name} -> {mp4_name}")
            for index, ts_str in tqdm(list(enumerate(timestamps)), total=len(timestamps)):
                try:
                    utc_dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00")).replace(tzinfo=None)
                    center_ms = get_avi_sync_ms(video_path, utc_dt)
                    if center_ms is None:
                        continue
                    sequence = extract_sequence(video_path, center_ms, detector_kind, detector)
                    if sequence is None:
                        continue
                    valid_feats.append(sequence)
                    valid_labels.append(labels[index])
                    valid_tss.append(ts_str)
                    debug[str(len(valid_feats) - 1)] = {"source_index": index, "timestamp": str(ts_str), "center_ms": center_ms}
                except Exception as exc:
                    print(f"[warn] segment {index}: {exc}")
            if not valid_feats:
                print("[warn] no valid hand geometry sequences")
                continue
            payload = {
                "features": np.stack(valid_feats, axis=0).astype(np.float32),
                "labels": np.asarray(valid_labels),
                "video_names": np.asarray([mp4_name] * len(valid_labels)),
                "approx_timestamps": valid_tss,
            }
            np.save(output_dir / f"strong_gesture_features_{mp4_base}.npy", payload)
            with (output_dir / f"hand_geometry_debug_{mp4_base}.json").open("w", encoding="utf-8") as file:
                json.dump(debug, file, indent=2, ensure_ascii=False)
            print(f"[saved] {mp4_name} {payload['features'].shape}")
    finally:
        close_landmark_detector(detector)


if __name__ == "__main__":
    main()
