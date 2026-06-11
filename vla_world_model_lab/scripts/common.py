from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAB_ROOT.parent
ARTIFACT_ROOT = LAB_ROOT / "artifacts"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from project_paths import PROCESSED_DATA_DIR  # noqa: E402


MODALITIES = ("imu", "gesture", "audio", "text", "scene")
INTENT_NAMES = {
    0: "menu",
    1: "select",
    2: "magnify",
    3: "narrow",
    4: "brush",
    5: "cancel",
}
VIDEO_LABELS = {
    "interaction_20260306_072344.mp4": 0,
    "interaction_20260227_122606.mp4": 1,
    "interaction_20260227_122952.mp4": 2,
    "interaction_20260227_123354.mp4": 3,
    "interaction_20260227_124559.mp4": 4,
    "interaction_20260227_123745.mp4": 5,
    "interaction_20260131_120024.mp4": 0,
    "interaction_20260227_132951.mp4": 1,
    "interaction_20260227_133408.mp4": 2,
    "interaction_20260131_114156.mp4": 3,
    "interaction_20260131_115150.mp4": 4,
    "interaction_20260131_114852.mp4": 5,
    "interaction_20260301_073041.mp4": 0,
    "interaction_20260301_064753.mp4": 1,
    "interaction_20260306_072721.mp4": 2,
    "interaction_20260301_071948.mp4": 3,
    "interaction_20260131_121548.mp4": 3,
    "interaction_20260301_073435.mp4": 4,
    "interaction_20260301_072503.mp4": 5,
    "interaction_20260131_071552.mp4": 0,
    "interaction_20260131_072412.mp4": 1,
    "interaction_20260131_084300.mp4": 1,
    "interaction_20260131_085611.mp4": 2,
    "interaction_20260131_090139.mp4": 3,
    "interaction_20260131_085207.mp4": 4,
    "interaction_20260131_084732.mp4": 5,
    "interaction_20260131_090917.mp4": 0,
    "interaction_20260131_090541.mp4": 1,
    "interaction_20260131_065459.mp4": 2,
    "interaction_20260131_070722.mp4": 3,
    "interaction_20260131_091657.mp4": 4,
    "interaction_20260131_091249.mp4": 5,
    "interaction_20260306_082346.mp4": 2,
    "interaction_20260306_083107.mp4": 3,
    "interaction_20260306_083434.mp4": 1,
    "interaction_20260306_084406.mp4": 0,
    "interaction_20260306_084853.mp4": 5,
    "interaction_20260306_085830.mp4": 4,
    "interaction_20260306_090441.mp4": 1,
}
TEST_VIDEO_NAMES = [
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
]
TRAIN_VIDEO_NAMES = [
    video_name for video_name in VIDEO_LABELS if video_name not in TEST_VIDEO_NAMES
]
OFFICE_VIDEO_NAMES = {
    "interaction_20260306_072344.mp4",
    "interaction_20260227_122606.mp4",
    "interaction_20260227_122952.mp4",
    "interaction_20260227_123354.mp4",
    "interaction_20260227_124559.mp4",
    "interaction_20260227_123745.mp4",
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
}
SCENE_BY_VIDEO = {video_name: "office" for video_name in OFFICE_VIDEO_NAMES}
SCENE_BY_VIDEO.update({video_name: "museum" for video_name in set(VIDEO_LABELS) - OFFICE_VIDEO_NAMES})


def infer_user(video_name: str) -> str:
    if video_name in TEST_VIDEO_NAMES:
        return "Bian"
    if video_name in {
        "interaction_20260131_120024.mp4",
        "interaction_20260131_114156.mp4",
        "interaction_20260131_115150.mp4",
        "interaction_20260131_114852.mp4",
        "interaction_20260131_071552.mp4",
        "interaction_20260131_072412.mp4",
        "interaction_20260131_084300.mp4",
        "interaction_20260131_084732.mp4",
        "interaction_20260131_085207.mp4",
        "interaction_20260131_085611.mp4",
        "interaction_20260131_090139.mp4",
    }:
        return "Luo"
    return "Gu"


def split_for_video(video_name: str) -> str:
    return "test" if video_name in TEST_VIDEO_NAMES else "train"


def stem(video_name: str) -> str:
    return Path(video_name).stem


def feature_paths(video_name: str) -> dict[str, Path]:
    name = stem(video_name)
    return {
        "timestamp": PROCESSED_DATA_DIR / f"features_timestamp_{name}.npy",
        "segments": PROCESSED_DATA_DIR / f"segments_info_{name}.json",
        "gesture": PROCESSED_DATA_DIR / "strong_gesture_features" / f"strong_gesture_features_{name}.npy",
        "audio": PROCESSED_DATA_DIR / "audio_features" / f"audio_features_{name}.npy",
        "text": PROCESSED_DATA_DIR / "text_features" / f"text_features_{name}.npy",
        "imu": PROCESSED_DATA_DIR / "imu_features" / f"imu_features_{name}.npy",
    }


def safe_len_npy(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        array = np.load(path, allow_pickle=True, mmap_mode=None)
        return int(len(array))
    except Exception:
        return None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def require_existing(path: Path, message: str) -> None:
    if not path.exists():
        raise SystemExit(f"{message}: {path}")
