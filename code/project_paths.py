from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(
    os.getenv("MM_INTENT_ROOT", Path(__file__).resolve().parents[1])
).resolve()

DATASET_DIR = Path(os.getenv("MM_INTENT_DATASET_DIR", PROJECT_ROOT / "dataset")).resolve()
FISHEYE_DIR = Path(os.getenv("MM_INTENT_FISHEYE_DIR", DATASET_DIR / "fisheye")).resolve()
HOLOLENS_DIR = Path(os.getenv("MM_INTENT_HOLOLENS_DIR", DATASET_DIR / "HoloLens")).resolve()

PROCESSED_DATA_DIR = Path(
    os.getenv("MM_INTENT_PROCESSED_DATA_DIR", DATASET_DIR / "AR_Data_Process3.0" / "data")
).resolve()
MODEL_OUTPUT_ROOT = Path(os.getenv("MM_INTENT_OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()

HF_CACHE_DIR = Path(os.getenv("HF_HOME", PROJECT_ROOT / ".cache" / "huggingface")).resolve()
HF_HUB_CACHE_DIR = Path(os.getenv("HF_HUB_CACHE", HF_CACHE_DIR / "hub")).resolve()
HF_TRANSFORMERS_CACHE_DIR = Path(
    os.getenv("TRANSFORMERS_CACHE", HF_CACHE_DIR / "transformers")
).resolve()

VIT_MODEL_NAME_OR_PATH = os.getenv(
    "MM_INTENT_VIT_MODEL",
    str(PROJECT_ROOT / "models" / "vit-base-patch16-224"),
)

CLIP_MODEL_NAME_OR_PATH = os.getenv(
    "MM_INTENT_CLIP_MODEL",
    str(PROJECT_ROOT / "models" / "clip_teacher_model"),
)

SENTENCE_MODEL_NAME_OR_PATH = os.getenv(
    "MM_INTENT_SENTENCE_MODEL",
    str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"),
)


def selected_video_names(default_names: Iterable[str]) -> list[str]:
    """Apply the optional per-run video filter used by raw inference benchmarks."""
    requested = {
        item.strip()
        for item in os.getenv("MM_INTENT_VIDEO_NAMES", "").replace(";", ",").split(",")
        if item.strip()
    }
    names = list(default_names)
    if not requested:
        return names
    selected = [name for name in names if name in requested]
    missing = requested - set(selected)
    if missing:
        raise ValueError(f"Unknown/unavailable video names: {sorted(missing)}")
    return selected


def configure_hf_cache() -> None:
    HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    HF_TRANSFORMERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_TRANSFORMERS_CACHE_DIR))
