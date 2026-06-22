from __future__ import annotations

import hashlib
import os
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


RAW_MODALITIES = ("imu", "gesture", "audio", "text", "scene")


def raw_noise_modality() -> str:
    modality = os.getenv("MM_INTENT_RAW_NOISE_MODALITY", "").strip().lower()
    if modality and modality not in RAW_MODALITIES:
        raise ValueError(f"Unknown raw-noise modality: {modality}")
    return modality


def raw_noise_level() -> float:
    level = float(os.getenv("MM_INTENT_RAW_NOISE_LEVEL", "0.0"))
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"MM_INTENT_RAW_NOISE_LEVEL must be in [0, 1], got {level}")
    return level


def raw_noise_seed() -> int:
    return int(os.getenv("MM_INTENT_RAW_NOISE_SEED", "42"))


def raw_noise_enabled(modality: str) -> bool:
    return raw_noise_modality() == modality and raw_noise_level() > 0.0


def select_video_names(default_names: Iterable[str]) -> list[str]:
    available = list(default_names)
    raw_value = os.getenv("MM_INTENT_VIDEO_NAMES", "").strip()
    if not raw_value:
        return available

    selected = [item.strip() for item in raw_value.replace(";", ",").split(",") if item.strip()]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown MM_INTENT_VIDEO_NAMES entries: {unknown}")
    selected_set = set(selected)
    return [name for name in available if name in selected_set]


def _stable_rng(modality: str, sample_key: str, seed: int | None = None) -> np.random.Generator:
    effective_seed = raw_noise_seed() if seed is None else seed
    payload = f"{effective_seed}|{modality}|{sample_key}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="little", signed=False))


def add_audio_waveform_noise(audio: np.ndarray, sample_key: str) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if not raw_noise_enabled("audio") or values.size == 0:
        return values

    rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    sigma = raw_noise_level() * max(rms, 1e-6)
    noise = _stable_rng("audio", sample_key).normal(0.0, sigma, size=values.shape)
    return np.clip(values + noise.astype(np.float32), -1.0, 1.0).astype(np.float32)


def add_image_pixel_noise(image: Image.Image, modality: str, sample_key: str) -> Image.Image:
    if modality not in ("gesture", "scene"):
        raise ValueError(f"Image noise is not defined for modality: {modality}")
    if not raw_noise_enabled(modality):
        return image

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    sigma = raw_noise_level() * 255.0
    noise = _stable_rng(modality, sample_key).normal(0.0, sigma, size=pixels.shape)
    noisy = np.clip(pixels + noise.astype(np.float32), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def add_image_pixel_noise_at_level(
    image: Image.Image,
    modality: str,
    sample_key: str,
    level: float,
    seed: int,
) -> Image.Image:
    if modality not in ("gesture", "scene"):
        raise ValueError(f"Image noise is not defined for modality: {modality}")
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"Noise level must be in [0, 1], got {level}")
    if level == 0.0:
        return image

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    unit_noise = _stable_rng(modality, sample_key, seed).normal(0.0, 1.0, size=pixels.shape)
    noisy = np.clip(pixels + unit_noise.astype(np.float32) * (level * 255.0), 0.0, 255.0)
    return Image.fromarray(noisy.astype(np.uint8), mode="RGB")


def add_imu_channel_noise(values: np.ndarray, column_names: Sequence[str]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not raw_noise_enabled("imu") or array.size == 0:
        return array
    if array.ndim != 2 or array.shape[1] != len(column_names):
        raise ValueError(
            f"Expected a 2-D IMU array with {len(column_names)} columns, got {array.shape}"
        )

    scale = np.std(array, axis=0, ddof=0)
    scale = np.maximum(scale, 1e-8) * raw_noise_level()
    noise = _stable_rng("imu", "|".join(column_names)).normal(0.0, scale, size=array.shape)
    return array + noise


def corrupt_text_characters(text: str, sample_key: str) -> str:
    if not text or not raw_noise_enabled("text"):
        return text

    characters = list(text)
    candidates = [index for index, value in enumerate(characters) if not value.isspace()]
    if not candidates:
        return text

    count = min(len(candidates), max(1, int(round(len(candidates) * raw_noise_level()))))
    selected = set(_stable_rng("text", sample_key).choice(candidates, size=count, replace=False).tolist())
    return "".join(value for index, value in enumerate(characters) if index not in selected)


def raw_noise_definition(modality: str, level: float) -> str:
    definitions = {
        "audio": "Gaussian noise on the 16 kHz waveform, sigma = level * waveform RMS, before MFCC extraction.",
        "gesture": "Gaussian RGB pixel noise, sigma = level * 255, before hand detection/cropping and CLIP encoding.",
        "scene": "Gaussian RGB pixel noise, sigma = level * 255, before ViT scene encoding.",
        "imu": "Channel-wise Gaussian noise, sigma = level * raw-channel standard deviation, before derived kinematics.",
        "text": "Deterministic deletion of level * non-whitespace transcript characters before text embedding.",
    }
    return definitions.get(modality, "No raw-modality noise.") if level > 0.0 else "No raw-modality noise."
