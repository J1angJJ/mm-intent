from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch

import baseline_real_scene as base
import train_and_test as improved
from real_scene_utils import RealSceneFeatureCache


def stack_payload_features(payload: Dict[str, object]) -> Dict[str, np.ndarray]:
    return {
        "imu": payload["imu"],  # type: ignore[return-value]
        "gesture": payload["gesture"],  # type: ignore[return-value]
        "audio": payload["audio"],  # type: ignore[return-value]
        "text": payload["text"],  # type: ignore[return-value]
        "scene": payload["scene"],  # type: ignore[return-value]
    }


def tensor_batch(features: Dict[str, np.ndarray], batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(value[:batch_size].astype(np.float32)).to(device)
        for key, value in features.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="interaction_20260131_065459.mp4")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print("[load]")
    print(f"video {args.video}")
    scene_cache = RealSceneFeatureCache(base.SCENE_CACHE_DIR)
    payload = base.load_aligned_video(args.video, scene_cache)
    if payload is None:
        raise RuntimeError(f"Failed to load aligned video: {args.video}")

    features = stack_payload_features(payload)
    sample_count = len(payload["labels"])  # type: ignore[arg-type]
    batch_size = min(args.batch_size, sample_count)
    print(f"samples {sample_count}, batch_size {batch_size}")
    for key, value in features.items():
        print(f"{key:8s} {value.shape} {value.dtype}")

    device = base.DEVICE
    print(f"[device] {device}")
    batch = tensor_batch(features, batch_size, device)

    print("[baseline]")
    baseline_model = base.PerceiverIOSceneBaseline(
        num_classes=len(base.ALL_JOINT_CLASS_NAMES),
        model_dim=base.MODEL_DIM,
        num_latents=base.NUM_LATENTS,
        depth=base.DEPTH,
        num_heads=base.NUM_HEADS,
        dropout=base.DROPOUT,
    ).to(device)
    baseline_model.eval()
    with torch.no_grad():
        logits = baseline_model(
            batch["imu"],
            batch["gesture"],
            batch["audio"],
            batch["text"],
            batch["scene"],
        )
    print(f"joint_logits {tuple(logits.shape)}")

    print("[improved]")
    improved_model = improved.Anchor2PerceiverIO(
        num_joint_classes=len(base.ALL_JOINT_CLASS_NAMES),
        num_intent_classes=len(base.INTENT_NAMES),
        num_scene_classes=len(base.SCENE_NAME_TO_ID),
        joint_class_names=base.ALL_JOINT_CLASS_NAMES.tolist(),
        model_dim=improved.MODEL_DIM,
        num_latents=improved.NUM_LATENTS,
        depth=improved.DEPTH,
        num_heads=improved.NUM_HEADS,
        dropout=improved.DROPOUT,
    ).to(device)
    improved_model.eval()
    with torch.no_grad():
        outputs = improved_model(
            batch["imu"],
            batch["gesture"],
            batch["audio"],
            batch["text"],
            batch["scene"],
        )
    for key, value in outputs.items():
        print(f"{key:20s} {tuple(value.shape)}")

    print("[ok] forward smoke test finished")


if __name__ == "__main__":
    main()
