from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from runtime_timing import timed_block, timing_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-level end-to-end smoke/benchmark for MM-Intent.")
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument("--video", default="interaction_20260131_065459.mp4")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs/batch_end_to_end")
    parser.add_argument("--gesture-feature-dir")
    parser.add_argument("--gesture-feature-dim", type=int)
    parser.add_argument("--hand-geometry", action="store_true", help="Use cached hand-geometry gesture features.")
    parser.add_argument(
        "--raw-hand-geometry",
        action="store_true",
        help="Extract hand geometry for this batch directly from fisheye video and replace cached gesture features.",
    )
    parser.add_argument(
        "--allow-raw-fallback",
        action="store_true",
        help="Keep cached gesture rows when raw hand geometry extraction fails for a sample.",
    )
    parser.add_argument("--hand-task-model")
    return parser.parse_args()


def configure_feature_env(args: argparse.Namespace) -> None:
    if args.hand_geometry or args.raw_hand_geometry:
        args.gesture_feature_dir = args.gesture_feature_dir or "dataset/AR_Data_Process3.0/data/hand_geometry_features"
        args.gesture_feature_dim = args.gesture_feature_dim or 96
    if args.gesture_feature_dir:
        os.environ["MM_INTENT_GESTURE_FEATURE_DIR"] = str(Path(args.gesture_feature_dir).resolve())
    if args.gesture_feature_dim is not None:
        os.environ["MM_INTENT_GESTURE_FEAT_DIM"] = str(args.gesture_feature_dim)


def take_batch(payload: dict[str, Any], batch_size: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    sample_count = len(payload["labels"])
    limit = min(batch_size, sample_count)
    features = {
        key: np.asarray(payload[key][:limit], dtype=np.float32)
        for key in ("imu", "gesture", "audio", "text", "scene")
    }
    labels = np.asarray(payload["labels"][:limit], dtype=np.int64)
    scene_targets = np.asarray(payload["scene_targets"][:limit], dtype=np.int64)
    timestamps = np.asarray(payload["approx_timestamps"][:limit], dtype=object)
    return features, labels, scene_targets, timestamps


def tensors(features: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    import torch

    return {
        key: torch.from_numpy(value.astype(np.float32)).to(device)
        for key, value in features.items()
    }


def extract_raw_hand_geometry_batch(
    video_name: str,
    timestamps: np.ndarray,
    task_model: str | None,
    allow_fallback: bool,
    cached_gesture: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    from feature_extraction import extract_hand_geometry_features as geometry
    from project_paths import FISHEYE_DIR, PROJECT_ROOT
    import cv2

    mp4_to_avi = {mp4: avi for avi, mp4 in geometry.load_avi_to_mp4_map().items()}
    if video_name not in mp4_to_avi:
        raise RuntimeError(f"No fisheye mapping for {video_name}")
    video_path = FISHEYE_DIR / mp4_to_avi[video_name]
    if not video_path.exists():
        raise RuntimeError(f"Missing fisheye video: {video_path}")

    task_path = Path(
        task_model
        or os.getenv("MM_INTENT_HAND_LANDMARKER_TASK", str(PROJECT_ROOT / "models" / "hand_landmarker.task"))
    )
    detector_kind, detector = geometry.create_landmark_detector(task_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_ms = frames / fps * 1000 if fps > 0 else 0.0

    output = np.asarray(cached_gesture, dtype=np.float32).copy()
    extracted = 0
    failed: list[int] = []
    try:
        for row, ts_value in enumerate(timestamps):
            utc_dt = datetime.fromisoformat(str(ts_value).replace("Z", "+00:00")).replace(tzinfo=None)
            center_ms = geometry.get_avi_sync_ms(video_path, utc_dt)
            if center_ms is None:
                failed.append(row)
                continue
            start_ms = center_ms - geometry.HALF_WINDOW_MS
            end_ms = center_ms + geometry.HALF_WINDOW_MS
            if fps <= 0 or start_ms < 0 or end_ms > total_ms:
                failed.append(row)
                continue
            sequence = []
            for msec in np.linspace(start_ms, end_ms, geometry.SEQ_LEN):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(msec))
                ok, frame = cap.read()
                if not ok:
                    break
                landmarks = geometry.detect_landmarks(detector_kind, detector, frame)
                sequence.append(geometry.hand_feature_from_landmarks(landmarks))
            if len(sequence) != geometry.SEQ_LEN:
                failed.append(row)
                continue
            output[row] = np.stack(sequence, axis=0).astype(np.float32)
            extracted += 1
    finally:
        cap.release()

    if failed and not allow_fallback:
        raise RuntimeError(f"Raw hand geometry failed for rows: {failed}")
    return output, {
        "video_path": str(video_path),
        "detector": detector_kind,
        "requested": int(len(timestamps)),
        "extracted": int(extracted),
        "fallback_rows": failed,
    }


def main() -> None:
    args = parse_args()
    configure_feature_env(args)

    import torch
    import torch.nn as nn

    import baseline_real_scene as base
    import train_and_test as improved
    from real_scene_utils import RealSceneFeatureCache

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with timed_block() as load_timer:
        scene_cache = RealSceneFeatureCache(base.SCENE_CACHE_DIR)
        payload = base.load_aligned_video(args.video, scene_cache)
    if payload is None:
        raise RuntimeError(f"Failed to load aligned video: {args.video}")

    features, intent_labels, scene_targets, timestamps = take_batch(payload, args.batch_size)
    raw_geometry_report: dict[str, Any] | None = None
    raw_geometry_seconds: float | None = None
    if args.raw_hand_geometry:
        with timed_block() as raw_timer:
            features["gesture"], raw_geometry_report = extract_raw_hand_geometry_batch(
                args.video,
                timestamps,
                args.hand_task_model,
                args.allow_raw_fallback,
                features["gesture"],
            )
        raw_geometry_seconds = raw_timer["seconds"]

    joint_raw = base.build_joint_labels(intent_labels, scene_targets)
    joint_class_names = base.ALL_JOINT_CLASS_NAMES.tolist()
    joint_index = {name: index for index, name in enumerate(joint_class_names)}
    joint_labels = np.array([joint_index[name] for name in joint_raw], dtype=np.int64)
    batch = tensors(features, base.DEVICE)
    batch_joint_y = torch.from_numpy(joint_labels.astype(np.int64)).to(base.DEVICE)
    batch_intent_y = torch.from_numpy(intent_labels.astype(np.int64)).to(base.DEVICE)
    batch_scene_y = torch.from_numpy(scene_targets.astype(np.int64)).to(base.DEVICE)

    if args.model == "baseline":
        model = base.PerceiverIOSceneBaseline(
            num_classes=len(joint_class_names),
            model_dim=base.MODEL_DIM,
            num_latents=base.NUM_LATENTS,
            depth=base.DEPTH,
            num_heads=base.NUM_HEADS,
            dropout=base.DROPOUT,
        ).to(base.DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)

        with timed_block() as train_timer:
            model.train()
            optimizer.zero_grad()
            logits = model(batch["imu"], batch["gesture"], batch["audio"], batch["text"], batch["scene"])
            loss = criterion(logits, batch_joint_y)
            loss.backward()
            optimizer.step()
        with timed_block() as test_timer:
            model.eval()
            with torch.no_grad():
                test_logits = model(batch["imu"], batch["gesture"], batch["audio"], batch["text"], batch["scene"])
            preds = test_logits.argmax(dim=1)
            test_acc = float((preds == batch_joint_y).float().mean().item())
        train_loss = float(loss.item())
    else:
        model = improved.Anchor2PerceiverIO(
            num_joint_classes=len(joint_class_names),
            num_intent_classes=len(base.INTENT_NAMES),
            num_scene_classes=len(base.SCENE_NAME_TO_ID),
            joint_class_names=joint_class_names,
            model_dim=improved.MODEL_DIM,
            num_latents=improved.NUM_LATENTS,
            depth=improved.DEPTH,
            num_heads=improved.NUM_HEADS,
            dropout=improved.DROPOUT,
        ).to(base.DEVICE)
        joint_criterion = nn.CrossEntropyLoss()
        intent_criterion = nn.CrossEntropyLoss()
        scene_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=improved.LEARNING_RATE, weight_decay=improved.WEIGHT_DECAY)

        with timed_block() as train_timer:
            model.train()
            optimizer.zero_grad()
            outputs = model(batch["imu"], batch["gesture"], batch["audio"], batch["text"], batch["scene"])
            loss, _ = improved.compute_loss(
                outputs,
                batch_joint_y,
                batch_intent_y,
                batch_scene_y,
                joint_criterion,
                intent_criterion,
                scene_criterion,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), improved.GRAD_CLIP_NORM)
            optimizer.step()
        with timed_block() as test_timer:
            model.eval()
            with torch.no_grad():
                test_outputs = model(batch["imu"], batch["gesture"], batch["audio"], batch["text"], batch["scene"])
            preds = test_outputs["joint_logits"].argmax(dim=1)
            test_acc = float((preds == batch_joint_y).float().mean().item())
        train_loss = float(loss.item())

    sample_count = int(len(joint_labels))
    runtime = timing_payload(train_timer["seconds"], sample_count, test_timer["seconds"], sample_count)
    report = {
        "config": {
            "model": args.model,
            "video": args.video,
            "batch_size": sample_count,
            "device": str(base.DEVICE),
            "gesture_feature_dir": str(base.STRONG_GESTURE_DIR),
            "gesture_feature_dim": int(base.GESTURE_FEAT_DIM),
            "raw_hand_geometry": bool(args.raw_hand_geometry),
        },
        "feature_timing": {
            "cache_load_seconds": float(load_timer["seconds"]),
            "cache_load_avg_seconds_per_sample": float(load_timer["seconds"]) / max(sample_count, 1),
            "raw_hand_geometry_seconds": raw_geometry_seconds,
            "raw_hand_geometry_avg_seconds_per_sample": None
            if raw_geometry_seconds is None
            else raw_geometry_seconds / max(sample_count, 1),
        },
        "runtime": runtime,
        "batch_metrics": {
            "train_loss": train_loss,
            "test_joint_acc_untrained": test_acc,
        },
        "features": {key: list(value.shape) for key, value in features.items()},
        "raw_hand_geometry": raw_geometry_report,
        "joint_classes": joint_class_names,
        "joint_classes_in_batch": sorted(set(joint_raw.tolist())),
    }

    output_path = output_dir / "batch_end_to_end_metrics.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("[batch-e2e]")
    print(f"  model                         {args.model}")
    print(f"  samples                       {sample_count}")
    print(f"  cache_load_avg/sample          {report['feature_timing']['cache_load_avg_seconds_per_sample']:.6f}s")
    if raw_geometry_seconds is not None:
        print(f"  raw_hand_geometry_avg/sample  {raw_geometry_seconds / max(sample_count, 1):.6f}s")
    print(f"  train_avg/sample              {runtime['train_avg_seconds_per_sample']:.6f}s")
    print(f"  test_avg/sample               {runtime['test_avg_seconds_per_sample']:.6f}s")
    print(f"  metrics                       {output_path}")


if __name__ == "__main__":
    main()
