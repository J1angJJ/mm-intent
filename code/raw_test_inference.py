from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT


DEFAULT_CHECKPOINT_DIR = MODEL_OUTPUT_ROOT / "feature_suite" / "hand_geometry" / "main"


def run_stage(name: str, action: Callable[[], None], timings: dict[str, float]) -> None:
    print(f"\n[raw-inference] stage={name} start", flush=True)
    started = time.perf_counter()
    action()
    elapsed = time.perf_counter() - started
    timings[name] = float(elapsed)
    print(f"[raw-inference] stage={name} seconds={elapsed:.3f}", flush=True)


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def build_fresh_metadata(processed_dir: Path, video_names: list[str]) -> None:
    """Build current-run alignment metadata without extracting unused CLIP gesture features."""
    for video_name in video_names:
        stem = Path(video_name).stem
        timestamp_path = processed_dir / f"features_timestamp_{stem}.npy"
        if not timestamp_path.exists():
            raise FileNotFoundError(f"Missing newly extracted timestamps: {timestamp_path}")
        timestamp_payload = np.load(timestamp_path, allow_pickle=True).item()
        metadata = {
            "labels": np.asarray(timestamp_payload["labels"]),
            "video_names": np.asarray(timestamp_payload["video_names"]),
            "approx_timestamps": np.asarray(timestamp_payload["approx_timestamps"]),
        }
        np.save(processed_dir / f"metadata_strong_gesture_{stem}.npy", metadata)


def subset_payload(value: Any, indices: list[int]) -> Any:
    if isinstance(value, np.ndarray) and len(value) >= max(indices, default=-1) + 1:
        return value[indices]
    if isinstance(value, list) and len(value) >= max(indices, default=-1) + 1:
        return [value[index] for index in indices]
    return value


def align_modalities_to_geometry(processed_dir: Path, video_names: list[str]) -> None:
    """Keep every modality aligned when raw geometry drops an invalid video segment."""
    for video_name in video_names:
        stem = Path(video_name).stem
        metadata_path = processed_dir / f"metadata_strong_gesture_{stem}.npy"
        geometry_path = processed_dir / "hand_geometry_features" / f"strong_gesture_features_{stem}.npy"
        if not geometry_path.exists():
            raise FileNotFoundError(f"Missing newly extracted hand geometry: {geometry_path}")

        metadata = np.load(metadata_path, allow_pickle=True).item()
        geometry = np.load(geometry_path, allow_pickle=True).item()
        source_timestamps = [str(value) for value in metadata["approx_timestamps"]]
        geometry_timestamps = [str(value) for value in geometry["approx_timestamps"]]
        index_by_timestamp = {value: index for index, value in enumerate(source_timestamps)}
        indices = [index_by_timestamp[value] for value in geometry_timestamps]

        aligned_metadata = {key: subset_payload(value, indices) for key, value in metadata.items()}
        np.save(metadata_path, aligned_metadata)

        audio_path = processed_dir / "audio_features" / f"audio_features_{stem}.npy"
        audio = np.load(audio_path, allow_pickle=True)
        audio_rows = list(audio.tolist())
        audio_by_id = {int(row["id"]): row for row in audio_rows}
        aligned_audio = [audio_by_id[index] for index in indices]
        np.save(audio_path, np.asarray(aligned_audio, dtype=object))

        text_path = processed_dir / "text_features" / f"text_features_{stem}.npy"
        text = np.load(text_path, allow_pickle=True).item()
        text["features"] = np.asarray(text["features"])[indices]
        if "metadata" in text:
            text["metadata"] = [text["metadata"][index] for index in indices]
        np.save(text_path, text)

        imu_path = processed_dir / "imu_features" / f"imu_features_{stem}.npy"
        imu = np.load(imu_path, allow_pickle=True).item()
        imu = {key: subset_payload(value, indices) for key, value in imu.items()}
        np.save(imu_path, imu)


def configure_checkpoint_environment(checkpoint_dir: Path) -> None:
    metrics_path = checkpoint_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = metrics.get("config", {})
    env_map = {
        "model_dim": "IMPROVED_REAL_SCENE_A2_MODEL_DIM",
        "num_latents": "IMPROVED_REAL_SCENE_A2_NUM_LATENTS",
        "depth": "IMPROVED_REAL_SCENE_A2_DEPTH",
        "num_heads": "IMPROVED_REAL_SCENE_A2_NUM_HEADS",
        "dropout": "IMPROVED_REAL_SCENE_A2_DROPOUT",
        "min_gate": "IMPROVED_REAL_SCENE_A2_MIN_GATE",
        "fallback_max_gate": "IMPROVED_REAL_SCENE_A2_FALLBACK_MAX_GATE",
        "fallback_aux_weight": "IMPROVED_REAL_SCENE_A2_FALLBACK_AUX_WEIGHT",
    }
    for key, env_name in env_map.items():
        if key in config:
            os.environ[env_name] = str(config[key])


def evaluate_from_fresh_features(
    checkpoint_dir: Path,
    processed_dir: Path,
    scene_cache_dir: Path,
    video_names: list[str],
    batch_size: int,
    predictions_path: Path,
) -> dict[str, Any]:
    configure_checkpoint_environment(checkpoint_dir)
    os.environ["MM_INTENT_PROCESSED_DATA_DIR"] = str(processed_dir)
    os.environ["MM_INTENT_SCENE_CACHE_DIR"] = str(scene_cache_dir)
    os.environ["MM_INTENT_GESTURE_FEATURE_DIR"] = str(processed_dir / "hand_geometry_features")
    os.environ["MM_INTENT_AUDIO_FEATURE_DIR"] = str(processed_dir / "audio_features")
    os.environ["MM_INTENT_TEXT_FEATURE_DIR"] = str(processed_dir / "text_features")
    os.environ["MM_INTENT_IMU_FEATURE_DIR"] = str(processed_dir / "imu_features")
    os.environ["MM_INTENT_GESTURE_FEAT_DIM"] = "96"
    os.environ["SMART_AR_TEST_VIDEO_NAMES"] = ",".join(video_names)

    import torch

    import baseline_real_scene as base
    import train_and_test as improved

    checkpoint_path = checkpoint_dir / "improved_real_scene_anchor2.pt"
    with (checkpoint_dir / "label_encoder.pkl").open("rb") as file:
        label_encoder = pickle.load(file)
    with (checkpoint_dir / "scalers.pkl").open("rb") as file:
        scalers = pickle.load(file)

    scene_cache = base.RealSceneFeatureCache(scene_cache_dir)
    raw_features, intent_targets, scene_targets, _ = base.load_multimodal_data(video_names, scene_cache)
    features = base.apply_scalers(raw_features, scalers)
    joint_names = base.build_joint_labels(intent_targets, scene_targets)
    joint_targets = label_encoder.transform(joint_names)
    joint_class_names = label_encoder.classes_.tolist()
    intent_class_names = [base.INTENT_NAMES[index] for index in sorted(base.INTENT_NAMES)]
    scene_class_names = [base.SCENE_ID_TO_NAME[index] for index in range(len(base.SCENE_ID_TO_NAME))]

    loader = improved.make_loader(
        features,
        joint_targets,
        intent_targets,
        scene_targets,
        batch_size,
        shuffle=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=base.DEVICE, weights_only=False)
    model = improved.Anchor2PerceiverIO(
        num_joint_classes=len(joint_class_names),
        num_intent_classes=len(intent_class_names),
        num_scene_classes=len(scene_class_names),
        joint_class_names=joint_class_names,
        model_dim=int(checkpoint.get("model_dim", improved.MODEL_DIM)),
        num_latents=int(checkpoint.get("num_latents", improved.NUM_LATENTS)),
        depth=int(checkpoint.get("depth", improved.DEPTH)),
        num_heads=int(checkpoint.get("num_heads", improved.NUM_HEADS)),
        dropout=float(checkpoint.get("dropout", improved.DROPOUT)),
        min_gate=improved.MIN_GATE,
    ).to(base.DEVICE)
    load_report = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_report.missing_keys or load_report.unexpected_keys:
        print(
            f"[checkpoint] missing={len(load_report.missing_keys)} "
            f"unexpected={len(load_report.unexpected_keys)}"
        )
    model.eval()

    joint_predictions: list[np.ndarray] = []
    intent_predictions: list[np.ndarray] = []
    scene_predictions: list[np.ndarray] = []
    if base.DEVICE.type == "cuda":
        torch.cuda.synchronize(base.DEVICE)
    inference_started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch_imu, batch_gesture, batch_audio, batch_text, batch_scene, *_ = batch
            outputs = model(
                batch_imu.to(base.DEVICE),
                batch_gesture.to(base.DEVICE),
                batch_audio.to(base.DEVICE),
                batch_text.to(base.DEVICE),
                batch_scene.to(base.DEVICE),
            )
            joint_predictions.append(outputs["joint_logits"].argmax(dim=1).cpu().numpy())
            intent_predictions.append(outputs["intent_logits"].argmax(dim=1).cpu().numpy())
            scene_predictions.append(outputs["scene_logits"].argmax(dim=1).cpu().numpy())
    if base.DEVICE.type == "cuda":
        torch.cuda.synchronize(base.DEVICE)
    classifier_seconds = time.perf_counter() - inference_started

    joint_pred = np.concatenate(joint_predictions)
    intent_pred = np.concatenate(intent_predictions)
    scene_pred = np.concatenate(scene_predictions)
    predicted_joint_names = label_encoder.inverse_transform(joint_pred)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("sample_index", "true_joint", "predicted_joint", "correct"),
        )
        writer.writeheader()
        for index, (true_name, predicted_name) in enumerate(zip(joint_names, predicted_joint_names)):
            writer.writerow(
                {
                    "sample_index": index,
                    "true_joint": str(true_name),
                    "predicted_joint": str(predicted_name),
                    "correct": int(true_name == predicted_name),
                }
            )

    return {
        "sample_count": int(len(joint_targets)),
        "joint_accuracy": float(np.mean(joint_pred == joint_targets)),
        "intent_accuracy": float(np.mean(intent_pred == intent_targets)),
        "scene_accuracy": float(np.mean(scene_pred == scene_targets)),
        "classifier_forward_seconds": float(classifier_seconds),
        "classifier_forward_seconds_per_sample": float(classifier_seconds / max(len(joint_targets), 1)),
        "device": str(base.DEVICE),
    }


def validate_checkpoint(checkpoint_dir: Path) -> None:
    required = (
        "metrics.json",
        "improved_real_scene_anchor2.pt",
        "label_encoder.pkl",
        "scalers.pkl",
    )
    missing = [name for name in required if not (checkpoint_dir / name).exists()]
    if missing:
        raise SystemExit(f"Checkpoint directory is incomplete: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure raw test inference from fresh feature extraction to predicted labels."
    )
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--run-dir", help="Must not already exist; this guarantees no test feature cache is reused.")
    parser.add_argument("--video-names", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    validate_checkpoint(checkpoint_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (
        MODEL_OUTPUT_ROOT / "raw_test_inference" / timestamp
    ).resolve()
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists; refusing possible cache reuse: {run_dir}")
    processed_dir = run_dir / "features"
    scene_cache_dir = run_dir / "scene_cache"
    processed_dir.mkdir(parents=True)
    scene_cache_dir.mkdir(parents=True)

    os.environ["MM_INTENT_PROCESSED_DATA_DIR"] = str(processed_dir)
    os.environ["MM_INTENT_SCENE_CACHE_DIR"] = str(scene_cache_dir)
    os.environ["MM_INTENT_GESTURE_FEAT_DIM"] = "96"
    configure_checkpoint_environment(checkpoint_dir)

    import baseline_real_scene as base

    video_names = args.video_names or list(base.DEFAULT_TEST_VIDEO_NAMES)
    unknown = set(video_names) - set(base.VIDEO_LABELS)
    if unknown:
        raise SystemExit(f"Unknown video names: {sorted(unknown)}")

    env = os.environ.copy()
    env["MM_INTENT_PROCESSED_DATA_DIR"] = str(processed_dir)
    env["MM_INTENT_SCENE_CACHE_DIR"] = str(scene_cache_dir)
    env["MM_INTENT_VIDEO_NAMES"] = ",".join(video_names)
    env["SMART_AR_TEST_VIDEO_NAMES"] = ",".join(video_names)

    timings: dict[str, float] = {}
    total_started = time.perf_counter()
    run_stage(
        "timestamp_vad",
        lambda: run_command([sys.executable, "code/feature_extraction/get_timestamp.py"], env),
        timings,
    )
    run_stage("alignment_metadata", lambda: build_fresh_metadata(processed_dir, video_names), timings)
    run_stage(
        "audio_mfcc",
        lambda: run_command([sys.executable, "code/feature_extraction/mfcc.py"], env),
        timings,
    )
    run_stage(
        "text_whisper_sentence",
        lambda: run_command([sys.executable, "code/feature_extraction/ASR.py"], env),
        timings,
    )
    run_stage(
        "imu",
        lambda: run_command([sys.executable, "code/feature_extraction/imu.py"], env),
        timings,
    )
    run_stage(
        "hand_geometry",
        lambda: run_command(
            [
                sys.executable,
                "code/feature_extraction/extract_hand_geometry_features.py",
                "--output-dir",
                str(processed_dir / "hand_geometry_features"),
            ],
            env,
        ),
        timings,
    )
    run_stage("align_modalities", lambda: align_modalities_to_geometry(processed_dir, video_names), timings)

    evaluation: dict[str, Any] = {}

    def classify() -> None:
        evaluation.update(
            evaluate_from_fresh_features(
                checkpoint_dir,
                processed_dir,
                scene_cache_dir,
                video_names,
                args.batch_size,
                run_dir / "predictions.csv",
            )
        )

    run_stage("scene_vit_and_classification", classify, timings)
    total_seconds = time.perf_counter() - total_started
    sample_count = int(evaluation["sample_count"])
    report = {
        "protocol": "fresh raw test feature extraction to classification label output",
        "cache_policy": "isolated new run directory; no precomputed test feature files",
        "checkpoint_dir": str(checkpoint_dir),
        "run_dir": str(run_dir),
        "video_names": video_names,
        "stage_seconds": timings,
        "raw_inference_total_seconds": float(total_seconds),
        "raw_inference_seconds_per_sample": float(total_seconds / max(sample_count, 1)),
        **evaluation,
    }
    report_path = run_dir / "raw_inference_timing.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[raw-inference] complete")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[raw-inference] report={report_path}")


if __name__ == "__main__":
    main()
