from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_VIDEO_NAMES = (
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
)
DEFAULT_MODEL_OUTPUTS = {
    "baseline": PROJECT_ROOT / "outputs" / "baseline_real_scene_perceiver_io",
    "improved": PROJECT_ROOT / "outputs" / "improved_real_scene_anchor2_perceiver_io",
}
DEFAULT_MODEL_OUTPUT_CANDIDATES = {
    ("baseline", "clip"): (
        PROJECT_ROOT / "outputs" / "baseline_real_scene_perceiver_io",
        PROJECT_ROOT / "outputs" / "baseline_real_scene_perceiver_io_mptasks",
        PROJECT_ROOT / "outputs" / "baseline_raw_end_to_end",
    ),
    ("baseline", "hand_geometry"): (
        PROJECT_ROOT / "outputs" / "baseline_hand_geometry_raw_end_to_end",
    ),
    ("improved", "clip"): (
        PROJECT_ROOT / "outputs" / "improved_real_scene_anchor2_perceiver_io",
        PROJECT_ROOT / "outputs" / "improved_real_scene_anchor2_perceiver_io_mptasks",
        PROJECT_ROOT / "outputs" / "improved_raw_end_to_end",
    ),
    ("improved", "hand_geometry"): (
        PROJECT_ROOT / "outputs" / "hand_geometry_raw_end_to_end",
        PROJECT_ROOT / "outputs" / "improved_hand_geometry_raw_end_to_end",
    ),
}
EXPECTED_GESTURE_FEAT_DIMS = {
    "clip": 768,
    "hand_geometry": 96,
}
CHECKPOINT_NAMES = {
    "baseline": "baseline_real_scene_perceiver_io.pt",
    "improved": "improved_real_scene_anchor2.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh raw-data inference benchmark: re-extract test features into "
            "empty directories, then time raw data -> classification labels."
        )
    )
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument(
        "--gesture-representation",
        choices=("clip", "hand_geometry"),
        default="clip",
        help="Use CLIP gesture features or freshly extracted hand-geometry gesture features.",
    )
    parser.add_argument("--model-output-dir", help="Directory containing checkpoint, scalers.pkl, and label_encoder.pkl.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--scalers")
    parser.add_argument("--label-encoder")
    parser.add_argument("--run-dir", help="Fresh output directory. Must be absent or empty.")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--hololens-dir")
    parser.add_argument("--fisheye-dir")
    parser.add_argument("--hand-task-model")
    parser.add_argument("--video-names", nargs="*", default=list(DEFAULT_TEST_VIDEO_NAMES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--show-report", action="store_true")
    return parser.parse_args()


def ensure_absent_or_empty(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(
            f"{label} must be absent or empty for fresh raw inference: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def resolve_run_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            PROJECT_ROOT
            / "outputs"
            / "fresh_raw_inference"
            / f"{args.model}_{args.gesture_representation}_{stamp}"
        ).resolve()

    ensure_absent_or_empty(run_dir, "run directory")
    feature_dir = run_dir / "features"
    scene_cache_dir = run_dir / "scene_cache"
    ensure_absent_or_empty(feature_dir, "fresh feature directory")
    ensure_absent_or_empty(scene_cache_dir, "fresh scene cache directory")
    return run_dir, feature_dir, scene_cache_dir


def build_env(args: argparse.Namespace, feature_dir: Path, scene_cache_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["MM_INTENT_PROCESSED_DATA_DIR"] = str(feature_dir)
    env["MM_INTENT_SCENE_CACHE_DIR"] = str(scene_cache_dir)
    env["MM_INTENT_VIDEO_NAMES"] = ",".join(args.video_names)
    env["SMART_AR_TEST_VIDEO_NAMES"] = ",".join(args.video_names)
    env["SMART_AR_MISSING_MODALITIES"] = ""
    env["SMART_AR_NOISE_MODALITY"] = ""
    env["SMART_AR_NOISE_LEVEL"] = "0.0"
    env["MM_INTENT_AUDIO_FEATURE_DIR"] = str(feature_dir / "audio_features")
    env["MM_INTENT_TEXT_FEATURE_DIR"] = str(feature_dir / "text_features")
    env["MM_INTENT_IMU_FEATURE_DIR"] = str(feature_dir / "imu_features")
    env["MM_INTENT_AUDIO_FEAT_DIM"] = "39"
    env["MM_INTENT_TEXT_FEAT_DIM"] = "384"
    env["MM_INTENT_IMU_FEAT_DIM"] = "12"
    if args.dataset_dir:
        env["MM_INTENT_DATASET_DIR"] = str(Path(args.dataset_dir).resolve())
    if args.hololens_dir:
        env["MM_INTENT_HOLOLENS_DIR"] = str(Path(args.hololens_dir).resolve())
    if args.fisheye_dir:
        env["MM_INTENT_FISHEYE_DIR"] = str(Path(args.fisheye_dir).resolve())
    if args.hand_task_model:
        env["MM_INTENT_HAND_LANDMARKER_TASK"] = str(Path(args.hand_task_model).resolve())
    if args.gesture_representation == "hand_geometry":
        env["MM_INTENT_GESTURE_FEATURE_DIR"] = str(feature_dir / "hand_geometry_features")
        env["MM_INTENT_GESTURE_FEAT_DIM"] = "96"
    else:
        env["MM_INTENT_GESTURE_FEATURE_DIR"] = str(feature_dir / "strong_gesture_features")
        env["MM_INTENT_GESTURE_FEAT_DIM"] = "768"
    return env


def configure_current_process(env: dict[str, str]) -> None:
    for key, value in env.items():
        if key.startswith("MM_INTENT_") or key.startswith("SMART_AR_"):
            os.environ[key] = value
    os.environ["PYTHONIOENCODING"] = env["PYTHONIOENCODING"]
    os.environ["PYTHONUTF8"] = env["PYTHONUTF8"]


def extraction_commands(args: argparse.Namespace, feature_dir: Path) -> list[list[str]]:
    commands = [
        [sys.executable, "code/feature_extraction/get_timestamp.py"],
        [sys.executable, "code/feature_extraction/strong_gesture2.0.py"],
        [sys.executable, "code/feature_extraction/mfcc.py"],
        [sys.executable, "code/feature_extraction/ASR.py"],
        [sys.executable, "code/feature_extraction/imu.py"],
    ]
    if args.gesture_representation == "hand_geometry":
        command = [
            sys.executable,
            "code/feature_extraction/extract_hand_geometry_features.py",
            "--output-dir",
            str(feature_dir / "hand_geometry_features"),
        ]
        if args.hand_task_model:
            command.extend(["--task-model", str(Path(args.hand_task_model).resolve())])
        commands.append(command)
    return commands


def run_commands(commands: Sequence[Sequence[str]], env: dict[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for command in commands:
        started = time.perf_counter()
        print("[fresh-preprocess]", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
        records.append(
            {
                "command": list(command),
                "seconds": float(time.perf_counter() - started),
            }
        )
    return records


def expected_feature_paths(
    feature_dir: Path,
    video_names: Iterable[str],
    gesture_representation: str,
) -> list[Path]:
    paths: list[Path] = []
    for video_name in video_names:
        base = Path(video_name).stem
        gesture_dir = (
            feature_dir / "hand_geometry_features"
            if gesture_representation == "hand_geometry"
            else feature_dir / "strong_gesture_features"
        )
        paths.extend(
            [
                feature_dir / f"features_timestamp_{base}.npy",
                feature_dir / f"metadata_strong_gesture_{base}.npy",
                gesture_dir / f"strong_gesture_features_{base}.npy",
                feature_dir / "audio_features" / f"audio_features_{base}.npy",
                feature_dir / "text_features" / f"text_features_{base}.npy",
                feature_dir / "imu_features" / f"imu_features_{base}.npy",
            ]
        )
    return paths


def validate_fresh_features(feature_dir: Path, video_names: Sequence[str], gesture_representation: str) -> None:
    missing = [
        path
        for path in expected_feature_paths(feature_dir, video_names, gesture_representation)
        if not path.exists()
    ]
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:20])
        raise RuntimeError(
            f"Fresh feature extraction left {len(missing)} required files missing.\n{examples}"
        )


def load_checkpoint(path: Path, device) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def synchronize_device(device) -> None:
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def safe_divide(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def artifact_paths(model_output_dir: Path, model_name: str) -> tuple[Path, Path, Path]:
    return (
        model_output_dir / CHECKPOINT_NAMES[model_name],
        model_output_dir / "scalers.pkl",
        model_output_dir / "label_encoder.pkl",
    )


def scaler_gesture_dim(scalers: dict[str, Any]) -> int | None:
    scaler = scalers.get("gesture")
    dim = getattr(scaler, "n_features_in_", None)
    return int(dim) if dim is not None else None


def checkpoint_gesture_dim(checkpoint: dict[str, Any]) -> int | None:
    state_dict = checkpoint.get("model_state_dict", {})
    for key in ("gesture_proj.0.weight", "module.gesture_proj.0.weight"):
        weight = state_dict.get(key)
        if weight is not None and len(weight.shape) == 2:
            return int(weight.shape[1])
    return None


def read_scaler_gesture_dim(scalers_path: Path) -> int | None:
    try:
        with scalers_path.open("rb") as file:
            return scaler_gesture_dim(pickle.load(file))
    except Exception:
        return None


def select_default_model_output_dir(model_name: str, gesture_representation: str) -> Path:
    expected_dim = EXPECTED_GESTURE_FEAT_DIMS[gesture_representation]
    candidates = DEFAULT_MODEL_OUTPUT_CANDIDATES.get((model_name, gesture_representation), ())
    diagnostics: list[str] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checkpoint_path, scalers_path, label_encoder_path = artifact_paths(candidate, model_name)
        missing = [
            path.name
            for path in (checkpoint_path, scalers_path, label_encoder_path)
            if not path.exists()
        ]
        if missing:
            diagnostics.append(f"{candidate}: missing {', '.join(missing)}")
            continue
        dim = read_scaler_gesture_dim(scalers_path)
        if dim != expected_dim:
            diagnostics.append(f"{candidate}: gesture scaler dim={dim}, expected {expected_dim}")
            continue
        return candidate

    checked = "\n".join(f"  - {item}" for item in diagnostics) or "  - no candidates configured"
    raise SystemExit(
        "No compatible default inference artifact was found.\n"
        f"model={model_name} gesture_representation={gesture_representation} expected_gesture_dim={expected_dim}\n"
        f"Checked:\n{checked}\n"
        "Train the matching model first, or pass --model-output-dir/--checkpoint/--scalers/--label-encoder explicitly."
    )


def resolve_artifact_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    model_output_dir = (
        Path(args.model_output_dir).resolve()
        if args.model_output_dir
        else select_default_model_output_dir(args.model, args.gesture_representation)
    )
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else artifact_paths(model_output_dir, args.model)[0]
    )
    scalers_path = (
        Path(args.scalers).resolve()
        if args.scalers
        else artifact_paths(model_output_dir, args.model)[1]
    )
    label_encoder_path = (
        Path(args.label_encoder).resolve()
        if args.label_encoder
        else artifact_paths(model_output_dir, args.model)[2]
    )
    return model_output_dir, checkpoint_path, scalers_path, label_encoder_path


def validate_artifact_compatibility(
    gesture_representation: str,
    scalers: dict[str, Any],
    checkpoint: dict[str, Any],
    scalers_path: Path,
    checkpoint_path: Path,
) -> None:
    expected_dim = EXPECTED_GESTURE_FEAT_DIMS[gesture_representation]
    scaler_dim = scaler_gesture_dim(scalers)
    checkpoint_dim = checkpoint_gesture_dim(checkpoint)
    problems = []
    if scaler_dim != expected_dim:
        problems.append(f"scalers gesture dim={scaler_dim}, expected {expected_dim}: {scalers_path}")
    if checkpoint_dim != expected_dim:
        problems.append(f"checkpoint gesture dim={checkpoint_dim}, expected {expected_dim}: {checkpoint_path}")
    if problems:
        raise SystemExit(
            "Inference artifacts do not match the requested gesture representation.\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


def build_model(model_name: str, checkpoint: dict[str, Any], label_encoder, device: torch.device):
    import baseline_real_scene as base

    joint_class_names = label_encoder.classes_.tolist()
    if model_name == "baseline":
        model = base.PerceiverIOSceneBaseline(
            num_classes=int(checkpoint.get("num_classes", len(joint_class_names))),
            model_dim=int(checkpoint.get("model_dim", base.MODEL_DIM)),
            num_latents=int(checkpoint.get("num_latents", base.NUM_LATENTS)),
            depth=int(checkpoint.get("depth", base.DEPTH)),
            num_heads=int(checkpoint.get("num_heads", base.NUM_HEADS)),
            dropout=float(checkpoint.get("dropout", base.DROPOUT)),
        )
    else:
        import train_and_test as improved

        improved.INTENT_REFINE_SCALE = float(
            checkpoint.get("intent_refine_scale", improved.INTENT_REFINE_SCALE)
        )
        improved.GESTURE_LOGIT_BLEND = float(
            checkpoint.get("gesture_logit_blend", improved.GESTURE_LOGIT_BLEND)
        )
        improved.FALLBACK_MAX_GATE = float(
            checkpoint.get("fallback_max_gate", improved.FALLBACK_MAX_GATE)
        )
        model = improved.Anchor2PerceiverIO(
            num_joint_classes=int(checkpoint.get("num_joint_classes", len(joint_class_names))),
            num_intent_classes=len(base.INTENT_NAMES),
            num_scene_classes=len(base.SCENE_ID_TO_NAME),
            joint_class_names=joint_class_names,
            model_dim=int(checkpoint.get("model_dim", improved.MODEL_DIM)),
            num_latents=int(checkpoint.get("num_latents", improved.NUM_LATENTS)),
            depth=int(checkpoint.get("depth", improved.DEPTH)),
            num_heads=int(checkpoint.get("num_heads", improved.NUM_HEADS)),
            dropout=float(checkpoint.get("dropout", improved.DROPOUT)),
            min_gate=float(checkpoint.get("min_gate", improved.MIN_GATE)),
        )
        model.support_scales = {
            "imu": float(checkpoint.get("imu_max_scale", improved.IMU_MAX_SCALE)),
            "audio": float(checkpoint.get("audio_max_scale", improved.AUDIO_MAX_SCALE)),
        }

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def predict(model_name: str, model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    true_values: list[np.ndarray] = []
    predicted_values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            imu, gesture, audio, text, scene, labels, _scene_labels = batch
            inputs = [value.to(device) for value in (imu, gesture, audio, text, scene)]
            if model_name == "baseline":
                logits = model(*inputs)
            else:
                logits = model(*inputs)["joint_logits"]
            true_values.append(labels.numpy())
            predicted_values.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(true_values), np.concatenate(predicted_values)


def split_joint_names(names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scenes: list[str] = []
    intents: list[str] = []
    for name in names.tolist():
        scene, intent = str(name).split("_", 1)
        scenes.append(scene)
        intents.append(intent)
    return np.asarray(scenes, dtype=object), np.asarray(intents, dtype=object)


def main() -> None:
    args = parse_args()
    if not args.video_names:
        raise SystemExit("--video-names must contain at least one video")

    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    run_dir, feature_dir, scene_cache_dir = resolve_run_dirs(args)
    env = build_env(args, feature_dir, scene_cache_dir)
    configure_current_process(env)
    model_output_dir, checkpoint_path, scalers_path, label_encoder_path = resolve_artifact_paths(args)
    for path in (checkpoint_path, scalers_path, label_encoder_path):
        if not path.exists():
            raise SystemExit(f"Missing inference artifact: {path}")

    artifact_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with scalers_path.open("rb") as file:
        scalers = pickle.load(file)
    with label_encoder_path.open("rb") as file:
        label_encoder = pickle.load(file)
    checkpoint = load_checkpoint(checkpoint_path, device)
    validate_artifact_compatibility(
        args.gesture_representation,
        scalers,
        checkpoint,
        scalers_path,
        checkpoint_path,
    )
    artifact_loading_seconds = time.perf_counter() - artifact_start

    model_start = time.perf_counter()
    model = build_model(args.model, checkpoint, label_encoder, device)
    synchronize_device(device)
    model_loading_seconds = time.perf_counter() - model_start

    print(f"[fresh-raw] model={args.model} gesture={args.gesture_representation} device={device}")
    print(f"[fresh-raw] checkpoint={checkpoint_path}")
    print(f"[fresh-raw] run_dir={run_dir}")
    print(f"[fresh-raw] videos={len(args.video_names)}")

    import baseline_real_scene as base
    from real_scene_utils import RealSceneFeatureCache

    raw_to_label_start = time.perf_counter()

    extraction_start = time.perf_counter()
    command_records = run_commands(extraction_commands(args, feature_dir), env)
    raw_feature_extraction_seconds = time.perf_counter() - extraction_start
    validate_fresh_features(feature_dir, args.video_names, args.gesture_representation)

    if args.preprocess_only:
        manifest = {
            "model": args.model,
            "gesture_representation": args.gesture_representation,
            "run_dir": str(run_dir),
            "fresh_feature_dir": str(feature_dir),
            "fresh_scene_cache_dir": str(scene_cache_dir),
            "video_names": args.video_names,
            "command_records": command_records,
            "raw_feature_extraction_seconds": float(raw_feature_extraction_seconds),
            "preprocess_only": True,
        }
        manifest_path = run_dir / "fresh_preprocessing_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[fresh-output] manifest={manifest_path}")
        return

    data_start = time.perf_counter()
    scene_cache = RealSceneFeatureCache(scene_cache_dir)
    raw_features, intent_targets, scene_targets, scene_selection = base.load_multimodal_data(
        args.video_names,
        scene_cache,
    )
    joint_names = base.build_joint_labels(intent_targets, scene_targets)
    encoded_joint = label_encoder.transform(joint_names)
    scaled_features = base.apply_scalers(raw_features, scalers)
    loader = base.make_loader(
        scaled_features,
        encoded_joint,
        scene_targets,
        args.batch_size,
        shuffle=False,
    )
    data_preparation_seconds = time.perf_counter() - data_start

    synchronize_device(device)
    forward_start = time.perf_counter()
    joint_true, joint_pred = predict(args.model, model, loader, device)
    synchronize_device(device)
    model_forward_seconds = time.perf_counter() - forward_start

    label_start = time.perf_counter()
    true_names = label_encoder.inverse_transform(joint_true)
    pred_names = label_encoder.inverse_transform(joint_pred)
    true_scenes, true_intents = split_joint_names(true_names)
    pred_scenes, pred_intents = split_joint_names(pred_names)
    label_decoding_seconds = time.perf_counter() - label_start

    raw_to_label_seconds = time.perf_counter() - raw_to_label_start
    sample_count = int(len(joint_true))
    joint_class_names = label_encoder.classes_.tolist()
    report = classification_report(
        joint_true,
        joint_pred,
        labels=np.arange(len(joint_class_names)),
        target_names=joint_class_names,
        zero_division=0,
        digits=4,
    )
    predictions = [
        {
            "sample_index": index,
            "true_joint": str(true_joint),
            "predicted_joint": str(pred_joint),
            "true_intent": str(true_intent),
            "predicted_intent": str(pred_intent),
            "true_scene": str(true_scene),
            "predicted_scene": str(pred_scene),
        }
        for index, (true_joint, pred_joint, true_intent, pred_intent, true_scene, pred_scene) in enumerate(
            zip(true_names, pred_names, true_intents, pred_intents, true_scenes, pred_scenes)
        )
    ]
    result = {
        "protocol": {
            "name": "fresh_raw_to_label_inference",
            "definition": (
                "Timer starts immediately before raw test feature extraction and stops "
                "after predicted classification labels are decoded."
            ),
            "cache_policy": (
                "The feature directory and scene cache directory must be absent or empty. "
                "All test feature files are generated inside this run directory before inference."
            ),
            "model_weight_cache_note": (
                "Pretrained model weight caches may be used; cached test feature files are not used."
            ),
        },
        "model": args.model,
        "gesture_representation": args.gesture_representation,
        "model_output_dir": str(model_output_dir),
        "checkpoint": str(checkpoint_path),
        "scalers": str(scalers_path),
        "label_encoder": str(label_encoder_path),
        "run_dir": str(run_dir),
        "fresh_feature_dir": str(feature_dir),
        "fresh_scene_cache_dir": str(scene_cache_dir),
        "video_names": args.video_names,
        "sample_count": sample_count,
        "joint_accuracy": float(accuracy_score(joint_true, joint_pred)),
        "intent_accuracy": float(accuracy_score(true_intents, pred_intents)),
        "scene_accuracy": float(accuracy_score(true_scenes, pred_scenes)),
        "confusion_matrix": confusion_matrix(
            joint_true,
            joint_pred,
            labels=np.arange(len(joint_class_names)),
        ).tolist(),
        "scene_selection": scene_selection,
        "timing": {
            "artifact_loading_seconds": float(artifact_loading_seconds),
            "model_loading_seconds": float(model_loading_seconds),
            "raw_feature_extraction_seconds": float(raw_feature_extraction_seconds),
            "data_preparation_seconds": float(data_preparation_seconds),
            "model_forward_seconds": float(model_forward_seconds),
            "label_decoding_seconds": float(label_decoding_seconds),
            "raw_to_label_seconds": float(raw_to_label_seconds),
            "raw_to_label_seconds_per_sample": safe_divide(raw_to_label_seconds, sample_count),
            "model_forward_seconds_per_sample": safe_divide(model_forward_seconds, sample_count),
            "sample_count": sample_count,
            "raw_to_label_seconds_definition": (
                "raw feature extraction + fresh feature loading/alignment + scaling + model forward "
                "+ label decoding, divided by sample_count for the per-sample value"
            ),
        },
        "command_records": command_records,
    }

    metrics_path = run_dir / "fresh_raw_inference_metrics.json"
    predictions_path = run_dir / "fresh_raw_inference_predictions.json"
    report_path = run_dir / "fresh_raw_inference_classification_report.txt"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")

    print(
        f"[fresh-result] n={sample_count} joint={result['joint_accuracy']:.4f} "
        f"intent={result['intent_accuracy']:.4f} scene={result['scene_accuracy']:.4f}"
    )
    print(
        "[fresh-timing] "
        f"raw_to_label={raw_to_label_seconds:.3f}s "
        f"avg_per_sample={safe_divide(raw_to_label_seconds, sample_count):.6f}s "
        f"model_forward={model_forward_seconds:.3f}s"
    )
    print(f"[fresh-output] metrics={metrics_path}")
    print(f"[fresh-output] predictions={predictions_path}")
    if args.show_report:
        print(report)


if __name__ == "__main__":
    main()
