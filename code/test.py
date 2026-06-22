from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from raw_pipeline import COURSE_TEST_VIDEO_NAMES, default_raw_cache_dir, prepare_raw_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = {
    "baseline": PROJECT_ROOT / "outputs" / "baseline_real_scene_perceiver_io_mptasks",
    "improved": PROJECT_ROOT / "outputs" / "improved_real_scene_anchor2_perceiver_io_mptasks",
}
CHECKPOINT_NAMES = {
    "baseline": "baseline_real_scene_perceiver_io.pt",
    "improved": "improved_real_scene_anchor2.pt",
}


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def configure_environment(args: argparse.Namespace, processed_data_dir: Path) -> None:
    os.environ["MM_INTENT_PROCESSED_DATA_DIR"] = str(processed_data_dir)
    os.environ["SMART_AR_TEST_VIDEO_NAMES"] = ",".join(args.video_names)
    if args.gesture_feature_dir:
        os.environ["MM_INTENT_GESTURE_FEATURE_DIR"] = str(Path(args.gesture_feature_dir).resolve())
    if args.audio_feature_dir:
        os.environ["MM_INTENT_AUDIO_FEATURE_DIR"] = str(Path(args.audio_feature_dir).resolve())
    if args.text_feature_dir:
        os.environ["MM_INTENT_TEXT_FEATURE_DIR"] = str(Path(args.text_feature_dir).resolve())
    if args.imu_feature_dir:
        os.environ["MM_INTENT_IMU_FEATURE_DIR"] = str(Path(args.imu_feature_dir).resolve())
    if args.gesture_feature_dim is not None:
        os.environ["MM_INTENT_GESTURE_FEAT_DIM"] = str(args.gesture_feature_dim)
    if args.audio_feature_dim is not None:
        os.environ["MM_INTENT_AUDIO_FEAT_DIM"] = str(args.audio_feature_dim)
    if args.text_feature_dim is not None:
        os.environ["MM_INTENT_TEXT_FEAT_DIM"] = str(args.text_feature_dim)
    if args.imu_feature_dim is not None:
        os.environ["MM_INTENT_IMU_FEAT_DIM"] = str(args.imu_feature_dim)
    if args.dataset_dir:
        os.environ["MM_INTENT_DATASET_DIR"] = str(Path(args.dataset_dir).resolve())
    if args.hololens_dir:
        os.environ["MM_INTENT_HOLOLENS_DIR"] = str(Path(args.hololens_dir).resolve())
    if args.fisheye_dir:
        os.environ["MM_INTENT_FISHEYE_DIR"] = str(Path(args.fisheye_dir).resolve())

    if args.noise_space == "raw":
        os.environ["MM_INTENT_RAW_NOISE_MODALITY"] = args.noise_modality or ""
        os.environ["MM_INTENT_RAW_NOISE_LEVEL"] = str(args.noise_level)
        os.environ["MM_INTENT_RAW_NOISE_SEED"] = str(args.noise_seed)
        os.environ.pop("SMART_AR_NOISE_MODALITY", None)
        os.environ["SMART_AR_NOISE_LEVEL"] = "0.0"
        os.environ["MM_INTENT_SCENE_CACHE_DIR"] = str(processed_data_dir / "scene_features")
    elif args.noise_modality:
        os.environ["SMART_AR_NOISE_MODALITY"] = args.noise_modality
        os.environ["SMART_AR_NOISE_LEVEL"] = str(args.noise_level)
        os.environ["SMART_AR_NOISE_SEED"] = str(args.noise_seed)


def build_model(
    model_name: str,
    checkpoint: dict[str, Any],
    joint_class_names: list[str],
    device: torch.device,
):
    import baseline_real_scene as base

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


@torch.no_grad()
def predict(model_name: str, model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    true_values: list[np.ndarray] = []
    predicted_values: list[np.ndarray] = []
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
    parser = argparse.ArgumentParser(
        description="Independent MM-Intent inference: raw/test data -> checkpoint -> intent predictions."
    )
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--scalers")
    parser.add_argument("--label-encoder")
    parser.add_argument("--input-mode", choices=("features", "raw"), default="features")
    parser.add_argument("--processed-data-dir")
    parser.add_argument("--raw-cache-dir")
    parser.add_argument(
        "--base-feature-dir",
        help="Clean raw-derived cache reused for modalities unchanged by raw noise.",
    )
    parser.add_argument("--gesture-representation", choices=("clip", "hand_geometry"), default="clip")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--hololens-dir")
    parser.add_argument("--fisheye-dir")
    parser.add_argument("--video-names", nargs="*", default=list(COURSE_TEST_VIDEO_NAMES))
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--preprocess-dry-run", action="store_true")
    parser.add_argument("--noise-modality", choices=("imu", "gesture", "audio", "text", "scene"))
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--noise-space", choices=("feature", "raw"))
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gesture-feature-dir")
    parser.add_argument("--audio-feature-dir")
    parser.add_argument("--text-feature-dir")
    parser.add_argument("--imu-feature-dir")
    parser.add_argument("--gesture-feature-dim", type=int)
    parser.add_argument("--audio-feature-dim", type=int)
    parser.add_argument("--text-feature-dim", type=int)
    parser.add_argument("--imu-feature-dim", type=int)
    parser.add_argument("--show-report", action="store_true")
    args = parser.parse_args()

    if not args.video_names:
        parser.error("--video-names must contain at least one video")
    if args.noise_space is None:
        args.noise_space = "raw" if args.input_mode == "raw" else "feature"
    if args.noise_space == "raw" and args.input_mode != "raw":
        parser.error("--noise-space raw requires --input-mode raw")
    if args.noise_level > 0.0 and not args.noise_modality:
        parser.error("--noise-modality is required when --noise-level is positive")
    if not 0.0 <= args.noise_level <= 1.0:
        parser.error("--noise-level must be in [0, 1]")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUTS[args.model]
    raw_cache_dir = (
        Path(args.raw_cache_dir).resolve()
        if args.raw_cache_dir
        else default_raw_cache_dir(args.noise_modality or "", args.noise_level, args.noise_seed)
    )
    processed_data_dir = (
        raw_cache_dir
        if args.input_mode == "raw"
        else Path(args.processed_data_dir).resolve()
        if args.processed_data_dir
        else (PROJECT_ROOT / "dataset" / "AR_Data_Process3.0" / "data_full").resolve()
    )
    if args.input_mode == "raw" and args.gesture_representation == "hand_geometry":
        args.gesture_feature_dir = str(raw_cache_dir / "hand_geometry_features")
        args.gesture_feature_dim = 96
    configure_environment(args, processed_data_dir)

    if args.input_mode == "raw":
        base_feature_dir = (
            Path(args.base_feature_dir).resolve()
            if args.base_feature_dir
            else (
                (Path(args.dataset_dir).resolve() if args.dataset_dir else PROJECT_ROOT / "dataset")
                / "AR_Data_Process3.0"
                / "data_full"
            ).resolve()
            if args.noise_modality and args.noise_level > 0.0
            else None
        )
        prepare_raw_features(
            output_dir=raw_cache_dir,
            video_names=args.video_names,
            noise_modality=args.noise_modality or "",
            noise_level=args.noise_level,
            noise_seed=args.noise_seed,
            base_env=os.environ.copy(),
            force=args.force_preprocess,
            dry_run=args.preprocess_dry_run,
            base_feature_dir=base_feature_dir,
            gesture_representation=args.gesture_representation,
        )
        if args.preprocess_dry_run:
            print("[test] preprocessing dry-run complete; checkpoint inference was not started.")
            return

    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else output_dir / CHECKPOINT_NAMES[args.model]
    scalers_path = Path(args.scalers).resolve() if args.scalers else output_dir / "scalers.pkl"
    label_encoder_path = (
        Path(args.label_encoder).resolve() if args.label_encoder else output_dir / "label_encoder.pkl"
    )
    for required_path in (checkpoint_path, scalers_path, label_encoder_path):
        if not required_path.exists():
            raise SystemExit(f"Missing independent-test artifact: {required_path}")

    import baseline_real_scene as base
    from real_scene_utils import RealSceneFeatureCache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with scalers_path.open("rb") as file:
        scalers = pickle.load(file)
    with label_encoder_path.open("rb") as file:
        label_encoder = pickle.load(file)
    checkpoint = load_checkpoint(checkpoint_path, device)

    print(f"[test] model={args.model} input_mode={args.input_mode} device={device}")
    print(f"[test] checkpoint={checkpoint_path}")
    print(f"[test] videos={len(args.video_names)} processed_data={processed_data_dir}")
    scene_cache = RealSceneFeatureCache(base.SCENE_CACHE_DIR)
    raw_features, intent_targets, scene_targets, _scene_selection = base.load_multimodal_data(
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

    joint_class_names = label_encoder.classes_.tolist()
    model = build_model(args.model, checkpoint, joint_class_names, device)
    joint_true, joint_pred = predict(args.model, model, loader, device)
    true_names = label_encoder.inverse_transform(joint_true)
    pred_names = label_encoder.inverse_transform(joint_pred)
    true_scenes, true_intents = split_joint_names(true_names)
    pred_scenes, pred_intents = split_joint_names(pred_names)

    report = classification_report(
        joint_true,
        joint_pred,
        labels=np.arange(len(joint_class_names)),
        target_names=joint_class_names,
        zero_division=0,
    )
    result = {
        "model": args.model,
        "input_mode": args.input_mode,
        "checkpoint": str(checkpoint_path),
        "processed_data_dir": str(processed_data_dir),
        "video_names": args.video_names,
        "raw_noise": {
            "modality": args.noise_modality if args.noise_space == "raw" else "",
            "level": args.noise_level if args.noise_space == "raw" else 0.0,
            "seed": args.noise_seed,
        },
        "sample_count": int(len(joint_true)),
        "joint_accuracy": float(accuracy_score(joint_true, joint_pred)),
        "intent_accuracy": float(accuracy_score(true_intents, pred_intents)),
        "scene_accuracy": float(accuracy_score(true_scenes, pred_scenes)),
        "confusion_matrix": confusion_matrix(
            joint_true,
            joint_pred,
            labels=np.arange(len(joint_class_names)),
        ).tolist(),
    }
    predictions = [
        {
            "sample_index": index,
            "true_joint": str(true_name),
            "predicted_joint": str(pred_name),
            "true_intent": str(true_intent),
            "predicted_intent": str(pred_intent),
            "true_scene": str(true_scene),
            "predicted_scene": str(pred_scene),
        }
        for index, (true_name, pred_name, true_intent, pred_intent, true_scene, pred_scene) in enumerate(
            zip(true_names, pred_names, true_intents, pred_intents, true_scenes, pred_scenes)
        )
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "independent_test_metrics.json"
    predictions_path = output_dir / "independent_test_predictions.json"
    report_path = output_dir / "independent_test_classification_report.txt"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")

    print(
        f"[test-result] n={result['sample_count']} joint={result['joint_accuracy']:.4f} "
        f"intent={result['intent_accuracy']:.4f} scene={result['scene_accuracy']:.4f}"
    )
    print(f"[test-output] metrics={metrics_path}")
    print(f"[test-output] predictions={predictions_path}")
    if args.show_report:
        print(report)


if __name__ == "__main__":
    main()
