from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from project_paths import MODEL_OUTPUT_ROOT


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def configure_from_metrics(output_dir: Path) -> None:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return
    config = load_json(metrics_path).get("config", {})
    for key, env_name in {
        "model_dim": "IMPROVED_REAL_SCENE_A2_MODEL_DIM",
        "num_latents": "IMPROVED_REAL_SCENE_A2_NUM_LATENTS",
        "depth": "IMPROVED_REAL_SCENE_A2_DEPTH",
        "num_heads": "IMPROVED_REAL_SCENE_A2_NUM_HEADS",
        "dropout": "IMPROVED_REAL_SCENE_A2_DROPOUT",
        "min_gate": "IMPROVED_REAL_SCENE_A2_MIN_GATE",
        "fallback_max_gate": "IMPROVED_REAL_SCENE_A2_FALLBACK_MAX_GATE",
    }.items():
        if key in config:
            os.environ[env_name] = str(config[key])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(pred == true)) if len(true) else 0.0


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norm, 1e-12, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prototype-assisted logits for a trained improved checkpoint.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-dir")
    parser.add_argument("--name")
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--use-val", action="store_true", help="Build prototypes from train+val instead of train only.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    name = args.name or output_dir.name
    analysis_dir = Path(args.analysis_dir or (MODEL_OUTPUT_ROOT / "prototype_eval" / name)).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    configure_from_metrics(output_dir)

    import torch

    import baseline_real_scene as base
    import train_and_test as improved

    with (output_dir / "label_encoder.pkl").open("rb") as file:
        label_encoder = pickle.load(file)
    with (output_dir / "scalers.pkl").open("rb") as file:
        scalers = pickle.load(file)

    scene_cache = base.RealSceneFeatureCache(base.SCENE_CACHE_DIR)
    train_raw_features, train_raw_labels, train_raw_scene_targets, _ = base.load_multimodal_data(base.TRAIN_VIDEO_NAMES, scene_cache)
    train_joint_raw = base.build_joint_labels(train_raw_labels, train_raw_scene_targets)
    (
        train_features_raw,
        val_features_raw,
        y_train_intent,
        y_val_intent,
        y_train_scene,
        y_val_scene,
        y_train_joint_raw,
        y_val_joint_raw,
    ) = base.split_train_val(train_raw_features, train_raw_labels, train_raw_scene_targets, train_joint_raw)
    if args.use_val:
        proto_features_raw = {
            key: np.concatenate([train_features_raw[key], val_features_raw[key]], axis=0)
            for key in train_features_raw
        }
        proto_joint_raw = np.concatenate([y_train_joint_raw, y_val_joint_raw], axis=0)
        proto_intent = np.concatenate([y_train_intent, y_val_intent], axis=0)
        proto_scene = np.concatenate([y_train_scene, y_val_scene], axis=0)
    else:
        proto_features_raw = train_features_raw
        proto_joint_raw = y_train_joint_raw
        proto_intent = y_train_intent
        proto_scene = y_train_scene

    test_features_raw, test_intent, test_scene, _ = base.load_multimodal_data(base.TEST_VIDEO_NAMES, scene_cache)
    test_joint_raw = base.build_joint_labels(test_intent, test_scene)

    proto_features = base.apply_scalers(proto_features_raw, scalers)
    test_features = base.apply_scalers(test_features_raw, scalers)
    proto_joint = label_encoder.transform(proto_joint_raw)
    test_joint = label_encoder.transform(test_joint_raw)

    joint_class_names = label_encoder.classes_.tolist()
    intent_class_names = [base.INTENT_NAMES[index] for index in sorted(base.INTENT_NAMES)]
    scene_class_names = [base.SCENE_ID_TO_NAME[index] for index in range(len(base.SCENE_ID_TO_NAME))]

    checkpoint = torch.load(output_dir / "improved_real_scene_anchor2.pt", map_location=base.DEVICE, weights_only=False)
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
    report = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"[load] missing_keys={len(report.missing_keys)} unexpected_keys={len(report.unexpected_keys)}")
    model.eval()

    def collect(features: dict[str, np.ndarray], joint: np.ndarray, intent: np.ndarray, scene: np.ndarray):
        loader = improved.make_loader(features, joint, intent, scene, args.batch_size, shuffle=False)
        embeddings: list[np.ndarray] = []
        logits: list[np.ndarray] = []
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
                embeddings.append(outputs["fused_embedding"].cpu().numpy())
                logits.append(outputs["joint_logits"].cpu().numpy())
        return normalize_rows(np.concatenate(embeddings, axis=0)), np.concatenate(logits, axis=0)

    proto_embeddings, _ = collect(proto_features, proto_joint, proto_intent, proto_scene)
    test_embeddings, test_logits = collect(test_features, test_joint, test_intent, test_scene)
    prototypes = np.zeros((len(joint_class_names), proto_embeddings.shape[1]), dtype=np.float32)
    for class_index in range(len(joint_class_names)):
        mask = proto_joint == class_index
        if not np.any(mask):
            continue
        prototypes[class_index] = proto_embeddings[mask].mean(axis=0)
    prototypes = normalize_rows(prototypes)
    similarity_logits = test_embeddings @ prototypes.T

    rows = []
    for alpha in args.alphas:
        combined = test_logits + alpha * similarity_logits
        pred = combined.argmax(axis=1)
        pred_names = label_encoder.inverse_transform(pred)
        true_names = label_encoder.inverse_transform(test_joint)
        pred_intent = np.array([base.split_joint_label(value)[1] for value in pred_names], dtype=object)
        true_intent = np.array([base.split_joint_label(value)[1] for value in true_names], dtype=object)
        pred_scene = np.array([base.split_joint_label(value)[0] for value in pred_names], dtype=object)
        true_scene = np.array([base.split_joint_label(value)[0] for value in true_names], dtype=object)
        rows.append(
            {
                "name": name,
                "alpha": alpha,
                "use_val": int(args.use_val),
                "joint_acc": accuracy(pred, test_joint),
                "intent_acc": accuracy(pred_intent, true_intent),
                "scene_acc": accuracy(pred_scene, true_scene),
            }
        )

    write_csv(analysis_dir / "prototype_alpha_sweep.csv", rows)
    best = max(rows, key=lambda row: (float(row["joint_acc"]), float(row["intent_acc"]), float(row["scene_acc"])))
    with (analysis_dir / "prototype_summary.json").open("w", encoding="utf-8") as file:
        json.dump({"best": best, "rows": rows}, file, indent=2, ensure_ascii=False)
    print("[prototype]")
    print(json.dumps({"best": best}, indent=2, ensure_ascii=False))
    print(f"[saved] {analysis_dir}")


if __name__ == "__main__":
    main()
