from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from project_paths import MODEL_OUTPUT_ROOT, PROCESSED_DATA_DIR


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_from_metrics(output_dir: Path) -> None:
    config = load_json(output_dir / "metrics.json").get("config", {})
    env_map = {
        "model_dim": "IMPROVED_REAL_SCENE_A2_MODEL_DIM",
        "num_latents": "IMPROVED_REAL_SCENE_A2_NUM_LATENTS",
        "depth": "IMPROVED_REAL_SCENE_A2_DEPTH",
        "num_heads": "IMPROVED_REAL_SCENE_A2_NUM_HEADS",
        "dropout": "IMPROVED_REAL_SCENE_A2_DROPOUT",
        "min_gate": "IMPROVED_REAL_SCENE_A2_MIN_GATE",
        "fallback_max_gate": "IMPROVED_REAL_SCENE_A2_FALLBACK_MAX_GATE",
    }
    for key, env_name in env_map.items():
        if key in config:
            os.environ[env_name] = str(config[key])


def accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(prediction == target)) if len(target) else 0.0


def collect_logits(
    output_dir: Path,
    gesture_dir: Path,
    gesture_dim: int,
    missing_modalities: tuple[str, ...],
    batch_size: int,
):
    import torch

    import baseline_real_scene as base
    import train_and_test as improved

    base.STRONG_GESTURE_DIR = gesture_dir
    base.GESTURE_FEAT_DIM = gesture_dim
    base.MISSING_MODALITIES = missing_modalities
    base.NOISE_MODALITY = ""
    base.NOISE_LEVEL = 0.0

    with (output_dir / "label_encoder.pkl").open("rb") as file:
        label_encoder = pickle.load(file)
    with (output_dir / "scalers.pkl").open("rb") as file:
        scalers = pickle.load(file)

    scene_cache = base.RealSceneFeatureCache(base.SCENE_CACHE_DIR)
    raw_features, intent_targets, scene_targets, _ = base.load_multimodal_data(base.TEST_VIDEO_NAMES, scene_cache)
    features = base.apply_scalers(raw_features, scalers)
    joint_names = base.build_joint_labels(intent_targets, scene_targets)
    joint_targets = label_encoder.transform(joint_names)
    class_names = label_encoder.classes_.tolist()

    loader = improved.make_loader(
        features,
        joint_targets,
        intent_targets,
        scene_targets,
        batch_size,
        shuffle=False,
    )
    checkpoint = torch.load(output_dir / "improved_real_scene_anchor2.pt", map_location=base.DEVICE, weights_only=False)
    model = improved.Anchor2PerceiverIO(
        num_joint_classes=len(class_names),
        num_intent_classes=len(base.INTENT_NAMES),
        num_scene_classes=len(base.SCENE_ID_TO_NAME),
        joint_class_names=class_names,
        model_dim=int(checkpoint.get("model_dim", improved.MODEL_DIM)),
        num_latents=int(checkpoint.get("num_latents", improved.NUM_LATENTS)),
        depth=int(checkpoint.get("depth", improved.DEPTH)),
        num_heads=int(checkpoint.get("num_heads", improved.NUM_HEADS)),
        dropout=float(checkpoint.get("dropout", improved.DROPOUT)),
        min_gate=improved.MIN_GATE,
    ).to(base.DEVICE)
    report = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(
        f"[load] {output_dir.name} gesture_dim={gesture_dim} "
        f"missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)}"
    )
    model.eval()

    intent_logits: list[np.ndarray] = []
    scene_logits: list[np.ndarray] = []
    joint_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            imu, gesture, audio, text, scene, *_ = batch
            outputs = model(
                imu.to(base.DEVICE),
                gesture.to(base.DEVICE),
                audio.to(base.DEVICE),
                text.to(base.DEVICE),
                scene.to(base.DEVICE),
            )
            intent_logits.append(outputs["intent_logits"].cpu().numpy())
            scene_logits.append(outputs["scene_logits"].cpu().numpy())
            joint_logits.append(outputs["joint_logits"].cpu().numpy())

    return {
        "intent_logits": np.concatenate(intent_logits),
        "scene_logits": np.concatenate(scene_logits),
        "joint_logits": np.concatenate(joint_logits),
        "intent_targets": np.asarray(intent_targets),
        "scene_targets": np.asarray(scene_targets),
        "joint_targets": np.asarray(joint_targets),
        "joint_names": np.asarray(joint_names, dtype=object),
        "class_names": class_names,
        "label_encoder": label_encoder,
    }


def evaluate_scenario(
    scenario: str,
    geometry: dict[str, Any],
    scene_model: dict[str, Any],
    scene_weights: list[float],
    intent_weights: list[float],
) -> list[dict[str, object]]:
    if not np.array_equal(geometry["intent_targets"], scene_model["intent_targets"]):
        raise ValueError("Intent target order differs between checkpoints")
    if not np.array_equal(geometry["scene_targets"], scene_model["scene_targets"]):
        raise ValueError("Scene target order differs between checkpoints")
    if geometry["class_names"] != scene_model["class_names"]:
        raise ValueError("Joint class order differs between checkpoints")

    import baseline_real_scene as base

    class_names = geometry["class_names"]
    scene_name_to_id = {name: index for index, name in base.SCENE_ID_TO_NAME.items()}
    intent_name_to_id = {name: index for index, name in base.INTENT_NAMES.items()}
    joint_scene_index = np.asarray(
        [scene_name_to_id[base.split_joint_label(name)[0]] for name in class_names], dtype=np.int64
    )
    joint_intent_index = np.asarray(
        [intent_name_to_id[base.split_joint_label(name)[1]] for name in class_names], dtype=np.int64
    )

    intent_targets = geometry["intent_targets"]
    scene_targets = geometry["scene_targets"]
    joint_targets = geometry["joint_targets"]
    geometry_intent = geometry["intent_logits"]
    baseline_scene = scene_model["scene_logits"]

    rows: list[dict[str, object]] = []
    for intent_weight in intent_weights:
        for scene_weight in scene_weights:
            combined = (
                intent_weight * geometry_intent[:, joint_intent_index]
                + scene_weight * baseline_scene[:, joint_scene_index]
            )
            joint_pred = combined.argmax(axis=1)
            rows.append(
                {
                    "scenario": scenario,
                    "intent_weight": intent_weight,
                    "scene_weight": scene_weight,
                    "joint_acc": accuracy(joint_pred, joint_targets),
                    "intent_acc": accuracy(geometry_intent.argmax(axis=1), intent_targets),
                    "scene_acc": accuracy(baseline_scene.argmax(axis=1), scene_targets),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine geometry intent logits with improved-baseline scene logits.")
    parser.add_argument(
        "--geometry-output-dir",
        default=str(MODEL_OUTPUT_ROOT / "feature_suite" / "hand_geometry" / "main"),
    )
    parser.add_argument(
        "--scene-output-dir",
        default=str(MODEL_OUTPUT_ROOT / "improved_real_scene_anchor2_perceiver_io_mptasks"),
    )
    parser.add_argument(
        "--geometry-feature-dir",
        default=str(PROCESSED_DATA_DIR / "hand_geometry_features"),
    )
    parser.add_argument(
        "--scene-gesture-feature-dir",
        default=str(PROCESSED_DATA_DIR / "strong_gesture_features"),
    )
    parser.add_argument("--geometry-feature-dim", type=int, default=96)
    parser.add_argument("--scene-gesture-feature-dim", type=int, default=768)
    parser.add_argument("--intent-weights", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    parser.add_argument("--scene-weights", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.25])
    parser.add_argument("--scenarios", nargs="+", choices=("full", "no_scene"), default=["full", "no_scene"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--analysis-dir",
        default=str(MODEL_OUTPUT_ROOT / "factorized_head_fusion"),
    )
    args = parser.parse_args()

    geometry_output_dir = Path(args.geometry_output_dir).resolve()
    scene_output_dir = Path(args.scene_output_dir).resolve()
    analysis_dir = Path(args.analysis_dir).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    configure_from_metrics(geometry_output_dir)

    all_rows: list[dict[str, object]] = []
    for scenario in args.scenarios:
        missing = ("scene",) if scenario == "no_scene" else ()
        print(f"[scenario] {scenario}")
        geometry = collect_logits(
            geometry_output_dir,
            Path(args.geometry_feature_dir).resolve(),
            args.geometry_feature_dim,
            missing,
            args.batch_size,
        )
        scene_model = collect_logits(
            scene_output_dir,
            Path(args.scene_gesture_feature_dir).resolve(),
            args.scene_gesture_feature_dim,
            missing,
            args.batch_size,
        )
        rows = evaluate_scenario(
            scenario,
            geometry,
            scene_model,
            args.scene_weights,
            args.intent_weights,
        )
        all_rows.extend(rows)
        best = max(rows, key=lambda row: float(row["joint_acc"]))
        print("[best]", json.dumps(best, ensure_ascii=False))

    csv_path = analysis_dir / "factorized_head_fusion.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["scenario", "intent_weight", "scene_weight", "joint_acc", "intent_acc", "scene_acc"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "geometry_output_dir": str(geometry_output_dir),
        "scene_output_dir": str(scene_output_dir),
        "best_by_scenario": {
            scenario: max(
                (row for row in all_rows if row["scenario"] == scenario),
                key=lambda row: float(row["joint_acc"]),
            )
            for scenario in args.scenarios
        },
    }
    (analysis_dir / "factorized_head_fusion.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {analysis_dir}")


if __name__ == "__main__":
    main()
