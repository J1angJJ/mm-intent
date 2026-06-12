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


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(pred == true)) if len(true) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate logits averaged from multiple improved checkpoints.")
    parser.add_argument("--output-dirs", nargs="+", required=True)
    parser.add_argument("--analysis-dir", default=str(MODEL_OUTPUT_ROOT / "ensemble_eval"))
    parser.add_argument("--name", default="checkpoint_ensemble")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    output_dirs = [Path(value).resolve() for value in args.output_dirs]
    analysis_dir = Path(args.analysis_dir).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    configure_from_metrics(output_dirs[0])

    import torch

    import baseline_real_scene as base
    import train_and_test as improved

    with (output_dirs[0] / "label_encoder.pkl").open("rb") as file:
        label_encoder = pickle.load(file)
    with (output_dirs[0] / "scalers.pkl").open("rb") as file:
        scalers = pickle.load(file)

    scene_cache = base.RealSceneFeatureCache(base.SCENE_CACHE_DIR)
    test_features_raw, test_intent, test_scene, _ = base.load_multimodal_data(base.TEST_VIDEO_NAMES, scene_cache)
    test_features = base.apply_scalers(test_features_raw, scalers)
    test_joint_raw = base.build_joint_labels(test_intent, test_scene)
    test_joint = label_encoder.transform(test_joint_raw)

    joint_class_names = label_encoder.classes_.tolist()
    intent_class_names = [base.INTENT_NAMES[index] for index in sorted(base.INTENT_NAMES)]
    scene_class_names = [base.SCENE_ID_TO_NAME[index] for index in range(len(base.SCENE_ID_TO_NAME))]
    loader = improved.make_loader(test_features, test_joint, test_intent, test_scene, args.batch_size, shuffle=False)

    all_joint_logits = []
    all_intent_logits = []
    all_scene_logits = []
    for output_dir in output_dirs:
        with (output_dir / "label_encoder.pkl").open("rb") as file:
            current_encoder = pickle.load(file)
        if list(current_encoder.classes_) != list(label_encoder.classes_):
            raise SystemExit(f"Label encoder mismatch: {output_dir}")

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
        print(f"[load] {output_dir} missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)}")
        model.eval()

        joint_logits = []
        intent_logits = []
        scene_logits = []
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
                joint_logits.append(outputs["joint_logits"].cpu().numpy())
                intent_logits.append(outputs["intent_logits"].cpu().numpy())
                scene_logits.append(outputs["scene_logits"].cpu().numpy())
        all_joint_logits.append(np.concatenate(joint_logits, axis=0))
        all_intent_logits.append(np.concatenate(intent_logits, axis=0))
        all_scene_logits.append(np.concatenate(scene_logits, axis=0))

    mean_joint_logits = np.mean(np.stack(all_joint_logits, axis=0), axis=0)
    mean_intent_logits = np.mean(np.stack(all_intent_logits, axis=0), axis=0)
    mean_scene_logits = np.mean(np.stack(all_scene_logits, axis=0), axis=0)
    joint_pred = mean_joint_logits.argmax(axis=1)
    intent_pred = mean_intent_logits.argmax(axis=1)
    scene_pred = mean_scene_logits.argmax(axis=1)

    joint_true_names = label_encoder.inverse_transform(test_joint)
    joint_pred_names = label_encoder.inverse_transform(joint_pred)
    intent_true_names = np.array([base.split_joint_label(value)[1] for value in joint_true_names], dtype=object)
    intent_pred_names = np.array([base.split_joint_label(value)[1] for value in joint_pred_names], dtype=object)
    scene_true_names = np.array([base.split_joint_label(value)[0] for value in joint_true_names], dtype=object)
    scene_pred_names = np.array([base.split_joint_label(value)[0] for value in joint_pred_names], dtype=object)

    single_rows = []
    for index, output_dir in enumerate(output_dirs):
        pred = all_joint_logits[index].argmax(axis=1)
        pred_names = label_encoder.inverse_transform(pred)
        pred_intent_names = np.array([base.split_joint_label(value)[1] for value in pred_names], dtype=object)
        pred_scene_names = np.array([base.split_joint_label(value)[0] for value in pred_names], dtype=object)
        single_rows.append(
            {
                "model": str(output_dir),
                "joint_acc": accuracy(pred, test_joint),
                "intent_acc": accuracy(pred_intent_names, intent_true_names),
                "scene_acc": accuracy(pred_scene_names, scene_true_names),
            }
        )

    summary = {
        "name": args.name,
        "models": [str(path) for path in output_dirs],
        "ensemble": {
            "joint_acc": accuracy(joint_pred, test_joint),
            "intent_acc": accuracy(intent_pred_names, intent_true_names),
            "scene_acc": accuracy(scene_pred_names, scene_true_names),
        },
        "single_models": single_rows,
    }
    with (analysis_dir / "ensemble_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    with (analysis_dir / "ensemble_summary.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["model", "joint_acc", "intent_acc", "scene_acc"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(single_rows)
        writer.writerow({"model": "ENSEMBLE", **summary["ensemble"]})
    print("[ensemble]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {analysis_dir}")


if __name__ == "__main__":
    main()
