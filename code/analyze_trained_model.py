from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def configure_from_metrics(output_dir: Path) -> None:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return
    config = load_json(metrics_path).get("config", {})
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


def set_missing_modalities(modalities: list[str]) -> None:
    if modalities:
        os.environ["SMART_AR_MISSING_MODALITIES"] = ",".join(modalities)
    else:
        os.environ.pop("SMART_AR_MISSING_MODALITIES", None)
    os.environ.pop("SMART_AR_NOISE_MODALITY", None)
    os.environ.pop("SMART_AR_NOISE_LEVEL", None)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)


def calibration_bins(confidence: np.ndarray, correct: np.ndarray, num_bins: int) -> tuple[list[dict[str, object]], float, float]:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    rows: list[dict[str, object]] = []
    ece = 0.0
    mce = 0.0
    total = max(len(confidence), 1)
    for index in range(num_bins):
        left = edges[index]
        right = edges[index + 1]
        if index == num_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        count = int(mask.sum())
        if count:
            bin_acc = float(correct[mask].mean())
            bin_conf = float(confidence[mask].mean())
            gap = abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = 0.0
            gap = 0.0
        ece += count / total * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin": index,
                "left": float(left),
                "right": float(right),
                "count": count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "gap": float(gap),
            }
        )
    return rows, float(ece), float(mce)


def save_confidence_plot(path: Path, confidence: np.ndarray, correct: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(confidence[correct], bins=20, alpha=0.75, label="correct")
    plt.hist(confidence[~correct], bins=20, alpha=0.75, label="wrong")
    plt.xlabel("max softmax confidence")
    plt.ylabel("count")
    plt.title("Confidence Distribution")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_reliability_plot(path: Path, rows: list[dict[str, object]], ece: float) -> None:
    import matplotlib.pyplot as plt

    centers = [(float(row["left"]) + float(row["right"])) / 2 for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    confidences = [float(row["confidence"]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    width = 1.0 / max(len(rows), 1) * 0.85

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(centers, accuracies, width=width, alpha=0.75, label="accuracy")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    ax1.scatter(centers, confidences, color="#d62728", label="confidence", zorder=3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("confidence bin")
    ax1.set_ylabel("accuracy / confidence")
    ax1.set_title(f"Reliability Diagram (ECE={ece:.4f})")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(centers, counts, color="#555555", alpha=0.45, linewidth=1.5, label="count")
    ax2.set_ylabel("count")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_bar(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    top_rows = rows[:12]
    labels = [f"{row['true']} -> {row['pred']}" for row in top_rows]
    counts = [int(row["count"]) for row in top_rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, max(4, len(top_rows) * 0.38)))
    y = np.arange(len(top_rows))
    plt.barh(y, counts)
    plt.yticks(y, labels)
    plt.xlabel("count")
    plt.title("Top Confusions")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a trained improved-model checkpoint on the test split.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-dir")
    parser.add_argument("--name")
    parser.add_argument("--missing-modalities", nargs="*", default=[])
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    name = args.name or output_dir.name
    analysis_dir = Path(args.analysis_dir or (MODEL_OUTPUT_ROOT / "diagnostics" / name)).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    configure_from_metrics(output_dir)
    set_missing_modalities(args.missing_modalities)

    import torch

    import baseline_real_scene as base
    import train_and_test as improved

    label_encoder_path = output_dir / "label_encoder.pkl"
    scalers_path = output_dir / "scalers.pkl"
    checkpoint_path = output_dir / "improved_real_scene_anchor2.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")
    if not label_encoder_path.exists():
        raise SystemExit(f"Missing label encoder: {label_encoder_path}")
    if not scalers_path.exists():
        raise SystemExit(f"Missing scalers: {scalers_path}")

    with label_encoder_path.open("rb") as file:
        label_encoder = pickle.load(file)
    with scalers_path.open("rb") as file:
        scalers = pickle.load(file)

    scene_cache = base.RealSceneFeatureCache(base.SCENE_CACHE_DIR)
    test_features_raw, test_labels, test_scene_targets, _ = base.load_multimodal_data(base.TEST_VIDEO_NAMES, scene_cache)
    test_features = base.apply_scalers(test_features_raw, scalers)
    test_joint_raw = base.build_joint_labels(test_labels, test_scene_targets)
    test_joint = label_encoder.transform(test_joint_raw)
    joint_class_names = label_encoder.classes_.tolist()
    intent_class_names = [base.INTENT_NAMES[index] for index in sorted(base.INTENT_NAMES)]
    scene_class_names = [base.SCENE_ID_TO_NAME[index] for index in range(len(base.SCENE_ID_TO_NAME))]

    loader = improved.make_loader(
        test_features,
        test_joint,
        test_labels,
        test_scene_targets,
        args.batch_size,
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
    print(f"[load] missing_keys={len(load_report.missing_keys)} unexpected_keys={len(load_report.unexpected_keys)}")
    model.eval()

    joint_probs_list: list[np.ndarray] = []
    intent_probs_list: list[np.ndarray] = []
    scene_probs_list: list[np.ndarray] = []
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
            joint_probs_list.append(torch.softmax(outputs["joint_logits"], dim=1).cpu().numpy())
            intent_probs_list.append(torch.softmax(outputs["intent_logits"], dim=1).cpu().numpy())
            scene_probs_list.append(torch.softmax(outputs["scene_logits"], dim=1).cpu().numpy())

    joint_probs = np.concatenate(joint_probs_list, axis=0)
    intent_probs = np.concatenate(intent_probs_list, axis=0)
    scene_probs = np.concatenate(scene_probs_list, axis=0)
    joint_pred = joint_probs.argmax(axis=1)
    intent_pred = intent_probs.argmax(axis=1)
    scene_pred = scene_probs.argmax(axis=1)
    joint_conf = joint_probs.max(axis=1)
    intent_conf = intent_probs.max(axis=1)
    scene_conf = scene_probs.max(axis=1)
    joint_correct = joint_pred == test_joint
    intent_correct = intent_pred == test_labels
    scene_correct = scene_pred == test_scene_targets

    joint_true_names = label_encoder.inverse_transform(test_joint)
    joint_pred_names = label_encoder.inverse_transform(joint_pred)
    intent_true_names = np.array([base.INTENT_NAMES[int(value)] for value in test_labels], dtype=object)
    intent_pred_names = np.array([base.INTENT_NAMES[int(value)] for value in intent_pred], dtype=object)
    scene_true_names = np.array([base.SCENE_ID_TO_NAME[int(value)] for value in test_scene_targets], dtype=object)
    scene_pred_names = np.array([base.SCENE_ID_TO_NAME[int(value)] for value in scene_pred], dtype=object)

    prediction_rows = []
    for index in range(len(test_joint)):
        prediction_rows.append(
            {
                "sample_index": index,
                "joint_true": str(joint_true_names[index]),
                "joint_pred": str(joint_pred_names[index]),
                "joint_correct": int(joint_correct[index]),
                "joint_confidence": float(joint_conf[index]),
                "joint_entropy": float(entropy(joint_probs[index : index + 1])[0]),
                "intent_true": str(intent_true_names[index]),
                "intent_pred": str(intent_pred_names[index]),
                "intent_correct": int(intent_correct[index]),
                "intent_confidence": float(intent_conf[index]),
                "scene_true": str(scene_true_names[index]),
                "scene_pred": str(scene_pred_names[index]),
                "scene_correct": int(scene_correct[index]),
                "scene_confidence": float(scene_conf[index]),
            }
        )
    write_csv(
        analysis_dir / "prediction_details.csv",
        prediction_rows,
        list(prediction_rows[0].keys()) if prediction_rows else [],
    )

    class_rows = []
    for class_index, class_name in enumerate(joint_class_names):
        mask = test_joint == class_index
        pred_mask = joint_pred == class_index
        tp = int((mask & pred_mask).sum())
        support = int(mask.sum())
        predicted = int(pred_mask.sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        class_rows.append(
            {
                "class": class_name,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "avg_confidence": float(joint_conf[mask].mean()) if support else 0.0,
                "wrong_count": int((mask & ~joint_correct).sum()),
            }
        )
    write_csv(analysis_dir / "per_class_summary.csv", class_rows, list(class_rows[0].keys()))

    confusion_counts: dict[tuple[str, str], int] = {}
    for true_name, pred_name, is_correct in zip(joint_true_names, joint_pred_names, joint_correct):
        if is_correct:
            continue
        key = (str(true_name), str(pred_name))
        confusion_counts[key] = confusion_counts.get(key, 0) + 1
    confusion_rows = [
        {"true": true, "pred": pred, "count": count}
        for (true, pred), count in sorted(confusion_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    write_csv(analysis_dir / "top_confusions.csv", confusion_rows, ["true", "pred", "count"])

    joint_bins, joint_ece, joint_mce = calibration_bins(joint_conf, joint_correct, args.bins)
    intent_bins, intent_ece, intent_mce = calibration_bins(intent_conf, intent_correct, args.bins)
    scene_bins, scene_ece, scene_mce = calibration_bins(scene_conf, scene_correct, args.bins)
    write_csv(analysis_dir / "joint_calibration_bins.csv", joint_bins, list(joint_bins[0].keys()))
    write_csv(analysis_dir / "intent_calibration_bins.csv", intent_bins, list(intent_bins[0].keys()))
    write_csv(analysis_dir / "scene_calibration_bins.csv", scene_bins, list(scene_bins[0].keys()))

    save_confidence_plot(analysis_dir / "joint_confidence_hist.png", joint_conf, joint_correct)
    save_reliability_plot(analysis_dir / "joint_reliability.png", joint_bins, joint_ece)
    save_confidence_plot(analysis_dir / "intent_confidence_hist.png", intent_conf, intent_correct)
    save_reliability_plot(analysis_dir / "intent_reliability.png", intent_bins, intent_ece)
    save_confidence_plot(analysis_dir / "scene_confidence_hist.png", scene_conf, scene_correct)
    save_reliability_plot(analysis_dir / "scene_reliability.png", scene_bins, scene_ece)
    if confusion_rows:
        save_confusion_bar(analysis_dir / "top_confusions.png", confusion_rows)

    summary = {
        "name": name,
        "output_dir": str(output_dir),
        "missing_modalities": args.missing_modalities,
        "samples": int(len(test_joint)),
        "joint_acc": float(joint_correct.mean()),
        "intent_acc": float(intent_correct.mean()),
        "scene_acc": float(scene_correct.mean()),
        "joint_ece": joint_ece,
        "joint_mce": joint_mce,
        "intent_ece": intent_ece,
        "intent_mce": intent_mce,
        "scene_ece": scene_ece,
        "scene_mce": scene_mce,
        "avg_joint_confidence": float(joint_conf.mean()),
        "avg_wrong_joint_confidence": float(joint_conf[~joint_correct].mean()) if np.any(~joint_correct) else math.nan,
        "top_confusions": confusion_rows[:10],
    }
    with (analysis_dir / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("[analysis]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[saved] {analysis_dir}")


if __name__ == "__main__":
    main()
