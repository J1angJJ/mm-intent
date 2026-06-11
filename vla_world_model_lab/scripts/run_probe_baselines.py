from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    ARTIFACT_ROOT,
    MODALITIES,
    feature_paths,
    load_csv,
    require_existing,
    write_csv,
    write_json,
)


def pool_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array.astype(np.float32)
    if array.ndim >= 2:
        return array.reshape(array.shape[0], -1).mean(axis=0).astype(np.float32)
    return np.asarray([float(array)], dtype=np.float32)


def load_modality_vector(video_name: str, segment_index: int, modality: str) -> np.ndarray:
    path = feature_paths(video_name)[modality]
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path, allow_pickle=True)
    value = array[segment_index]
    if modality == "audio" and isinstance(value, dict):
        value = value["feature"]
    return pool_array(value)


def build_matrix(rows: list[dict[str, str]], modalities: tuple[str, ...]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for row in rows:
        parts = [
            load_modality_vector(row["video_name"], int(row["segment_index"]), modality)
            for modality in modalities
        ]
        vectors.append(np.concatenate(parts, axis=0))
    return np.stack(vectors).astype(np.float32)


def target_values(rows: list[dict[str, str]], target: str) -> np.ndarray:
    return np.array([row[target] for row in rows], dtype=object)


def run_probe(
    rows: list[dict[str, str]],
    target: str,
    modalities: tuple[str, ...],
    max_iter: int,
) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not train_rows or not test_rows:
        raise ValueError("Need both train and test rows.")

    x_train = build_matrix(train_rows, modalities)
    x_test = build_matrix(test_rows, modalities)
    y_train = target_values(train_rows, target)
    y_test = target_values(test_rows, target)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, class_weight="balanced", n_jobs=None),
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return {
        "target": target,
        "modalities": list(modalities),
        "train_samples": int(len(train_rows)),
        "test_samples": int(len(test_rows)),
        "feature_dim": int(x_train.shape[1]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }


def parse_modalities(value: str) -> tuple[str, ...]:
    if value == "all":
        return MODALITIES
    modalities = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(modalities) - set(MODALITIES)
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    return modalities


def main() -> None:
    parser = argparse.ArgumentParser(description="Run light sklearn probes for derived multimodal tasks.")
    parser.add_argument("--samples", default=str(ARTIFACT_ROOT / "index" / "samples.csv"))
    parser.add_argument("--transitions", default=str(ARTIFACT_ROOT / "episodes" / "transitions_h1.csv"))
    parser.add_argument("--out-dir", default=str(ARTIFACT_ROOT / "probes"))
    parser.add_argument("--modalities", default="all", help="all or comma-separated modalities.")
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    samples_path = Path(args.samples)
    require_existing(samples_path, "Sample index not found")
    sample_rows = load_csv(samples_path)
    modalities = parse_modalities(args.modalities)
    out_dir = Path(args.out_dir)

    jobs: list[tuple[str, list[dict[str, str]], str]] = [
        ("current_intent", sample_rows, "intent"),
        ("current_joint", sample_rows, "joint_label"),
        ("current_scene", sample_rows, "scene"),
    ]
    transitions_path = Path(args.transitions)
    if transitions_path.exists():
        jobs.append(("next_intent", load_csv(transitions_path), "next_intent"))

    summary_rows = []
    for name, rows, target in jobs:
        print(f"[probe] {name} target={target} modalities={'+'.join(modalities)}")
        result = run_probe(rows, target, modalities, args.max_iter)
        result_path = out_dir / f"{name}_{'_'.join(modalities)}.json"
        write_json(result_path, result)
        summary_rows.append(
            {
                "name": name,
                "target": target,
                "modalities": "+".join(modalities),
                "train_samples": result["train_samples"],
                "test_samples": result["test_samples"],
                "feature_dim": result["feature_dim"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
            }
        )
        print(f"  accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}")

    write_csv(out_dir / "summary.csv", summary_rows)
    print(f"[saved] {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
