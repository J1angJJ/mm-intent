from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from project_paths import MODEL_OUTPUT_ROOT, PROCESSED_DATA_DIR, PROJECT_ROOT


DEFAULT_SPECS = (
    "seq5_win500",
    "seq10_win500",
    "seq10_win750",
    "seq10_win1000",
    "seq20_win750",
)


def parse_spec(raw_spec: str) -> tuple[str, int, int]:
    parts = raw_spec.split("_")
    if len(parts) != 2 or not parts[0].startswith("seq") or not parts[1].startswith("win"):
        raise argparse.ArgumentTypeError(
            f"Invalid spec '{raw_spec}'. Expected format like seq10_win750."
        )
    try:
        seq_len = int(parts[0][3:])
        half_window_ms = int(parts[1][3:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid spec '{raw_spec}'. Expected numeric seq/window values."
        ) from exc
    if seq_len <= 0 or half_window_ms <= 0:
        raise argparse.ArgumentTypeError("seq_len and half_window_ms must be positive.")
    return raw_spec, seq_len, half_window_ms


def tee_process(command: list[str], log_path: Path) -> None:
    print("[run]", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def read_metrics(output_dir: Path) -> dict[str, object] | None:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hand Geometry temporal window ablation.")
    parser.add_argument("--specs", nargs="*", default=list(DEFAULT_SPECS), help="Specs like seq10_win750.")
    parser.add_argument("--feature-root", default=str(PROCESSED_DATA_DIR / "hand_geometry_window_ablation"))
    parser.add_argument("--output-root", default=str(MODEL_OUTPUT_ROOT / "hand_geometry_window_ablation"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--expected-count", type=int, default=39)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only prints them.")
    args = parser.parse_args()

    specs = [parse_spec(raw_spec) for raw_spec in args.specs]
    feature_root = Path(args.feature_root).resolve()
    output_root = Path(args.output_root).resolve()
    log_root = PROJECT_ROOT / "logs_hand_geometry_window_ablation"

    jobs: list[tuple[str, list[str], Path]] = []
    output_dirs: dict[str, Path] = {}
    for name, seq_len, half_window_ms in specs:
        feature_dir = feature_root / name
        output_dir = output_root / name
        output_dirs[name] = output_dir
        feature_marker_count = len(list(feature_dir.glob("strong_gesture_features_*.npy")))
        metrics_path = output_dir / "metrics.json"

        if not args.skip_generate and not (args.skip_existing and feature_marker_count >= args.expected_count):
            jobs.append(
                (
                    f"extract_{name}",
                    [
                        sys.executable,
                        "code/feature_extraction/extract_hand_geometry_features.py",
                        "--output-dir",
                        str(feature_dir),
                        "--seq-len",
                        str(seq_len),
                        "--half-window-ms",
                        str(half_window_ms),
                    ],
                    log_root / f"extract_{name}.txt",
                )
            )
        elif args.skip_existing and feature_marker_count >= args.expected_count:
            print(f"[skip-existing] features {name}: {feature_marker_count}/{args.expected_count}")

        if not args.skip_training and not (args.skip_existing and metrics_path.exists()):
            jobs.append(
                (
                    f"train_{name}",
                    [
                        sys.executable,
                        "code/train.py",
                        "--model",
                        "improved",
                        "--epochs",
                        str(args.epochs),
                        "--patience",
                        str(args.patience),
                        "--expected-count",
                        str(args.expected_count),
                        "--output-dir",
                        str(output_dir),
                        "--gesture-feature-dir",
                        str(feature_dir),
                        "--gesture-feature-dim",
                        "96",
                        "--target-timesteps",
                        str(seq_len),
                    ],
                    log_root / f"train_{name}.txt",
                )
            )
        elif args.skip_existing and metrics_path.exists():
            print(f"[skip-existing] metrics {name}: {metrics_path}")

    print(f"[hand-geometry-window-ablation] specs={len(specs)} jobs={len(jobs)} execute={args.execute}")
    for index, (job_name, command, log_path) in enumerate(jobs, start=1):
        print(f"[{index:02d}/{len(jobs):02d}] {job_name}")
        print(" ".join(command))
        print(f"log: {log_path}")
        if args.execute:
            tee_process(command, log_path)

    if args.execute:
        rows: list[dict[str, object]] = []
        for name, seq_len, half_window_ms in specs:
            metrics = read_metrics(output_dirs[name])
            if metrics is None:
                continue
            final = metrics.get("final_metrics", {})
            best = metrics.get("best_checkpoint", {})
            runtime = metrics.get("runtime", {})
            rows.append(
                {
                    "spec": name,
                    "seq_len": seq_len,
                    "half_window_ms": half_window_ms,
                    "window_ms": half_window_ms * 2,
                    "best_epoch": best.get("epoch"),
                    "val_acc": best.get("val_acc"),
                    "test_joint_acc": final.get("test_joint_acc"),
                    "test_intent_acc": final.get("test_intent_acc"),
                    "test_scene_acc": final.get("test_scene_acc"),
                    "train_avg_seconds_per_sample": runtime.get("train_avg_seconds_per_sample"),
                    "test_avg_seconds_per_sample": runtime.get("test_avg_seconds_per_sample"),
                }
            )
        write_summary(rows, output_root / "window_ablation_summary.csv")
    else:
        print("[dry-run] add --execute to run the full suite.")


if __name__ == "__main__":
    main()
