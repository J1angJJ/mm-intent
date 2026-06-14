from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from project_paths import MODEL_OUTPUT_ROOT, PROCESSED_DATA_DIR, PROJECT_ROOT


TASKS = {
    "main": [],
    "no_text": ["--missing-modalities", "text"],
    "no_scene": ["--missing-modalities", "scene"],
}


def tee_process(command: list[str], log_path: Path) -> None:
    print("[run]", " ".join(command), flush=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLIP gesture + hand geometry fused-feature probes.")
    parser.add_argument("--base-gesture-dir", default=str(PROCESSED_DATA_DIR / "strong_gesture_features"))
    parser.add_argument("--geometry-dir", default=str(PROCESSED_DATA_DIR / "hand_geometry_features"))
    parser.add_argument("--fused-output-dir", default=str(PROCESSED_DATA_DIR / "gesture_clip_hand_geometry_features"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--skip-geometry-generate", action="store_true")
    parser.add_argument("--skip-fuse", action="store_true")
    parser.add_argument("--skip-feature-check", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only prints them.")
    args = parser.parse_args()

    base_gesture_dir = Path(args.base_gesture_dir).resolve()
    geometry_dir = Path(args.geometry_dir).resolve()
    fused_output_dir = Path(args.fused_output_dir).resolve()

    jobs: list[tuple[str, list[str], Path]] = []
    if not args.skip_geometry_generate:
        jobs.append(
            (
                "generate_hand_geometry",
                [
                    sys.executable,
                    "code/feature_extraction/extract_hand_geometry_features.py",
                    "--output-dir",
                    str(geometry_dir),
                ],
                PROJECT_ROOT / "logs_feature_clip_hand_geometry_generate.txt",
            )
        )

    if not args.skip_fuse:
        jobs.append(
            (
                "fuse_clip_and_hand_geometry",
                [
                    sys.executable,
                    "code/feature_extraction/concat_gesture_features.py",
                    "--base-dir",
                    str(base_gesture_dir),
                    "--geometry-dir",
                    str(geometry_dir),
                    "--output-dir",
                    str(fused_output_dir),
                    "--overwrite",
                ],
                PROJECT_ROOT / "logs_feature_clip_hand_geometry_fuse.txt",
            )
        )

    for task_name, task_args in TASKS.items():
        output_dir = MODEL_OUTPUT_ROOT / "feature_suite" / "clip_hand_geometry" / task_name
        command = [
            sys.executable,
            "code/train.py",
            "--model",
            "improved",
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--output-dir",
            str(output_dir),
            "--gesture-feature-dir",
            str(fused_output_dir),
            "--gesture-feature-dim",
            "864",
        ]
        if args.skip_feature_check:
            command.append("--skip-feature-check")
        command.extend(task_args)
        jobs.append(
            (
                f"clip_hand_geometry_{task_name}",
                command,
                PROJECT_ROOT / f"logs_feature_clip_hand_geometry_{task_name}.txt",
            )
        )

    print(f"[gesture-fusion-suite] jobs={len(jobs)} execute={args.execute}")
    for index, (name, command, log_path) in enumerate(jobs, start=1):
        print(f"[{index:02d}/{len(jobs):02d}] {name}")
        print(" ".join(command))
        print(f"log: {log_path}")
        if args.execute:
            tee_process(command, log_path)


if __name__ == "__main__":
    main()
