from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT


EXPERIMENTS = {
    "focal": [
        "--focal-loss-gamma",
        "1.5",
    ],
    "fallback": [
        "--fallback-max-gate",
        "0.35",
        "--fallback-aux-weight",
        "0.25",
    ],
    "fallback_focal": [
        "--fallback-max-gate",
        "0.35",
        "--fallback-aux-weight",
        "0.25",
        "--focal-loss-gamma",
        "1.5",
    ],
    "distill_light": [
        "--missing-distill-weight",
        "0.05",
        "--missing-distill-temperature",
        "2.0",
        "--missing-distill-modalities",
        "scene",
        "audio",
        "imu",
        "--missing-distill-prob",
        "scene",
        "0.10",
        "--missing-distill-prob",
        "audio",
        "0.10",
        "--missing-distill-prob",
        "imu",
        "0.10",
        "--no-missing-distill-force-mask",
    ],
}

TASKS = {
    "main": [],
    "no_text": ["--missing-modalities", "text"],
    "no_scene": ["--missing-modalities", "scene"],
}


def build_command(
    experiment_name: str,
    experiment_args: list[str],
    task_name: str,
    task_args: list[str],
    epochs: int,
    patience: int,
    skip_feature_check: bool,
) -> list[str]:
    output_dir = MODEL_OUTPUT_ROOT / "innovation_suite" / experiment_name / task_name
    command = [
        sys.executable,
        "code/train.py",
        "--model",
        "improved",
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--output-dir",
        str(output_dir),
    ]
    if skip_feature_check:
        command.append("--skip-feature-check")
    command.extend(task_args)
    command.extend(experiment_args)
    return command


def tee_process(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def parse_methods(values: Iterable[str]) -> list[str]:
    methods = list(values)
    if not methods or methods == ["all"]:
        return list(EXPERIMENTS)
    unknown = sorted(set(methods) - set(EXPERIMENTS))
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Available: {sorted(EXPERIMENTS)}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Run innovation candidates with main/no_text/no_scene probes.")
    parser.add_argument("--methods", nargs="*", default=["all"], help=f"Choose from: {', '.join(EXPERIMENTS)}")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--skip-feature-check", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only prints them.")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    jobs: list[tuple[str, str, list[str], Path]] = []
    for experiment_name in methods:
        for task_name, task_args in TASKS.items():
            command = build_command(
                experiment_name,
                EXPERIMENTS[experiment_name],
                task_name,
                task_args,
                args.epochs,
                args.patience,
                args.skip_feature_check,
            )
            log_path = PROJECT_ROOT / f"logs_innovation_{experiment_name}_{task_name}.txt"
            jobs.append((experiment_name, task_name, command, log_path))

    print(f"[innovation-suite] jobs={len(jobs)} execute={args.execute}")
    for index, (experiment_name, task_name, command, log_path) in enumerate(jobs, start=1):
        print(f"[{index:02d}/{len(jobs):02d}] {experiment_name}/{task_name}")
        print(" ".join(command))
        print(f"log: {log_path}")
        if args.execute:
            tee_process(command, log_path)


if __name__ == "__main__":
    main()
