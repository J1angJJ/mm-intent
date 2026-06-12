from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT


DEFAULT_MODELS = {
    "teacher_improved": MODEL_OUTPUT_ROOT / "improved_real_scene_anchor2_perceiver_io_mptasks",
    "ours_margin": MODEL_OUTPUT_ROOT / "ours_margin_mptasks",
}

TASKS = {
    "full": [],
    "no_text": ["text"],
    "no_scene": ["scene"],
}


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


def parse_model_specs(values: list[str]) -> dict[str, Path]:
    if not values:
        return DEFAULT_MODELS
    specs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("Model specs must look like name=outputs/path")
        name, path = value.split("=", 1)
        specs[name.strip()] = Path(path).resolve()
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diagnostic analysis for trained checkpoints.")
    parser.add_argument("--model", action="append", default=[], help="Add model as name=output_dir.")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), choices=tuple(TASKS))
    parser.add_argument("--analysis-root", default=str(MODEL_OUTPUT_ROOT / "diagnostics"))
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only prints them.")
    args = parser.parse_args()

    models = parse_model_specs(args.model)
    analysis_root = Path(args.analysis_root).resolve()
    jobs: list[tuple[str, list[str], Path]] = []
    for model_name, output_dir in models.items():
        for task_name in args.tasks:
            analysis_name = f"{model_name}_{task_name}"
            command = [
                sys.executable,
                "code/analyze_trained_model.py",
                "--output-dir",
                str(output_dir),
                "--analysis-dir",
                str(analysis_root / analysis_name),
                "--name",
                analysis_name,
            ]
            missing = TASKS[task_name]
            if missing:
                command.extend(["--missing-modalities", *missing])
            log_path = PROJECT_ROOT / f"logs_diagnostic_{analysis_name}.txt"
            jobs.append((analysis_name, command, log_path))

    print(f"[diagnostic-suite] jobs={len(jobs)} execute={args.execute}")
    for index, (name, command, log_path) in enumerate(jobs, start=1):
        print(f"[{index:02d}/{len(jobs):02d}] {name}")
        print(" ".join(command))
        print(f"log: {log_path}")
        if args.execute:
            tee_process(command, log_path)


if __name__ == "__main__":
    main()
