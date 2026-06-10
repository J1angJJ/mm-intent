from __future__ import annotations

import argparse
import subprocess
import sys

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT


MODALITIES = ("imu", "gesture", "audio", "text", "scene")
NOISE_LEVELS = (0.2, 0.4, 0.6)


def build_command(args: argparse.Namespace, modality: str, level: float) -> list[str]:
    percent = int(round(level * 100))
    output_dir = MODEL_OUTPUT_ROOT / "noise_experiments" / args.model / f"{modality}_noise_{percent}"
    command = [
        sys.executable,
        "code/train.py",
        "--model",
        args.model,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--noise-modality",
        modality,
        "--noise-level",
        str(level),
    ]
    if args.skip_test_eval:
        command.append("--skip-test-eval")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-modality noise experiments.")
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only prints them.")
    args = parser.parse_args()

    jobs = [(modality, level) for modality in MODALITIES for level in NOISE_LEVELS]
    print(f"[noise] model={args.model} experiments={len(jobs)} execute={args.execute}")
    for index, (modality, level) in enumerate(jobs, start=1):
        command = build_command(args, modality, level)
        print(f"[{index:02d}/{len(jobs):02d}]", " ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
