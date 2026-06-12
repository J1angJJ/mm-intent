from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from project_paths import MODEL_OUTPUT_ROOT, PROJECT_ROOT


MODALITIES = ("imu", "gesture", "audio", "text", "scene")


def experiment_groups(max_missing: int) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []
    for size in range(1, max_missing + 1):
        groups.extend(tuple(group) for group in itertools.combinations(MODALITIES, size))
    return groups


def build_command(args: argparse.Namespace, group: Sequence[str]) -> list[str]:
    name = "no_" + "_".join(group)
    output_dir = MODEL_OUTPUT_ROOT / "missing_experiments" / args.output_model_name / name
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
        "--missing-modalities",
        *group,
    ]
    if args.skip_test_eval:
        command.append("--skip-test-eval")
    if args.consistency_weight is not None:
        command.extend(["--consistency-weight", str(args.consistency_weight)])
    if args.consistency_mask_prob is not None:
        command.extend(["--consistency-mask-prob", str(args.consistency_mask_prob)])
    if args.consistency_noise_std is not None:
        command.extend(["--consistency-noise-std", str(args.consistency_noise_std)])
    if args.consistency_temperature is not None:
        command.extend(["--consistency-temperature", str(args.consistency_temperature)])
    if args.consistency_modalities:
        command.extend(["--consistency-modalities", *args.consistency_modalities])
    if args.margin_loss_weight is not None:
        command.extend(["--margin-loss-weight", str(args.margin_loss_weight)])
    if args.margin_value is not None:
        command.extend(["--margin-value", str(args.margin_value)])
    if args.margin_intent_confusion_weight is not None:
        command.extend(["--margin-intent-confusion-weight", str(args.margin_intent_confusion_weight)])
    if args.margin_scene_confusion_weight is not None:
        command.extend(["--margin-scene-confusion-weight", str(args.margin_scene_confusion_weight)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modality missing experiments.")
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument("--output-model-name", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-missing", type=int, choices=(1, 2), default=2)
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--consistency-weight", type=float)
    parser.add_argument("--consistency-mask-prob", type=float)
    parser.add_argument("--consistency-noise-std", type=float)
    parser.add_argument("--consistency-temperature", type=float)
    parser.add_argument("--consistency-modalities", nargs="*", default=[])
    parser.add_argument("--margin-loss-weight", type=float)
    parser.add_argument("--margin-value", type=float)
    parser.add_argument("--margin-intent-confusion-weight", type=float)
    parser.add_argument("--margin-scene-confusion-weight", type=float)
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default only prints them.")
    args = parser.parse_args()
    if args.output_model_name is None:
        args.output_model_name = args.model

    groups = experiment_groups(args.max_missing)
    print(
        f"[missing] model={args.model} output_model_name={args.output_model_name} "
        f"experiments={len(groups)} execute={args.execute}"
    )
    for index, group in enumerate(groups, start=1):
        command = build_command(args, group)
        print(f"[{index:02d}/{len(groups):02d}]", " ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
