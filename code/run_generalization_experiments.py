from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from project_paths import PROJECT_ROOT


DATE_GROUPS = {
    "date_20260131": "interaction_20260131_",
    "date_20260227": "interaction_20260227_",
    "date_20260301": "interaction_20260301_",
    "date_20260306": "interaction_20260306_",
}


def all_video_names() -> list[str]:
    return list(load_baseline_literal("VIDEO_LABELS"))


def default_test_video_names() -> list[str]:
    return list(load_baseline_literal("DEFAULT_TEST_VIDEO_NAMES"))


def load_baseline_literal(name: str):
    source_path = PROJECT_ROOT / "code" / "baseline_real_scene.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"Cannot find literal assignment: {name}")


def video_names_for_prefix(prefix: str) -> list[str]:
    return [video_name for video_name in all_video_names() if video_name.startswith(prefix)]


def command_to_string(command: Iterable[str]) -> str:
    return " ".join(command)


def run_experiment(
    *,
    name: str,
    model: str,
    seed: int,
    epochs: int,
    patience: int,
    test_video_names: list[str],
    execute: bool,
    skip_existing: bool,
) -> None:
    output_dir = PROJECT_ROOT / "outputs" / "generalization" / name
    metrics_path = output_dir / "metrics.json"
    if skip_existing and metrics_path.exists():
        print(f"[skip-existing] {name}")
        return

    env = os.environ.copy()
    env["SMART_AR_RANDOM_SEED"] = str(seed)
    env["SMART_AR_TEST_VIDEO_NAMES"] = ",".join(test_video_names)
    env["SMART_AR_MODEL_OUTPUT_DIR"] = str(output_dir)

    command = [
        sys.executable,
        "code/train.py",
        "--model",
        model,
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--skip-feature-check",
    ]

    print(f"[experiment] {name}")
    print(f"  seed={seed}")
    print(f"  test_videos={len(test_video_names)}")
    print(f"  output={output_dir}")
    print(f"  command={command_to_string(command)}")
    if execute:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generalization checks before missing/noise experiments.")
    parser.add_argument("--model", choices=("baseline", "improved"), default="improved")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="*", default=[7, 42, 123])
    parser.add_argument("--seed-only", action="store_true", help="Only run repeated seeds on the default test split.")
    parser.add_argument("--date-only", action="store_true", help="Only run date-holdout tests.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.seed_only and args.date_only:
        raise SystemExit("--seed-only and --date-only cannot be used together.")

    experiments: list[tuple[str, int, list[str]]] = []
    if not args.date_only:
        default_test = default_test_video_names()
        for seed in args.seeds:
            experiments.append((f"{args.model}_seed{seed}_default_test", seed, default_test))

    if not args.seed_only:
        for group_name, prefix in DATE_GROUPS.items():
            test_videos = video_names_for_prefix(prefix)
            if not test_videos:
                print(f"[warn] empty date group: {group_name}")
                continue
            experiments.append((f"{args.model}_{group_name}_test_seed42", 42, test_videos))

    for name, seed, test_videos in experiments:
        run_experiment(
            name=name,
            model=args.model,
            seed=seed,
            epochs=args.epochs,
            patience=args.patience,
            test_video_names=test_videos,
            execute=args.execute,
            skip_existing=args.skip_existing,
        )

    if not args.execute:
        print("[dry-run] add --execute to run these experiments.")


if __name__ == "__main__":
    main()
