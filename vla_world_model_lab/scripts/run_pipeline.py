from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import ARTIFACT_ROOT, LAB_ROOT


def run(command: list[str], keep_going: bool) -> bool:
    print("[run]", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=LAB_ROOT.parent)
    if result.returncode != 0:
        if keep_going:
            print(f"[warn] command failed with code {result.returncode}")
            return False
        raise SystemExit(result.returncode)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local VLA/world-model preparation pipeline.")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--probe-modalities", default="all")
    args = parser.parse_args()

    py = sys.executable
    commands = [
        [py, "vla_world_model_lab/scripts/build_sample_index.py"],
        [py, "vla_world_model_lab/scripts/build_episode_transitions.py"],
        [py, "vla_world_model_lab/scripts/visualize_dataset.py"],
    ]
    if not args.skip_probes:
        commands.extend(
            [
                [py, "vla_world_model_lab/scripts/run_probe_baselines.py", "--modalities", args.probe_modalities],
                [py, "vla_world_model_lab/scripts/visualize_results.py", "--summary", str(ARTIFACT_ROOT / "probes" / "summary.csv")],
            ]
        )

    for command in commands:
        run(command, keep_going=args.keep_going)

    print(f"[done] artifacts: {ARTIFACT_ROOT}")


if __name__ == "__main__":
    main()
