from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import ARTIFACT_ROOT, load_csv, require_existing, write_csv, write_json


def build_transitions(samples: list[dict[str, str]], horizon: int) -> list[dict[str, Any]]:
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        by_video[row["video_id"]].append(row)

    transitions: list[dict[str, Any]] = []
    for video_id, rows in sorted(by_video.items()):
        rows = sorted(rows, key=lambda item: int(item["segment_index"]))
        for index, current in enumerate(rows):
            next_index = index + horizon
            if next_index >= len(rows):
                continue
            future = rows[next_index]
            transitions.append(
                {
                    "transition_id": f"{video_id}::{int(current['segment_index']):04d}->{int(future['segment_index']):04d}",
                    "video_id": video_id,
                    "video_name": current["video_name"],
                    "split": current["split"],
                    "user": current["user"],
                    "scene": current["scene"],
                    "segment_index": current["segment_index"],
                    "next_segment_index": future["segment_index"],
                    "intent": current["intent"],
                    "intent_id": current["intent_id"],
                    "joint_label": current["joint_label"],
                    "next_intent": future["intent"],
                    "next_intent_id": future["intent_id"],
                    "next_joint_label": future["joint_label"],
                }
            )
    return transitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build episode transition index for next-intent/world-model tasks.")
    parser.add_argument("--samples", default=str(ARTIFACT_ROOT / "index" / "samples.csv"))
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--out-dir", default=str(ARTIFACT_ROOT / "episodes"))
    args = parser.parse_args()

    samples_path = Path(args.samples)
    require_existing(samples_path, "Sample index not found. Run build_sample_index.py first")
    samples = load_csv(samples_path)
    transitions = build_transitions(samples, args.horizon)

    out_dir = Path(args.out_dir)
    out_path = out_dir / f"transitions_h{args.horizon}.csv"
    write_csv(out_path, transitions)
    write_json(
        out_dir / f"summary_h{args.horizon}.json",
        {
            "horizon": args.horizon,
            "num_transitions": len(transitions),
            "train_transitions": sum(1 for row in transitions if row["split"] == "train"),
            "test_transitions": sum(1 for row in transitions if row["split"] == "test"),
        },
    )
    print(f"[saved] {out_path}")
    print(f"[transitions] {len(transitions)}")


if __name__ == "__main__":
    main()
