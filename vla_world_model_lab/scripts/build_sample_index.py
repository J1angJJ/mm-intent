from __future__ import annotations

import argparse
from typing import Any

from common import (
    ARTIFACT_ROOT,
    INTENT_NAMES,
    MODALITIES,
    SCENE_BY_VIDEO,
    VIDEO_LABELS,
    feature_paths,
    infer_user,
    safe_len_npy,
    split_for_video,
    stem,
    write_csv,
    write_json,
)


def min_available_count(lengths: dict[str, int | None]) -> int:
    available = [value for value in lengths.values() if value is not None]
    return min(available) if available else 0


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []

    for video_name, intent_id in VIDEO_LABELS.items():
        paths = feature_paths(video_name)
        lengths = {name: safe_len_npy(path) for name, path in paths.items() if name in MODALITIES or name == "timestamp"}
        sample_count = min_available_count(lengths)
        scene = SCENE_BY_VIDEO[video_name]
        split = split_for_video(video_name)
        user = infer_user(video_name)
        intent = INTENT_NAMES[int(intent_id)]

        video_rows.append(
            {
                "video_id": stem(video_name),
                "video_name": video_name,
                "split": split,
                "user": user,
                "scene": scene,
                "intent_id": intent_id,
                "intent": intent,
                "sample_count": sample_count,
                **{f"{key}_count": value if value is not None else "" for key, value in lengths.items()},
                **{f"has_{key}": int(path.exists()) for key, path in paths.items()},
            }
        )

        for segment_index in range(sample_count):
            sample_rows.append(
                {
                    "sample_id": f"{stem(video_name)}::{segment_index:04d}",
                    "video_id": stem(video_name),
                    "video_name": video_name,
                    "segment_index": segment_index,
                    "split": split,
                    "user": user,
                    "scene": scene,
                    "intent_id": intent_id,
                    "intent": intent,
                    "joint_label": f"{scene}_{intent}",
                }
            )

    return sample_rows, video_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sample/video indexes for derived VLA/world-model tasks.")
    parser.add_argument("--out-dir", default=str(ARTIFACT_ROOT / "index"))
    args = parser.parse_args()

    out_dir = ARTIFACT_ROOT / "index" if args.out_dir is None else __import__("pathlib").Path(args.out_dir)
    sample_rows, video_rows = build_rows()
    write_csv(out_dir / "samples.csv", sample_rows)
    write_csv(out_dir / "videos.csv", video_rows)
    write_json(
        out_dir / "summary.json",
        {
            "num_videos": len(video_rows),
            "num_samples": len(sample_rows),
            "train_samples": sum(1 for row in sample_rows if row["split"] == "train"),
            "test_samples": sum(1 for row in sample_rows if row["split"] == "test"),
        },
    )
    print(f"[saved] {out_dir / 'samples.csv'}")
    print(f"[saved] {out_dir / 'videos.csv'}")
    print(f"[samples] {len(sample_rows)}")


if __name__ == "__main__":
    main()
