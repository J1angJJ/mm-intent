from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ARTIFACT_ROOT, load_csv, require_existing


def save_bar(counts: dict[str, int], title: str, output_path: Path, rotation: int = 25) -> None:
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, axis = plt.subplots(figsize=(max(7, len(labels) * 0.55), 4.8))
    axis.bar(labels, values, color="#4C78A8", edgecolor="white", linewidth=0.8)
    axis.set_title(title)
    axis.set_ylabel("Count")
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=rotation)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dataset/index distributions.")
    parser.add_argument("--samples", default=str(ARTIFACT_ROOT / "index" / "samples.csv"))
    parser.add_argument("--videos", default=str(ARTIFACT_ROOT / "index" / "videos.csv"))
    parser.add_argument("--out-dir", default=str(ARTIFACT_ROOT / "figures" / "dataset"))
    args = parser.parse_args()

    samples_path = Path(args.samples)
    videos_path = Path(args.videos)
    require_existing(samples_path, "Sample index not found")
    require_existing(videos_path, "Video index not found")
    samples = load_csv(samples_path)
    videos = load_csv(videos_path)
    out_dir = Path(args.out_dir)

    if samples:
        save_bar(dict(Counter(row["split"] for row in samples)), "Samples by Split", out_dir / "samples_by_split.png")
        save_bar(dict(Counter(row["scene"] for row in samples)), "Samples by Scene", out_dir / "samples_by_scene.png")
        save_bar(dict(Counter(row["intent"] for row in samples)), "Samples by Intent", out_dir / "samples_by_intent.png")
        save_bar(dict(Counter(row["joint_label"] for row in samples)), "Samples by Joint Label", out_dir / "samples_by_joint_label.png", rotation=35)
        save_bar(dict(Counter(row["user"] for row in samples)), "Samples by User", out_dir / "samples_by_user.png")
    else:
        print("[warn] samples.csv is empty; drawing video-level figures only.")

    video_counts = {
        row["video_id"]: int(row["sample_count"] or 0)
        for row in videos
    }
    save_bar(video_counts, "Aligned Samples per Video", out_dir / "samples_per_video.png", rotation=65)
    save_bar(dict(Counter(row["split"] for row in videos)), "Videos by Split", out_dir / "videos_by_split.png")
    save_bar(dict(Counter(row["scene"] for row in videos)), "Videos by Scene", out_dir / "videos_by_scene.png")
    save_bar(dict(Counter(row["intent"] for row in videos)), "Videos by Intent", out_dir / "videos_by_intent.png")

    split_intent: dict[str, Counter[str]] = defaultdict(Counter)
    for row in samples:
        split_intent[row["split"]][row["intent"]] += 1
    for split, counts in split_intent.items():
        save_bar(dict(counts), f"{split.title()} Samples by Intent", out_dir / f"{split}_intent_distribution.png")

    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
