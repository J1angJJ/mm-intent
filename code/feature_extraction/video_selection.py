from __future__ import annotations

import os
from collections.abc import Iterable


def parse_requested_video_names(default_video_names: Iterable[str]) -> list[str]:
    """Return the requested MP4 names, preserving the default order."""
    defaults = list(default_video_names)
    raw_value = os.getenv("MM_INTENT_VIDEO_NAMES", "").strip()
    if not raw_value:
        return defaults

    requested = [
        item.strip()
        for item in raw_value.replace(";", ",").split(",")
        if item.strip()
    ]
    default_set = set(defaults)
    unknown = sorted(set(requested) - default_set)
    if unknown:
        raise SystemExit(f"Unknown MM_INTENT_VIDEO_NAMES entries: {unknown}")

    requested_set = set(requested)
    return [video_name for video_name in defaults if video_name in requested_set]


def filter_avi_mp4_items(items: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    pairs = list(items)
    selected_mp4 = parse_requested_video_names(mp4_name for _avi_name, mp4_name in pairs)
    selected_set = set(selected_mp4)
    return [
        (avi_name, mp4_name)
        for avi_name, mp4_name in pairs
        if mp4_name in selected_set
    ]
