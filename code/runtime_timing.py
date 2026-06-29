from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


def sync_device() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed_block() -> Iterator[dict[str, float]]:
    sync_device()
    start = time.perf_counter()
    payload: dict[str, float] = {}
    try:
        yield payload
    finally:
        sync_device()
        payload["seconds"] = time.perf_counter() - start


def per_sample(seconds: float, samples: int) -> float:
    return float(seconds) / max(int(samples), 1)


def timing_payload(train_seconds: float, train_samples: int, test_seconds: float | None, test_samples: int) -> dict[str, float | int | None]:
    return {
        "train_total_seconds": float(train_seconds),
        "train_samples_seen": int(train_samples),
        "train_avg_seconds_per_sample": per_sample(train_seconds, train_samples),
        "test_total_seconds": None if test_seconds is None else float(test_seconds),
        "test_samples_seen": int(test_samples),
        "test_avg_seconds_per_sample": None if test_seconds is None else per_sample(test_seconds, test_samples),
    }
