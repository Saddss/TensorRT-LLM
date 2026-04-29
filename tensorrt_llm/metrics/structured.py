# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight structured rolling-window metrics for TRT-LLM serving.

The hot path only appends small integer samples. Window aggregation is done
lazily when the HTTP endpoint is queried.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Iterable, Optional

DEFAULT_WINDOWS_SECONDS = (10.0, 60.0, 300.0)
DEFAULT_TOKEN_BUCKETS = (0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
DEFAULT_MAX_SAMPLES = 200_000


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_csv_numbers(raw: Optional[str], default: Iterable[float]) -> list[float]:
    if raw is None or raw == "":
        return [float(v) for v in default]
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values or [float(v) for v in default]


class _LengthAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0
        self.minimum: Optional[int] = None
        self.maximum: Optional[int] = None
        self.values: list[int] = []

    def add(self, value: int) -> None:
        value = _safe_int(value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.values.append(value)

    def as_dict(self, buckets: list[int]) -> dict[str, Any]:
        bucket_counts = [0] * (len(buckets) + 1)
        for value in self.values:
            placed = False
            for idx, upper in enumerate(buckets):
                if value <= upper:
                    bucket_counts[idx] += 1
                    placed = True
                    break
            if not placed:
                bucket_counts[-1] += 1

        bucket_rows = []
        lower: Optional[int] = None
        for idx, upper in enumerate(buckets):
            bucket_rows.append({"gt": lower, "le": upper, "count": bucket_counts[idx]})
            lower = upper
        bucket_rows.append({"gt": lower, "le": "+Inf", "count": bucket_counts[-1]})

        return {
            "count": self.count,
            "sum": self.total,
            "min": self.minimum,
            "max": self.maximum,
            "avg": (self.total / self.count) if self.count else None,
            "buckets": bucket_rows,
        }


class _LengthSummary:
    def __init__(self) -> None:
        self.prompt = _LengthAccumulator()
        self.generation = _LengthAccumulator()
        self.total = _LengthAccumulator()

    def add(self, prompt_tokens: int, generation_tokens: int) -> None:
        total_tokens = prompt_tokens + generation_tokens
        self.prompt.add(prompt_tokens)
        self.generation.add(generation_tokens)
        self.total.add(total_tokens)

    def as_dict(self, buckets: list[int]) -> dict[str, Any]:
        return {
            "requests": self.total.count,
            "prompt_tokens": self.prompt.as_dict(buckets),
            "generation_tokens": self.generation.as_dict(buckets),
            "total_tokens": self.total.as_dict(buckets),
        }


class _KvAccumulator:
    FIELDS = (
        "reused_blocks", "full_reused_blocks", "partial_reused_blocks",
        "missed_blocks", "gen_alloc_blocks", "onboard_blocks",
        "onboard_bytes", "offload_blocks", "offload_bytes",
        "intra_device_copy_blocks", "intra_device_copy_bytes",
    )

    def __init__(self) -> None:
        self.data = {field: 0 for field in self.FIELDS}
        self.secondary_max_blocks = 0
        self.secondary_used_blocks = 0
        self.samples = 0

    def add(self, sample: dict[str, int]) -> None:
        self.samples += 1
        for field in self.FIELDS:
            self.data[field] += _safe_int(sample.get(field, 0))
        self.secondary_max_blocks += _safe_int(sample.get("secondary_max_blocks", 0))
        self.secondary_used_blocks += _safe_int(sample.get("secondary_used_blocks", 0))

    def merge(self, other: "_KvAccumulator") -> None:
        self.samples += other.samples
        for field in self.FIELDS:
            self.data[field] += other.data[field]
        self.secondary_max_blocks += other.secondary_max_blocks
        self.secondary_used_blocks += other.secondary_used_blocks

    def as_dict(self) -> dict[str, Any]:
        reused = self.data["reused_blocks"]
        missed = self.data["missed_blocks"]
        total = reused + missed
        result = dict(self.data)
        result.update({
            "samples": self.samples,
            "lookup_blocks": total,
            "hit_rate": (reused / total) if total else None,
            "secondary_max_blocks": self.secondary_max_blocks,
            "secondary_used_blocks": self.secondary_used_blocks,
            "secondary_utilization": (
                self.secondary_used_blocks / self.secondary_max_blocks
                if self.secondary_max_blocks else None
            ),
        })
        return result


class StructuredMetricsStore:
    """In-process structured metrics with global and rolling-window views."""

    def __init__(self, max_samples: Optional[int] = None) -> None:
        configured = os.getenv("TRTLLM_STRUCTURED_METRICS_MAX_SAMPLES")
        self.max_samples = max_samples or _safe_int(configured, DEFAULT_MAX_SAMPLES)
        self._lock = threading.Lock()
        self._request_samples = deque(maxlen=self.max_samples)
        self._kv_samples = deque(maxlen=self.max_samples)
        self._request_global = _LengthSummary()
        self._kv_global = defaultdict(_KvAccumulator)

    def record_request(self, prompt_tokens: int, generation_tokens: int) -> None:
        prompt_tokens = _safe_int(prompt_tokens)
        generation_tokens = _safe_int(generation_tokens)
        now = time.time()
        with self._lock:
            self._request_samples.append((now, prompt_tokens, generation_tokens))
            self._request_global.add(prompt_tokens, generation_tokens)

    def record_kv_iteration(self, kv_iter_stats: dict[str, Any]) -> None:
        if not kv_iter_stats:
            return
        now = time.time()
        samples: list[tuple[float, str, dict[str, int]]] = []
        for window_size, stats in kv_iter_stats.items():
            if not isinstance(stats, dict):
                continue
            sample = {
                "reused_blocks": _safe_int(stats.get("iterReusedBlocks", 0)),
                "full_reused_blocks": _safe_int(stats.get("iterFullReusedBlocks", 0)),
                "partial_reused_blocks": _safe_int(stats.get("iterPartialReusedBlocks", 0)),
                "missed_blocks": _safe_int(stats.get("iterMissedBlocks", 0)),
                "gen_alloc_blocks": _safe_int(stats.get("iterGenAllocBlocks", 0)),
                "onboard_blocks": _safe_int(stats.get("iterOnboardBlocks", 0)),
                "onboard_bytes": _safe_int(stats.get("iterOnboardBytes", 0)),
                "offload_blocks": _safe_int(stats.get("iterOffloadBlocks", 0)),
                "offload_bytes": _safe_int(stats.get("iterOffloadBytes", 0)),
                "intra_device_copy_blocks": _safe_int(stats.get("iterIntraDeviceCopyBlocks", 0)),
                "intra_device_copy_bytes": _safe_int(stats.get("iterIntraDeviceCopyBytes", 0)),
                "secondary_max_blocks": _safe_int(stats.get("secondaryMaxNumBlocks", 0)),
                "secondary_used_blocks": _safe_int(stats.get("secondaryUsedNumBlocks", 0)),
            }
            samples.append((now, str(window_size), sample))

        if not samples:
            return
        with self._lock:
            for item in samples:
                _, window_size, sample = item
                self._kv_samples.append(item)
                self._kv_global[window_size].add(sample)

    def snapshot(self,
                 windows_seconds: Optional[Iterable[float]] = None,
                 token_buckets: Optional[Iterable[int]] = None) -> dict[str, Any]:
        windows = [float(w) for w in (windows_seconds or DEFAULT_WINDOWS_SECONDS)]
        buckets = sorted(int(b) for b in (token_buckets or DEFAULT_TOKEN_BUCKETS))
        now = time.time()
        with self._lock:
            request_samples = list(self._request_samples)
            kv_samples = list(self._kv_samples)
            request_global = self._request_global
            kv_global_items = list(self._kv_global.items())

        response = {
            "timestamp": now,
            "windows_seconds": windows,
            "request_length_buckets": buckets,
            "kv_cache": {
                "global": self._kv_snapshot_from_items(kv_global_items),
                "windows": {},
            },
            "request_lengths": {
                "global": request_global.as_dict(buckets),
                "windows": {},
            },
        }

        for window in windows:
            cutoff = now - window
            label = self._window_label(window)
            kv_acc = defaultdict(_KvAccumulator)
            for ts, window_size, sample in kv_samples:
                if ts >= cutoff:
                    kv_acc[window_size].add(sample)
            length_summary = _LengthSummary()
            for ts, prompt_tokens, generation_tokens in request_samples:
                if ts >= cutoff:
                    length_summary.add(prompt_tokens, generation_tokens)
            response["kv_cache"]["windows"][label] = self._kv_snapshot_from_items(kv_acc.items())
            response["request_lengths"]["windows"][label] = length_summary.as_dict(buckets)
        return response

    @staticmethod
    def _window_label(window: float) -> str:
        return str(int(window)) if window.is_integer() else str(window)

    @staticmethod
    def _kv_snapshot_from_items(items) -> dict[str, Any]:
        total = _KvAccumulator()
        by_window_size = {}
        for key, acc in items:
            by_window_size[str(key)] = acc.as_dict()
            total.merge(acc)
        return {"total": total.as_dict(), "by_kv_window_size": by_window_size}
