from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timezone
from typing import Sequence


def nearest_point_index(x_values: Sequence[float], query_x: float) -> int | None:
    """Return the nearest index in an ascending timestamp sequence."""
    if not x_values:
        return None
    position = bisect_left(x_values, query_x)
    if position <= 0:
        return 0
    if position >= len(x_values):
        return len(x_values) - 1
    before = position - 1
    return position if abs(x_values[position] - query_x) < abs(query_x - x_values[before]) else before


def latest_delta(y_values: Sequence[float]) -> float | None:
    """Return latest minus previous prediction when two valid values exist."""
    if len(y_values) < 2:
        return None
    return float(y_values[-1]) - float(y_values[-2])


def seconds_since(timestamp: str | None, now: datetime | None = None) -> float | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    return max(0.0, (reference - parsed.astimezone(timezone.utc)).total_seconds())
