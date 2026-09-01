from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RollingProcessBuffer:
    maximum_rows: int
    expected_cadence_seconds: int
    tolerance_seconds: int
    rows: list[dict] = field(default_factory=list)

    def append(self, snapshot: dict) -> None:
        timestamp = pd.Timestamp(snapshot["Date"])
        if self.rows:
            previous = pd.Timestamp(self.rows[-1]["Date"])
            if timestamp <= previous:
                raise ValueError(f"non-monotonic timestamp: {timestamp} <= {previous}")
            gap = (timestamp - previous).total_seconds()
            maximum = self.expected_cadence_seconds + self.tolerance_seconds
            if gap > maximum:
                # Rolling features must not silently bridge a data gap.
                self.rows.clear()
        self.rows.append(snapshot)
        self.rows = self.rows[-self.maximum_rows :]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def ready(self, minimum_rows: int) -> bool:
        return len(self.rows) >= minimum_rows

    def clear(self) -> None:
        self.rows.clear()

    def __len__(self) -> int:
        return len(self.rows)
