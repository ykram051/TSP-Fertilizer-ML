from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RollingProcessBuffer:
    maximum_rows: int
    expected_cadence_seconds: int
    tolerance_seconds: int
    rows: list[dict] = field(default_factory=list)
    reset_count: int = 0
    last_reset_reason: str | None = None
    last_gap_seconds: float | None = None

    def append(self, snapshot: dict) -> bool:
        """Append a new source row and return whether history was reset.

        A source-time gap cannot be bridged safely because temporal features need
        contiguous history.  The reset is deliberately observable so the operator
        can distinguish normal warm-up from a data continuity problem.
        """
        timestamp = pd.Timestamp(snapshot["Date"])
        reset = False
        if self.rows:
            previous = pd.Timestamp(self.rows[-1]["Date"])
            if timestamp <= previous:
                raise ValueError(f"non-monotonic timestamp: {timestamp} <= {previous}")
            gap = (timestamp - previous).total_seconds()
            maximum = self.expected_cadence_seconds + self.tolerance_seconds
            if gap > maximum:
                # Rolling features must not silently bridge a data gap.
                self.rows.clear()
                self.reset_count += 1
                self.last_gap_seconds = gap
                self.last_reset_reason = (
                    f"source timestamp gap of {gap:.0f}s exceeds {maximum:.0f}s"
                )
                reset = True
        self.rows.append(snapshot)
        self.rows = self.rows[-self.maximum_rows :]
        return reset

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def ready(self, minimum_rows: int) -> bool:
        return len(self.rows) >= minimum_rows

    def clear(self) -> None:
        self.rows.clear()

    def __len__(self) -> int:
        return len(self.rows)
