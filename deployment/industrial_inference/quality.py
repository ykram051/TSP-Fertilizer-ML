from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityDecision:
    accepted: bool
    status: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class DataQualityShield:
    def __init__(self, service_config: dict[str, Any], tag_config: dict[str, Any]):
        self.config = service_config
        self.tags = tag_config

    def evaluate(self, snapshot: dict[str, Any], qualities: dict[str, bool]) -> QualityDecision:
        reasons: list[str] = []
        required = [name for name, meta in self.tags["inputs"].items() if meta.get("required")]

        missing = [name for name in required if name not in snapshot or pd.isna(snapshot[name])]
        warnings: list[str] = []
        missing_share = len(missing) / max(len(required), 1)
        allowed_share = float(
            self.config["quality"].get("maximum_missing_required_share", 0.0)
        )
        if missing_share > allowed_share:
            reasons.append(
                f"excessive_missing_inputs:{len(missing)}/{len(required)}:"
                + ",".join(missing)
            )
        elif missing:
            warnings.append("imputed_inputs:" + ",".join(missing))

        if self.config["quality"]["reject_bad_or_uncertain"]:
            bad = [name for name in required if not qualities.get(name, False)]
            if bad:
                reasons.append("bad_or_uncertain_quality:" + ",".join(bad))

        if self.config["quality"]["reject_impossible_values"]:
            outside = []
            for name, meta in self.tags["inputs"].items():
                if name not in snapshot or pd.isna(snapshot[name]):
                    continue
                minimum, maximum = meta.get("minimum"), meta.get("maximum")
                value = snapshot[name]
                if isinstance(value, (int, float, np.number)):
                    if minimum is not None and value < minimum:
                        outside.append(name)
                    if maximum is not None and value > maximum:
                        outside.append(name)
            if outside:
                reasons.append("outside_engineering_limits:" + ",".join(sorted(set(outside))))

        if reasons:
            return QualityDecision(False, "DATA_ERROR", tuple(reasons), tuple(warnings))
        status = "IMPUTED" if missing else "GOOD"
        return QualityDecision(True, status, (), tuple(warnings))

    def laboratory_target_is_fresh(self, snapshot: dict[str, Any], source_time: pd.Timestamp) -> QualityDecision:
        policy = self.config["manual_laboratory_target"]
        value = snapshot.get(policy["variable"])
        stamp = snapshot.get(policy["timestamp_node_key"])
        reasons = []
        if value is None or pd.isna(value):
            reasons.append("manual_laboratory_value_missing")
        if stamp is None or pd.isna(stamp):
            reasons.append("manual_laboratory_timestamp_missing")
        else:
            age = source_time - pd.Timestamp(stamp)
            if age.total_seconds() < 0:
                reasons.append("manual_laboratory_timestamp_in_future")
            if age > pd.Timedelta(minutes=policy["maximum_age_minutes"]):
                reasons.append("manual_laboratory_value_stale")
        return QualityDecision(not reasons, "GOOD" if not reasons else "DATA_ERROR", tuple(reasons))
