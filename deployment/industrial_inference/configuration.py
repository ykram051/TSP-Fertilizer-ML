from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SERVICE_KEYS = (
    "service.name", "service.mode", "service.forecast_horizon_minutes",
    "service.cadence_seconds", "service.cadence_tolerance_seconds",
    "service.maximum_buffer_rows", "service.heartbeat_interval_seconds",
    "opc.endpoint", "opc.subscription_period_ms", "opc.reconnect_delay_seconds",
    "quality.reject_bad_or_uncertain", "quality.reject_impossible_values",
    "manual_laboratory_target.variable", "manual_laboratory_target.timestamp_node_key",
    "manual_laboratory_target.maximum_age_minutes",
    "outputs.advisory_write_enabled", "outputs.sqlite_database", "outputs.sqlite_schema",
    "outputs.jsonl_predictions", "outputs.health_file",
    "simulator.replay_interval_seconds",
)
REQUIRED_TAG_KEYS = ("inputs", "system.sequence.node_id", "system.source_timestamp.node_id")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _lookup(mapping: dict[str, Any], dotted: str, source: str) -> Any:
    value: Any = mapping
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing required entry '{dotted}' in {source}")
        value = value[key]
    return value


def _number(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def validate_configuration(config: dict[str, Any], tags: dict[str, Any]) -> None:
    """Fail at start-up with a precise message instead of failing mid-stream."""
    for key in REQUIRED_SERVICE_KEYS:
        _lookup(config, key, "service.yaml")
    for key in REQUIRED_TAG_KEYS:
        _lookup(tags, key, "opc_tags.yaml")
    if not isinstance(tags["inputs"], dict) or not tags["inputs"]:
        raise ValueError("opc_tags.yaml: 'inputs' must be a non-empty mapping")
    for name, meta in tags["inputs"].items():
        if not isinstance(meta, dict) or not meta.get("node_id"):
            raise ValueError(f"opc_tags.yaml: input '{name}' has no node_id")

    service = config["service"]
    if service["mode"] not in {"target_free", "target_anchored"}:
        raise ValueError("service.mode must be target_free or target_anchored")
    if service["forecast_horizon_minutes"] not in {1, 5, 10, 15}:
        raise ValueError("forecast_horizon_minutes must be one of 1, 5, 10, 15")
    for key, minimum in (("cadence_seconds", 1), ("maximum_buffer_rows", 1),
                         ("heartbeat_interval_seconds", 1)):
        if _number(service[key], f"service.{key}") < minimum:
            raise ValueError(f"service.{key} must be >= {minimum}")
    if _number(service["cadence_tolerance_seconds"], "service.cadence_tolerance_seconds") < 0:
        raise ValueError("service.cadence_tolerance_seconds must be >= 0")
    if _number(config["opc"]["subscription_period_ms"], "opc.subscription_period_ms") <= 0:
        raise ValueError("opc.subscription_period_ms must be > 0")
    if _number(config["opc"]["reconnect_delay_seconds"], "opc.reconnect_delay_seconds") < 0:
        raise ValueError("opc.reconnect_delay_seconds must be >= 0")
    share = _number(config["quality"].get("maximum_missing_required_share", 0.0),
                    "quality.maximum_missing_required_share")
    if not 0.0 <= share <= 1.0:
        raise ValueError("quality.maximum_missing_required_share must be between 0 and 1")
    if _number(config["manual_laboratory_target"]["maximum_age_minutes"],
               "manual_laboratory_target.maximum_age_minutes") <= 0:
        raise ValueError("manual_laboratory_target.maximum_age_minutes must be > 0")


def load_configuration() -> tuple[dict[str, Any], dict[str, Any]]:
    service_path = os.getenv("SERVICE_CONFIG", "/config/service.yaml")
    tags_path = os.getenv("TAG_CONFIG", "/config/opc_tags.yaml")
    config, tags = deepcopy(_read_yaml(service_path)), _read_yaml(tags_path)
    validate_configuration(config, tags)  # structure first, so overrides below cannot KeyError

    config["opc"]["endpoint"] = os.getenv("OPC_ENDPOINT", config["opc"]["endpoint"])
    config["service"]["mode"] = os.getenv("INFERENCE_MODE", config["service"]["mode"])
    try:
        config["service"]["forecast_horizon_minutes"] = int(os.getenv(
            "FORECAST_HORIZON_MINUTES", config["service"]["forecast_horizon_minutes"]
        ))
    except ValueError:
        raise ValueError("FORECAST_HORIZON_MINUTES must be an integer (1, 5, 10 or 15)") from None
    config["outputs"]["advisory_write_enabled"] = _as_bool(os.getenv(
        "ADVISORY_WRITE_ENABLED", str(config["outputs"]["advisory_write_enabled"])
    ))
    config["simulator"]["replay_interval_seconds"] = _number(os.getenv(
        "REPLAY_INTERVAL_SECONDS", config["simulator"]["replay_interval_seconds"]
    ), "REPLAY_INTERVAL_SECONDS")

    validate_configuration(config, tags)  # values again, now including overrides
    return config, tags