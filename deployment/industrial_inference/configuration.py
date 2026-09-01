from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


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


def load_configuration() -> tuple[dict[str, Any], dict[str, Any]]:
    service_path = os.getenv("SERVICE_CONFIG", "/config/service.yaml")
    tags_path = os.getenv("TAG_CONFIG", "/config/opc_tags.yaml")
    config, tags = deepcopy(_read_yaml(service_path)), _read_yaml(tags_path)

    config["opc"]["endpoint"] = os.getenv("OPC_ENDPOINT", config["opc"]["endpoint"])
    config["service"]["mode"] = os.getenv("INFERENCE_MODE", config["service"]["mode"])
    config["service"]["forecast_horizon_minutes"] = int(os.getenv(
        "FORECAST_HORIZON_MINUTES", config["service"]["forecast_horizon_minutes"]
    ))
    config["outputs"]["advisory_write_enabled"] = _as_bool(os.getenv(
        "ADVISORY_WRITE_ENABLED", str(config["outputs"]["advisory_write_enabled"])
    ))
    config["simulator"]["replay_interval_seconds"] = float(os.getenv(
        "REPLAY_INTERVAL_SECONDS", config["simulator"]["replay_interval_seconds"]
    ))

    if config["service"]["mode"] not in {"target_free", "target_anchored"}:
        raise ValueError("service.mode must be target_free or target_anchored")
    if config["service"]["forecast_horizon_minutes"] not in {1, 5, 10, 15}:
        raise ValueError("forecast_horizon_minutes must be one of 1, 5, 10, 15")
    return config, tags
