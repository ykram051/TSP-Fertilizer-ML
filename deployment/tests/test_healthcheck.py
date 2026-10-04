import json
from datetime import datetime, timedelta, timezone

import pytest

from industrial_inference.healthcheck import main


def _write(path, status="RUNNING", age_seconds=1, naive=False):
    stamp = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    stamp = stamp.replace(tzinfo=None) if naive else stamp
    path.write_text(json.dumps({"status": status, "updated_at": stamp.isoformat()}), encoding="utf-8")


@pytest.fixture
def health(tmp_path, monkeypatch):
    path = tmp_path / "health.json"
    monkeypatch.setenv("HEALTH_FILE", str(path))
    monkeypatch.delenv("HEALTH_MAX_AGE_SECONDS", raising=False)
    return path


def test_fresh_running_service_is_healthy(health):
    _write(health)
    assert main() == 0


def test_warming_up_is_healthy(health):
    _write(health, status="WARMING_UP")
    assert main() == 0


@pytest.mark.parametrize("status", ["DISCONNECTED", "DATA_ERROR"])
def test_failed_states_are_unhealthy(health, status):
    _write(health, status=status)
    assert main() == 1


def test_stale_file_is_unhealthy_and_limit_is_configurable(health, monkeypatch):
    _write(health, age_seconds=200)
    assert main() == 1
    monkeypatch.setenv("HEALTH_MAX_AGE_SECONDS", "600")
    assert main() == 0


def test_missing_or_malformed_file_is_unhealthy(health):
    assert main() == 1
    health.write_text("{not json", encoding="utf-8")
    assert main() == 1
    health.write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
    assert main() == 1


def test_naive_timestamp_is_treated_as_utc(health):
    _write(health, naive=True)
    assert main() == 0
