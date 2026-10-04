from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from industrial_inference.configuration import load_configuration, validate_configuration

ROOT = Path(__file__).resolve().parents[1]


def _repository_files():
    config = yaml.safe_load((ROOT / "config" / "service.yaml").read_text(encoding="utf-8"))
    tags = yaml.safe_load((ROOT / "config" / "opc_tags.yaml").read_text(encoding="utf-8"))
    return config, tags


def _write(tmp_path, config, tags, monkeypatch):
    service, tag_file = tmp_path / "service.yaml", tmp_path / "opc_tags.yaml"
    service.write_text(yaml.safe_dump(config), encoding="utf-8")
    tag_file.write_text(yaml.safe_dump(tags), encoding="utf-8")
    monkeypatch.setenv("SERVICE_CONFIG", str(service))
    monkeypatch.setenv("TAG_CONFIG", str(tag_file))


def test_repository_configuration_is_valid():
    config, tags = _repository_files()
    validate_configuration(config, tags)


def test_load_configuration_accepts_repository_files(monkeypatch):
    monkeypatch.setenv("SERVICE_CONFIG", str(ROOT / "config" / "service.yaml"))
    monkeypatch.setenv("TAG_CONFIG", str(ROOT / "config" / "opc_tags.yaml"))
    for name in ("INFERENCE_MODE", "FORECAST_HORIZON_MINUTES", "ADVISORY_WRITE_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    config, _ = load_configuration()
    assert config["outputs"]["advisory_write_enabled"] is False


def test_missing_entry_is_named_in_the_error(tmp_path, monkeypatch):
    config, tags = _repository_files()
    del config["service"]["maximum_buffer_rows"]
    _write(tmp_path, config, tags, monkeypatch)
    with pytest.raises(ValueError, match="service.maximum_buffer_rows"):
        load_configuration()


@pytest.mark.parametrize("path,value,message", [
    (("service", "mode"), "autopilot", "target_free or target_anchored"),
    (("service", "forecast_horizon_minutes"), 7, "one of 1, 5, 10, 15"),
    (("service", "cadence_seconds"), 0, "cadence_seconds"),
    (("quality", "maximum_missing_required_share"), 1.5, "between 0 and 1"),
    (("manual_laboratory_target", "maximum_age_minutes"), 0, "maximum_age_minutes"),
])
def test_invalid_values_are_rejected(path, value, message):
    config, tags = _repository_files()
    config = deepcopy(config)
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError, match=message):
        validate_configuration(config, tags)


def test_input_without_node_id_is_rejected():
    config, tags = _repository_files()
    tags = deepcopy(tags)
    first = next(iter(tags["inputs"]))
    tags["inputs"][first] = {"required": True}
    with pytest.raises(ValueError, match=first):
        validate_configuration(config, tags)


def test_non_numeric_horizon_override_has_a_clear_message(tmp_path, monkeypatch):
    config, tags = _repository_files()
    _write(tmp_path, config, tags, monkeypatch)
    monkeypatch.setenv("FORECAST_HORIZON_MINUTES", "ten")
    with pytest.raises(ValueError, match="must be an integer"):
        load_configuration()
