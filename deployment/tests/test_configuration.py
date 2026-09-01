from pathlib import Path

import yaml


def test_every_research_variable_has_a_tag_mapping():
    root = Path(__file__).resolve().parents[2]
    mapping = yaml.safe_load((root / "deployment/config/opc_tags.yaml").read_text(encoding="utf-8"))
    expected = {
        "PRODUCTION_TSP_BALANCE", "FLOW_RATE_PHOSPHORIC_ACID_1",
        "FLOW_RATE_PHOSPHORIC_ACID_2", "FLOW_RATE_GROUND_PHOSPHATE",
        "FLOW_RATE_SLURRY_T_PER_HR", "FLOW_RATE_SLURRY_M3_HR", "RECYCLAGE",
        "PRESSURE_VAPEUR", "TEMPERATURE_VAPEUR", "TEMPERATURE_CUVE_ATTAQUE",
        "TEMPERATURE_CUVE_DE_PASSAGE", "TEMPERATURE_SLURRY_PULVERISATEUR",
        "DEPRESSURE_SECHEUR", "LEVEL_CUVE_PASSAGE",
        "TEMPERATURE_GAZ_SORTIE_SECHEUR", "FLOW_RATE_FIOUL",
        "TEMPERATURE_AIR_CHAUD", "FLOW_RATE_LIQUIDE_LAVAGE", "FLOW_RATE_VAPEUR",
        "FLOW_RATE_SLURRY_M3_HR_2", "TEMPERATURE_BRIQUE", "SLURRY_FREE_ACID",
    }
    assert expected <= set(mapping["inputs"])


def test_plant_writes_are_disabled_by_default():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "deployment/config/service.yaml").read_text(encoding="utf-8"))
    assert config["outputs"]["advisory_write_enabled"] is False
