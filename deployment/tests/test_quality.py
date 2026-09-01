import pandas as pd

from industrial_inference.quality import DataQualityShield


def configuration():
    return {
        "quality": {
            "reject_bad_or_uncertain": True,
            "reject_impossible_values": True,
        },
        "manual_laboratory_target": {
            "variable": "SLURRY_FREE_ACID",
            "timestamp_node_key": "SLURRY_FREE_ACID_LAB_TIMESTAMP",
            "maximum_age_minutes": 120,
        },
    }


def tag_configuration():
    return {
        "inputs": {
            "FLOW": {"required": True, "minimum": 0, "maximum": 100},
            "SLURRY_FREE_ACID": {"required": False},
            "SLURRY_FREE_ACID_LAB_TIMESTAMP": {"required": False},
        }
    }


def test_bad_quality_is_rejected():
    shield = DataQualityShield(configuration(), tag_configuration())
    decision = shield.evaluate({"FLOW": 10}, {"FLOW": False})
    assert not decision.accepted
    assert "bad_or_uncertain_quality" in decision.reasons[0]


def test_stale_manual_laboratory_result_is_rejected():
    shield = DataQualityShield(configuration(), tag_configuration())
    snapshot = {
        "SLURRY_FREE_ACID": 11.2,
        "SLURRY_FREE_ACID_LAB_TIMESTAMP": "2026-01-01T00:00:00Z",
    }
    decision = shield.laboratory_target_is_fresh(
        snapshot, pd.Timestamp("2026-01-01T03:00:00Z")
    )
    assert not decision.accepted
    assert "manual_laboratory_value_stale" in decision.reasons
