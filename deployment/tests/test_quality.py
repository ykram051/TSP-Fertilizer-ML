import pandas as pd

from industrial_inference.quality import DataQualityShield


def configuration():
    return {
        "quality": {
            "reject_bad_or_uncertain": True,
            "reject_impossible_values": True,
            "maximum_missing_required_share": 0.5,
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


def test_isolated_missing_value_is_accepted_for_train_fitted_imputation():
    tags = {
        "inputs": {
            "FLOW_A": {"required": True},
            "FLOW_B": {"required": True},
        }
    }
    shield = DataQualityShield(configuration(), tags)
    decision = shield.evaluate(
        {"FLOW_A": 10, "FLOW_B": float("nan")},
        {"FLOW_A": True, "FLOW_B": True},
    )
    assert decision.accepted
    assert decision.status == "IMPUTED"
    assert decision.warnings == ("imputed_inputs:FLOW_B",)


def test_excessive_simultaneous_missingness_is_rejected():
    config = configuration()
    config["quality"]["maximum_missing_required_share"] = 0.25
    tags = {
        "inputs": {
            "FLOW_A": {"required": True},
            "FLOW_B": {"required": True},
        }
    }
    shield = DataQualityShield(config, tags)
    decision = shield.evaluate(
        {"FLOW_A": float("nan"), "FLOW_B": 20},
        {"FLOW_A": True, "FLOW_B": True},
    )
    assert not decision.accepted
    assert "excessive_missing_inputs" in decision.reasons[0]


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
