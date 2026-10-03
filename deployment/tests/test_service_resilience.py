from industrial_inference.service import InferenceService


def test_closed_connection_is_escalated_to_reconnect_path():
    assert InferenceService.is_connection_failure(ConnectionError("Connection is closed"))
    assert InferenceService.is_connection_failure(RuntimeError("session closed by server"))


def test_model_error_is_not_mistaken_for_connection_failure():
    assert not InferenceService.is_connection_failure(ValueError("feature alignment failed"))
