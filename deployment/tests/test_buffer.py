import pandas as pd

from industrial_inference.buffer import RollingProcessBuffer


def test_buffer_requires_monotonic_timestamps():
    buffer = RollingProcessBuffer(10, 60, 5)
    buffer.append({"Date": pd.Timestamp("2026-01-01T00:00:00Z"), "x": 1})
    try:
        buffer.append({"Date": pd.Timestamp("2026-01-01T00:00:00Z"), "x": 2})
    except ValueError as error:
        assert "non-monotonic" in str(error)
    else:
        raise AssertionError("duplicate timestamp should be rejected")


def test_large_gap_resets_history():
    buffer = RollingProcessBuffer(10, 60, 5)
    buffer.append({"Date": pd.Timestamp("2026-01-01T00:00:00Z"), "x": 1})
    reset = buffer.append({"Date": pd.Timestamp("2026-01-01T00:02:00Z"), "x": 2})
    assert reset is True
    assert len(buffer) == 1
    assert buffer.frame().iloc[0]["x"] == 2
    assert buffer.reset_count == 1
    assert "timestamp gap" in buffer.last_reset_reason
