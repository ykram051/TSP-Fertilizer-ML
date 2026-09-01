# 05 — Stage 2: read-only shadow mode

## Laboratory Kepware gate

Before production, connect to a laboratory Kepware instance and verify
certificates, user permissions, namespace mapping, disconnect/reconnect and
quality-code behaviour.

## Plant shadow policy

The production OPC account is read-only. Predictions are written to an external
log or historian and compared later with laboratory results. No prediction is
used to change plant control.

## Data-quality shield

The service refuses or delays prediction when it detects:

- missing required tags;
- bad or uncertain OPC quality;
- duplicate or non-monotonic timestamps;
- gaps larger than the configured cadence tolerance;
- stale laboratory target measurement for target-anchored mode;
- values outside configured engineering limits;
- insufficient warm-up history.

Every rejection is logged with a reason. Imputation must never be silent.
