# 00 — Architecture overview

## Intended data path

```text
Sensors / laboratory result
        ↓
PLC (automate) and/or DCS controller
        ↓ native industrial protocol
Kepware / KEPServerEX on an approved Windows host
        ↓ OPC UA, authenticated and encrypted
Dockerized Python inference service
        ↓ persistence and advisory output only
SQLite audit database / JSON mirror / dedicated advisory tags / HMI
```

## Responsibility boundary

| Component | Responsible for | Not delegated to the ML service |
|---|---|---|
| Sensor | Measuring the process | Prediction or control |
| PLC/DCS | Deterministic control and safety | Model availability |
| Kepware | Protocol translation and OPC UA exposure | Model decisions |
| Docker service | Validation, buffering and inference | Interlocks or actuator commands |
| SQLite | Local prediction/run/event audit trail | Plant historian or control database |
| HMI/operator | Interpreting advisory output | Automatic intervention |

## Prototype topology

Stage 1 replaces Kepware and the live plant with a CSV replay OPC UA server:

```text
tsp_1min.csv → OPC UA simulator → inference container → SQLite + JSON outputs
```

The simulator and inference service are separate containers on one private
Docker network. This deliberately tests the same client/server boundary that a
future Kepware connection will use.

Detailed views are in `09_macro_integration_architecture.md`,
`10_micro_inference_architecture.md` and `11_sqlite_output_database.md`.

## Model modes

### `target_free`

Uses process measurements only to estimate the current free-acid value. It can
operate when the uploaded/live period has no free-acid measurement.

### `target_anchored`

Uses current and historical free acid plus process features to estimate a future
change, then reconstructs the future level:

```text
predicted_future = current_free_acid + predicted_change
```

Because free acid is a manual laboratory measurement, this mode must receive a
reliable measurement timestamp. Carrying an old laboratory result forward does
not turn it into a current measurement. The prototype therefore defaults to
`target_free` until laboratory timestamp semantics are confirmed.
