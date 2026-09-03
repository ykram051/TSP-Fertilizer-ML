# 06 — Stage 3: advisory HMI

After successful external shadow validation, outputs may be exposed through a
dedicated namespace such as `TSP.ML_Advisory`.

Recommended fields:

| Output | Meaning |
|---|---|
| `PredictedFreeAcid` | Model estimate |
| `PredictedChange10Min` | Target-anchored change, when valid |
| `GeneratedAt` | Time inference ran |
| `PredictionFor` | Future timestamp represented by the forecast |
| `ModelStatus` | `WARMING_UP`, `RUNNING`, `DATA_ERROR`, `DISCONNECTED` |
| `InputQuality` | Aggregate quality decision |
| `ModelVersion` | Exact deployed artifact/image version |

The HMI must visibly label the output **Advisory**. No service permission may
modify setpoints, valve/pump commands, modes, permissives, interlocks or safety
logic.
