# 04 — Stage 1: CSV replay simulator

## Purpose

The simulator proves connectivity without touching plant systems. It creates
one OPC UA node per configured research variable and replays `tsp_1min.csv`.

## What it validates

- research-variable to Node-ID mapping;
- subscription handling;
- one-minute snapshots;
- warm-up and rolling history;
- missing/bad data policy;
- container restart and reconnect behaviour;
- model artifact loading;
- prediction timestamps and audit logs.

## Accelerated time

One historical minute is replayed every configured number of seconds. The CSV
timestamp remains the source timestamp, so the service can distinguish process
time from wall-clock replay time.

The laboratory default is one historical minute per wall-clock second. This is
60 times faster than the plant cadence while leaving enough time for one full
OPC snapshot, feature calculation and prediction on Docker Desktop.

## Warm-up

The feature pipeline needs 30 historical rows. The service reports
`WARMING_UP` until sufficient consecutive snapshots exist.

## Docker Hub TLS timeout

The first build must download `python:3.11-slim` and Python packages. A message
such as `failed to fetch oauth token` or `TLS handshake timeout` means Docker
could not reach Docker Hub; no container was built or started. Test the network
path separately:

```powershell
docker pull python:3.11-slim
```

If it continues to fail, verify Docker Desktop proxy configuration and ask
IT/OT whether `auth.docker.io`, `registry-1.docker.io` and the Python package
index are permitted. Do not weaken TLS verification.
