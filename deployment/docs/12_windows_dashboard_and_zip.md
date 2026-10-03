# 12 — Native Windows dashboard and release ZIP

## Runtime separation

```text
Docker network (headless)                 Windows host (native)
----------------------------------        -----------------------------
OPC simulator -> inference service  --->  runtime/health.json
                                  \---->  runtime/predictions.sqlite3
                                           |
                                           v
                                  tsp_dashboard.exe (read-only)
```

The GUI is not part of the inference container. Closing or crashing it cannot
interrupt OPC acquisition or model logic. The launcher deliberately stops the
laboratory containers after the user closes the GUI; production shadow mode
would normally use a separately managed long-running service.

## Polling contract

- Health: worker-thread read every 2 seconds.
- SQLite trend: read-only query every 10 seconds, latest 720 records.
- The database connection has a 500 ms busy timeout and is always closed
  deterministically, which avoids Windows file locks.
- Missing, empty or briefly busy files result in a visible message, not a GUI
  freeze or crash.

## Build and package

```powershell
cd deployment
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\build_dashboard.ps1
.\scripts\package_release.ps1
```

The build script accepts 64-bit Python 3.11 or 3.12 and automatically checks
the Python Launcher, PATH, and the local Codex bundled runtime.

For a receiving computer without registry/PyPI access:

```powershell
.\scripts\package_release.ps1 -IncludeDockerImages
```

The optional image archive substantially increases the ZIP size but prevents
the receiving computer from needing to download/build Docker images.

## Final package structure

```text
TSP-Predictor-Advisory/
|-- launch.bat
|-- compose.yaml
|-- Dockerfile.inference
|-- Dockerfile.simulator
|-- PACKAGE_README.txt
|-- .env
|-- dashboard/
|   `-- tsp_dashboard.exe
|-- config/
|-- database/schema.sql
|-- industrial_inference/
|-- simulator/
|-- artifacts/
|   |-- feature_preparation_artifacts/
|   |-- exogenous_virtual_sensor_artifacts/models/
|   |-- target_free_upgrade_artifacts/models/
|   `-- delta_transition_artifacts/models/
|-- data/tsp_1min.csv
|-- runtime/
|-- scripts/
`-- images/tsp-docker-images.tar   (optional)
```

The distribution Compose file uses only paths beneath this root. The package
does not depend on the research notebooks or on the original `reports/` folder.

## One-click lifecycle

1. `launch.bat` relaunches itself as a minimized worker.
2. It checks the Docker CLI and Docker daemon.
3. If necessary, it starts Docker Desktop and waits up to two minutes.
4. If an offline image archive exists, it loads it.
5. It tries `docker compose up -d --no-build`; if images are unavailable it
   falls back to `docker compose up -d --build`.
6. It opens `dashboard/tsp_dashboard.exe` with `start /wait`.
7. When the dashboard closes, it runs `docker compose down`.
8. Diagnostics are retained in `runtime/launcher.log`.

## Practical limitation

A batch file always begins through `cmd.exe`; the implementation immediately
relaunches the worker minimized. A completely invisible launcher would require
an additional `.vbs`/compiled launcher, which would contradict the requested
single `launch.bat` entry point.
