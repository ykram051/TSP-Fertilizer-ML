# Native Windows dashboard

The dashboard is a read-only desktop client. It does not connect to OPC UA and
does not call the model directly. Every two seconds it reads `runtime/health.json`;
every ten seconds it opens `runtime/predictions.sqlite3` in SQLite read-only
mode and requests the latest 720 rows.

The GUI uses a worker from `QThreadPool`, so a temporarily locked, missing or
empty database cannot freeze the interface. The green line shows predictions;
amber points show predictions produced with one or more imputed inputs.

## Chart interaction

- The dashboard displays only the active `run_id`. Restarting the inference
  service starts a clean chart timeline while older runs remain safely stored
  in SQLite for audit and export.
- Hover over the chart to snap a crosshair to the nearest prediction. The
  tooltip shows the UTC process timestamp, prediction to three decimals, and
  whether measured or imputed inputs were used.
- Left-click a point to pin its tooltip. Select **Clear pin** to return to hover.
- Use **60 points**, **360 points**, or **720 points** in the chart header to
  change the displayed time range. The database is not queried on mouse move.
- The prediction KPI shows the change from the immediately previous prediction.
  Blue means an increase and amber a decrease; neither color claims that the
  process movement is beneficial or harmful.
- The status LED breathes while the service is running. A heartbeat older than
  the selected dashboard watchdog threshold is labeled `STALE`; warning/error
  states also change card borders. Choose 30, 90, or 180 seconds in the
  **HEARTBEAT** selector, or disable only the dashboard alert with **Off**.
  The inference service still writes its five-second health heartbeat either way.

## Warm-up and restart behaviour

The model needs 31 contiguous one-minute source rows because selected lags and
rolling features require historical context. A real source-time gap causes a
visible, safety-preserving history reset rather than an invented feature value.
The status card reports the reset reason and count.

For the CSV laboratory simulator only, replay flow control waits for the
inference service acknowledgement before sending the next historical row. This
prevents OPC notification coalescing from turning a fast replay into artificial
source-time gaps. It must remain disabled for a live Kepware/plant connection.

## Build

On a Windows machine with 64-bit Python 3.11 or 3.12. The script detects, in
order, `py.exe`, `python.exe`, or the bundled Codex Python runtime:

```powershell
cd deployment
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\build_dashboard.ps1
```

The executable is created as `dashboard/tsp_dashboard.exe`.

Equivalent PyInstaller command:

```powershell
pyinstaller --noconfirm --clean --onefile --noconsole `
  --name tsp_dashboard `
  --distpath dashboard `
  --workpath dashboard/build `
  --specpath dashboard `
  --paths dashboard `
  --hidden-import sqlite3 `
  --hidden-import PySide6.QtCore `
  --hidden-import PySide6.QtGui `
  --hidden-import PySide6.QtWidgets `
  --exclude-module PyQt5 `
  --exclude-module PyQt6 `
  --exclude-module PySide2 `
  dashboard/dashboard.py
```

`sqlite3` is included with Python; the explicit hidden import is defensive.
Only the three Qt modules used by this application are collected. Collecting
all PySide6/PyQtGraph submodules would unnecessarily include WebEngine, 3D,
Bluetooth, multimedia and example applications.
