TSP PREDICTOR ADVISORY PACKAGE
==============================

Prepared by: Ikram Benfellah

Prerequisites
-------------
- Windows 10/11 64-bit
- Docker Desktop installed
- Hardware virtualization enabled

Start
-----
Double-click launch.bat. The first start can take several minutes if Docker
must build/download the images. The native dashboard opens when services are
ready enough to start; it displays STARTING/WARMING_UP until prediction begins.

Stop
----
Close the dashboard window. The launcher then runs docker compose down.

Outputs
-------
- runtime/predictions.sqlite3 : authoritative audit database
- runtime/predictions.jsonl   : readable prediction mirror
- runtime/latest_prediction.json
- runtime/health.json
- runtime/launcher.log        : launcher/build diagnostics

Safety
------
This package is an advisory laboratory demonstration. Advisory OPC writes are
disabled. It does not change PLC/DCS setpoints, actuators, interlocks or safety
logic.
