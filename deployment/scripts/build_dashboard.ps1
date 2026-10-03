$ErrorActionPreference = "Stop"
$DeploymentRoot = Split-Path -Parent $PSScriptRoot
Set-Location $DeploymentRoot

$VirtualEnvironment = Join-Path $DeploymentRoot ".dashboard-build"
$Python = Join-Path $VirtualEnvironment "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    $CodexPython = Join-Path $env:USERPROFILE `
        ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

    if ($PythonLauncher) {
        Write-Host "Creating dashboard environment with Python Launcher..."
        & $PythonLauncher.Source -3.11 -m venv $VirtualEnvironment
    }
    elseif ($PythonCommand) {
        Write-Host "Creating dashboard environment with $($PythonCommand.Source)..."
        & $PythonCommand.Source -m venv $VirtualEnvironment
    }
    elseif (Test-Path $CodexPython) {
        Write-Host "Creating dashboard environment with the bundled Python runtime..."
        & $CodexPython -m venv $VirtualEnvironment
    }
    else {
        throw @"
No usable 64-bit Python installation was found.
Install Python 3.11 or 3.12 from https://www.python.org/downloads/windows/
and select 'Add python.exe to PATH', then rerun this script.
"@
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Python)) {
        throw "The isolated dashboard Python environment could not be created."
    }
}

$RuntimeCheck = & $Python -c `
    "import platform,struct,sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{struct.calcsize(`"P`")*8}|{platform.machine()}')"
if ($LASTEXITCODE -ne 0) { throw "The dashboard Python environment is not runnable." }
$RuntimeParts = $RuntimeCheck.Trim().Split("|")
if ($RuntimeParts[1] -ne "64") {
    throw "A 64-bit Python runtime is required; detected $RuntimeCheck."
}
Write-Host "Dashboard build runtime: Python $($RuntimeParts[0]), $($RuntimeParts[1])-bit"

$PyInstaller = Join-Path $VirtualEnvironment "Scripts\pyinstaller.exe"

& $Python -m pip install --timeout 300 --retries 10 --prefer-binary `
    -r "dashboard\requirements-dashboard.txt"
if ($LASTEXITCODE -ne 0) { throw "Dashboard dependencies could not be installed." }

& $PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --noconsole `
    --name tsp_dashboard `
    --distpath dashboard `
    --workpath dashboard\build `
    --specpath dashboard `
    --paths dashboard `
    --hidden-import sqlite3 `
    --hidden-import PySide6.QtCore `
    --hidden-import PySide6.QtGui `
    --hidden-import PySide6.QtWidgets `
    --exclude-module PyQt5 `
    --exclude-module PyQt6 `
    --exclude-module PySide2 `
    dashboard\dashboard.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

Write-Host "Dashboard executable created: dashboard\tsp_dashboard.exe"
