@echo off
setlocal EnableExtensions

rem A .bat file necessarily starts through cmd.exe. Relaunch the worker minimized
rem so the operator does not keep a command window behind the native dashboard.
if /I not "%~1"=="--worker" (
    start "" /min "%ComSpec%" /d /c ""%~f0" --worker"
    exit /b 0
)

pushd "%~dp0"

if not exist "runtime" mkdir "runtime"
set "LOG_FILE=runtime\launcher.log"
set "DASHBOARD_EXE=dashboard\tsp_dashboard.exe"

where docker >nul 2>&1
if errorlevel 1 goto :docker_missing

docker info >nul 2>&1
if errorlevel 1 call :start_docker_desktop
if errorlevel 1 goto :docker_unavailable

echo [%date% %time%] Starting Docker services>>"%LOG_FILE%"
if exist "images\tsp-docker-images.tar" (
    docker load -i "images\tsp-docker-images.tar" >>"%LOG_FILE%" 2>&1
)
docker compose up -d --no-build >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    docker compose up -d --build >>"%LOG_FILE%" 2>&1
    if errorlevel 1 goto :compose_failed
)

if not exist "%DASHBOARD_EXE%" goto :dashboard_missing

echo [%date% %time%] Opening dashboard>>"%LOG_FILE%"
start "" /wait "%DASHBOARD_EXE%"
set "DASHBOARD_EXIT=%errorlevel%"
goto :cleanup

:start_docker_desktop
set "DOCKER_DESKTOP=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
if not exist "%DOCKER_DESKTOP%" exit /b 1
start "" "%DOCKER_DESKTOP%"
for /L %%I in (1,1,60) do (
    docker info >nul 2>&1 && exit /b 0
    timeout /t 2 /nobreak >nul
)
exit /b 1

:cleanup
echo [%date% %time%] Dashboard closed; stopping Docker services>>"%LOG_FILE%"
docker compose down >>"%LOG_FILE%" 2>&1
popd
exit /b %DASHBOARD_EXIT%

:docker_missing
call :show_error "Docker was not found. Install and start Docker Desktop, then try again."
goto :failed

:docker_unavailable
call :show_error "Docker Desktop did not become ready within two minutes. See runtime\launcher.log."
goto :failed

:compose_failed
call :show_error "The Docker services could not start. See runtime\launcher.log."
goto :failed

:dashboard_missing
docker compose down >>"%LOG_FILE%" 2>&1
call :show_error "dashboard\tsp_dashboard.exe is missing. Build the dashboard before packaging."
goto :failed

:show_error
set "TSP_LAUNCH_ERROR=%~1"
powershell -NoProfile -WindowStyle Hidden -Command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show($env:TSP_LAUNCH_ERROR, 'TSP Predictor', 'OK', 'Error')" >nul 2>&1
set "TSP_LAUNCH_ERROR="
exit /b 0

:failed
popd
exit /b 1
