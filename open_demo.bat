@echo off
rem Double-click to open the consultation web page (scripts/webui/server.py).
rem Closing this window stops the server.
cd /d "%~dp0"

set PY=
where py >nul 2>nul && set PY=py
if not defined PY (
    where python >nul 2>nul && set PY=python
)
if not defined PY (
    echo Python 3.10+ was not found. Install it from https://www.python.org/downloads/
    pause
    exit /b 1
)

%PY% -c "import fastapi, uvicorn" >nul 2>nul
if errorlevel 1 (
    echo Installing the web interface dependencies, once...
    %PY% -m pip install -e ".[webui]"
    if errorlevel 1 (
        echo Installation failed; see the messages above.
        pause
        exit /b 1
    )
)

echo Starting dxagent on http://127.0.0.1:8420 -- close this window to stop it.
start "" /b cmd /c "ping -n 4 127.0.0.1 >nul & start http://127.0.0.1:8420"
%PY% scripts\webui\server.py
pause
