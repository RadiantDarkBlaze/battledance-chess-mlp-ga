@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if %errorlevel%==0 (
  python battledance_net_server.py
) else (
  py -3 battledance_net_server.py
)
if errorlevel 1 (
  echo.
  echo Server stopped with an error. Make sure Python and numpy are installed,
  echo and that battledance_training.py is in this same folder.
  pause
)
