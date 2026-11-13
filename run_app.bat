@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3 and try again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the app environment...
    py -3 -m venv .venv
    if errorlevel 1 goto :error
)

echo Checking app dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo Starting the app...
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo The app could not be started. Check the message above.
pause
exit /b 1
