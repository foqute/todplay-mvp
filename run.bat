@echo off
setlocal enabledelayedexpansion

REM ▲ 루트에 둔다고 가정
cd /d "%~dp0"

set "BACK=%cd%\backend"
set "FRONT=%cd%\frontend"
set "VENV=%BACK%\.venv\Scripts\activate.bat"

REM ---- 백엔드 창 ----
start "TOD-Backend" cmd /k "cd /d "%BACK%" && call "%VENV%" && uvicorn main:app --reload --host 127.0.0.1 --port 8000"

REM ---- 프론트엔드 창 ----
start "TOD-Frontend" cmd /k "python -m http.server 5500 --directory "%FRONT%""

REM ---- 브라우저 열기 (선택) ----
start "" http://127.0.0.1:5500/index.html

endlocal
