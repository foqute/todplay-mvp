@echo off
setlocal enabledelayedexpansion

REM ==== 경로 설정 ====
cd /d "%~dp0"

REM ==== 가상환경 활성화 ====
call .venv\Scripts\activate.bat

REM ==== 백엔드 실행 (8000포트) ====
start "TOD-Backend" cmd /k uvicorn main:app --reload --host 127.0.0.1 --port 8000

REM ==== 프론트엔드 실행 (5500포트) ====
cd frontend
start "TOD-Frontend" cmd /k python -m http.server 5500

endlocal
