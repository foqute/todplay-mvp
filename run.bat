@echo off
chcp 65001 >nul

REM === 이 run.bat 파일이 있는 폴더(프로젝트 루트)로 이동
cd /d "%~dp0"

REM === 백엔드 실행
start "TOD Backend" cmd /k "cd backend && python -m uvicorn main:app --reload --port 8000"

REM === 프론트 실행 (frontend 폴더에서!)
start "TOD Frontend" cmd /k "cd frontend && python -m http.server 5500 --bind 127.0.0.1"

REM === 브라우저 열기 (반드시 5500)
timeout /t 2 >nul
start "" "http://127.0.0.1:5500/index.html"
