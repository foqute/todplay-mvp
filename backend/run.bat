@echo off
setlocal
cd /d "%~dp0"
call .venv\Scripts\activate.bat
uvicorn main:app --reload --host 127.0.0.1 --port 8000
endlocal
