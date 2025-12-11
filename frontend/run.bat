@echo off
setlocal
rem 이 파일이 있는 폴더(frontend)에서 바로 서빙
python -m http.server 5500
endlocal
