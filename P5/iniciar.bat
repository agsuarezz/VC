@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py %*
) else (
  echo No se encuentra el entorno de Python. Sigue la instalacion del README.md.
  pause
  exit /b 1
)
if errorlevel 1 (
  pause
  exit /b 1
)
