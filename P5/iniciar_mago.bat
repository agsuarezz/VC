@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%USERPROFILE%\anaconda3\envs\VC_P5\python.exe"
if not exist "%PYTHON%" set "PYTHON=%USERPROFILE%\anaconda3\envs\VC_P1\python.exe"
if not exist "%PYTHON%" (
  echo No se encuentra un entorno de Python. Sigue la instalacion del README.md.
  pause
  exit /b 1
)
"%PYTHON%" app_mago.py %*
if errorlevel 1 (
  pause
  exit /b 1
)
