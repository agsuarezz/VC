@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app_arte.py %*
) else if exist "%USERPROFILE%\anaconda3\envs\VC_P5\python.exe" (
  "%USERPROFILE%\anaconda3\envs\VC_P5\python.exe" app_arte.py %*
) else if exist "%USERPROFILE%\anaconda3\envs\VC_P1\python.exe" (
  "%USERPROFILE%\anaconda3\envs\VC_P1\python.exe" app_arte.py %*
) else (
  echo No se encuentra el entorno de Python (.venv, VC_P5 o VC_P1). Sigue la instalacion del README.md.
  pause
  exit /b 1
)
if errorlevel 1 (
  pause
  exit /b 1
)
