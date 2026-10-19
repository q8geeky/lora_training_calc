@echo off
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel%==0 (
  py -3 lora_calc.py
) else (
  python lora_calc.py
)
if errorlevel 1 pause
