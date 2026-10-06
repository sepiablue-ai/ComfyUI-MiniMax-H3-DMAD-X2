@echo off
setlocal
if "%~1"=="" goto usage
if not exist "%~1" (
  echo ComfyUI Python was not found: "%~1"
  exit /b 2
)
set "DMAD_CONFIG=%~2"
if not defined DMAD_CONFIG set "DMAD_CONFIG=%~dp0config.local.json"
"%~1" -B -X utf8 "%~dp0scripts\setup.py" --config "%DMAD_CONFIG%" %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%
:usage
echo Usage: setup.bat "ComfyUI Python executable" "config.local.json" [--check]
echo Copy config.example.json to config.local.json and set the model paths.
echo Normal setup installs dependencies and X2, prepares inputs, and reuses a verified
echo fused model or creates it on the selected CUDA GPU. It does not start ComfyUI.
exit /b 2
