@echo off
rem Start here on a new machine. Checks that uv is installed (the only thing you install by hand), then runs
rem `inkling doctor`, which checks everything else and lists the commands that install what is missing.
where uv >nul 2>nul
if errorlevel 1 (
  echo uv is not installed. Install it with:
  echo.
  echo     winget install --id astral-sh.uv -e
  echo.
  echo Then open a NEW terminal and run setup.cmd again.
  exit /b 1
)
uv run --script "%~dp0scripts\inkling.py" doctor %*
