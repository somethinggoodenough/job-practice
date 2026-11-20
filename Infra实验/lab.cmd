@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run: python scripts/setup.py
    popd
    exit /b 1
)
".venv\Scripts\python.exe" -m kernellab %*
set "LAB_EXIT=%ERRORLEVEL%"
popd
exit /b %LAB_EXIT%
