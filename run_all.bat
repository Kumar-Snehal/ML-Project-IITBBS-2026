@echo off
rem ==========================================================================
rem  Runs every pipeline step in order (Windows equivalent of run_all.sh).
rem
rem  Usage, from any folder:      run_all.bat
rem
rem  Python interpreter used (first match wins):
rem    1. the PYTHON environment variable, if set
rem         set PYTHON=C:\path\to\python.exe
rem         run_all.bat
rem    2. .venv\Scripts\python.exe in the project root, if it exists
rem    3. "python" from your PATH
rem
rem  The script stops at the first step that fails.
rem ==========================================================================
setlocal

rem Always work from the project root (the folder containing this file),
rem because the steps are run as modules:  python -m src....
cd /d "%~dp0"

if not defined PYTHON (
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON=.venv\Scripts\python.exe"
    ) else (
        set "PYTHON=python"
    )
)

echo Using interpreter: %PYTHON%

for %%M in (
    src.cleaning.clean_data
    src.features.build_features
    src.features.check_leakage
    src.features.split_data
    src.features.analyse_features
    src.models.naive_bayes
    src.models.logistic_regression
    src.models.evaluate
) do (
    echo.
    echo ================================================================
    echo  %%M
    echo ================================================================
    "%PYTHON%" -m %%M
    if errorlevel 1 (
        echo.
        echo FAILED in step %%M. Stopping.
        exit /b 1
    )
)

echo.
echo All steps completed successfully.
endlocal