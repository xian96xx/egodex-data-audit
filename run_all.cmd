@echo off
setlocal
rem Run the whole EgoDex data audit from the repo root.
rem If "python" is not the interpreter with numpy/h5py/pandas/scipy/matplotlib/torch, set it first:
rem   set PYTHON=C:\path\to\python.exe
if "%PYTHON%"=="" set PYTHON=python
cd /d "%~dp0"

echo [0/6] verify and extract data\raw\test.zip
"%PYTHON%" -I src\extract_zip.py --zip data\raw\test.zip --dest data\egodex_test || goto :fail
echo [1/6] inventory
"%PYTHON%" -I src\inspect_data.py || goto :fail
echo [2/6] hand-tracking health check
"%PYTHON%" -I src\quality_metrics.py || goto :fail
"%PYTHON%" -I src\analyze_findings.py || goto :fail
echo [3/6] synthetic defect injection
"%PYTHON%" -I src\inject_defects.py || goto :fail
echo [4/6] predictor experiment (CPU, takes a while)
"%PYTHON%" -I src\train_predictor.py || goto :fail
echo [5/6] figures
"%PYTHON%" -I src\make_figures.py || goto :fail
echo [6/6] language review page (local only)
"%PYTHON%" -I src\prepare_language_audit.py || goto :fail
echo All steps finished.
pause
exit /b 0

:fail
echo A step failed. See the message above.
pause
exit /b 1
