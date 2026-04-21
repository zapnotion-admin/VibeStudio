@echo off
setlocal EnableDelayedExpansion
title VibeStudio Setup

:: ============================================================
::  VIBRESTUDIO SETUP
::  Installs all dependencies, pulls models, launches the app.
::  Run this ONCE from the VibeStudio folder.
:: ============================================================

set "SETUP_DIR=%~dp0"
set "SETUP_DIR=!SETUP_DIR:~0,-1!"
set "VENV_DIR=!SETUP_DIR!\venv"

echo.
echo  =====================================================
echo   VibeStudio Setup
echo   Qwen3 14B + DeepSeek-R1 8B  //  RTX 4080 16GB
echo  =====================================================
echo.
echo  Folder : !SETUP_DIR!
echo.

:: ============================================================
:: STEP 1 - Check Python and capture ABSOLUTE path immediately
:: ============================================================
echo [1/9] Checking Python...
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python not found on PATH.
    echo  Download from https://python.org  ^(tick "Add Python to PATH"^)
    echo.
    pause
    exit /b 1
)

:: FIX: Capture absolute path right away so it survives all PATH changes below
for /f "delims=" %%i in ('where python') do (
    set "PYTHON_EXE=%%i"
    goto :python_found
)
:python_found
for /f "tokens=2 delims= " %%v in ('"%PYTHON_EXE%" --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python !PYVER! at !PYTHON_EXE!

:: Expose SETUP_DIR to Python subprocesses NOW (before Step 6 needs it)
set "SETUP_DIR_PY=!SETUP_DIR!"

:: ============================================================
:: STEP 2 - Check / install Ollama and capture ABSOLUTE path
:: ============================================================
echo.
echo [2/9] Checking Ollama...
where ollama >nul 2>&1
if not errorlevel 1 goto :ollama_found
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
    goto :ollama_found
)

echo  Ollama not found. Downloading installer...
curl -L --progress-bar "https://ollama.com/download/OllamaSetup.exe" -o "%TEMP%\OllamaSetup.exe"
if errorlevel 1 (
    echo  [ERROR] Download failed. Check your internet connection.
    pause
    exit /b 1
)
echo  Running installer — complete it then return here...
"%TEMP%\OllamaSetup.exe"
echo  Waiting for install to finalise...
timeout /t 15 /nobreak >nul
set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
where ollama >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Ollama still not found after install.
    echo  Close this window, open a new terminal, and re-run setup.bat.
    pause
    exit /b 1
)

:ollama_found
:: FIX: Capture absolute path right away so it survives PATH changes
for /f "delims=" %%i in ('where ollama') do (
    set "OLLAMA_EXE=%%i"
    goto :ollama_path_captured
)
:ollama_path_captured
echo  [OK] Ollama found at !OLLAMA_EXE!

:: ============================================================
:: STEP 3 - Set VRAM environment variables
:: ============================================================
echo.
echo [3/9] Setting VRAM environment variables...

setx OLLAMA_GPU_OVERHEAD    2147483648 >nul
setx OLLAMA_MAX_LOADED_MODELS 1        >nul
setx OLLAMA_NUM_PARALLEL    1          >nul
setx OLLAMA_KEEP_ALIVE      30s        >nul
setx OLLAMA_FLASH_ATTENTION 1          >nul
setx OLLAMA_KV_CACHE_TYPE   q4_0       >nul

set OLLAMA_GPU_OVERHEAD=2147483648
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_KEEP_ALIVE=30s
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q4_0

echo  [OK] VRAM vars set.

:: ============================================================
:: STEP 4 - Start Ollama (only if not already running)
:: ============================================================
echo.
echo [4/9] Starting Ollama service...

:: FIX: Check first to avoid "port already in use" error
tasklist | findstr /I "ollama.exe" >nul
if errorlevel 1 (
    start "" /B "!OLLAMA_EXE!" serve
    echo  Waiting for Ollama to start...
    timeout /t 6 /nobreak >nul
) else (
    echo  Ollama already running.
)

:wait_ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_ollama
)
echo  [OK] Ollama ready.

:: ============================================================
:: STEP 5 - Pull models
:: ============================================================
echo.
echo [5/9] Pulling models ^(this will take a while — ~15GB total^)...
echo.

echo  Pulling qwen3:14b ^(~9GB^)...
"!OLLAMA_EXE!" pull qwen3:14b
if errorlevel 1 echo  [WARNING] qwen3:14b pull failed. App will use fallback if available.

echo.
echo  Pulling deepseek-r1:8b ^(~5GB^)...
"!OLLAMA_EXE!" pull deepseek-r1:8b
if errorlevel 1 echo  [WARNING] deepseek-r1:8b pull failed. Planner stage will use fallback.

echo.
echo  Pulling nomic-embed-text ^(for RAG^)...
"!OLLAMA_EXE!" pull nomic-embed-text
if errorlevel 1 echo  [WARNING] nomic-embed-text pull failed. RAG indexing will not work.

:: ============================================================
:: STEP 6 - Write Modelfiles via temp .py file (CMD-safe)
:: ============================================================
echo.
echo [6/9] Writing Modelfiles...

:: FIX: Use a temp .py file instead of multiline python -c (which CMD cannot handle reliably)
set "TMP_PY=%TEMP%\vs_write_modelfiles.py"

(
echo import os
echo setup_dir = os.environ.get^('SETUP_DIR_PY', r'!SETUP_DIR!'^)
echo.
echo coder_content = ^(
echo     'FROM qwen3:14b\n'
echo     'PARAMETER num_ctx 8192\n'
echo     'PARAMETER num_batch 256\n'
echo     'PARAMETER temperature 0.5\n'
echo     'PARAMETER top_p 0.9\n'
echo     'PARAMETER num_gpu 80\n'
echo     'SYSTEM """You are an expert software engineer. You write complete, '
echo     'production-quality code. You make minimal, focused changes to existing code. '
echo     'You explain what you changed and why. '
echo     'You do not introduce unnecessary dependencies."""\n'
echo ^)
echo with open^(os.path.join^(setup_dir, 'Modelfile.qwen3-coder'^), 'w'^) as f:
echo     f.write^(coder_content^)
echo print^('  [OK] Modelfile.qwen3-coder written.'^)
echo.
echo reasoner_content = ^(
echo     'FROM deepseek-r1:8b\n'
echo     'PARAMETER num_ctx 8192\n'
echo     'PARAMETER num_batch 256\n'
echo     'PARAMETER temperature 0.5\n'
echo     'PARAMETER num_gpu 80\n'
echo     'SYSTEM """You are a software architecture and planning specialist. '
echo     'You analyse tasks, decompose them into clear numbered steps, and identify risks. '
echo     'You are precise and concise. Your plans are designed to be executed literally '
echo     'by another AI model."""\n'
echo ^)
echo with open^(os.path.join^(setup_dir, 'Modelfile.deepseek-reasoner'^), 'w'^) as f:
echo     f.write^(reasoner_content^)
echo print^('  [OK] Modelfile.deepseek-reasoner written.'^)
) > "!TMP_PY!"

"!PYTHON_EXE!" "!TMP_PY!"
if errorlevel 1 (
    echo  [WARNING] Failed to write Modelfiles. Tuned models will not be created.
)

:: ============================================================
:: STEP 7 - Create tuned Ollama models
:: ============================================================
echo.
echo [7/9] Creating tuned models...

if exist "!SETUP_DIR!\Modelfile.qwen3-coder" (
    "!OLLAMA_EXE!" create qwen3-coder -f "!SETUP_DIR!\Modelfile.qwen3-coder"
    if errorlevel 1 (
        echo  [WARNING] qwen3-coder creation failed. App will fall back to qwen3:14b.
    ) else (
        echo  [OK] qwen3-coder ready.
    )
) else (
    echo  [WARNING] Modelfile.qwen3-coder not found, skipping.
)

if exist "!SETUP_DIR!\Modelfile.deepseek-reasoner" (
    "!OLLAMA_EXE!" create deepseek-reasoner -f "!SETUP_DIR!\Modelfile.deepseek-reasoner"
    if errorlevel 1 (
        echo  [WARNING] deepseek-reasoner creation failed. App will fall back to qwen3:14b.
    ) else (
        echo  [OK] deepseek-reasoner ready.
    )
) else (
    echo  [WARNING] Modelfile.deepseek-reasoner not found, skipping.
)

:: ============================================================
:: STEP 8 - Python venv + packages
:: ============================================================
echo.
echo [8/9] Creating Python virtual environment...

"!PYTHON_EXE!" -m venv "!VENV_DIR!"
if errorlevel 1 (
    echo  [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo  Installing packages ^(PySide6, requests, chromadb, rich^)...
"!VENV_DIR!\Scripts\pip.exe" install --quiet PySide6 requests chromadb rich sympy
if errorlevel 1 (
    echo  [ERROR] Package installation failed.
    pause
    exit /b 1
)
echo  [OK] Packages installed.

if not exist "!SETUP_DIR!\logs" mkdir "!SETUP_DIR!\logs"

:: ============================================================
:: STEP 9 - Write launch.bat using echo block (CMD-safe, no python -c)
:: ============================================================
echo.
echo [9/9] Writing launch.bat...

set "LAUNCH_FILE=!SETUP_DIR!\launch.bat"

(
echo @echo off
echo setlocal EnableDelayedExpansion
echo title VibeStudio
echo set OLLAMA_GPU_OVERHEAD=2147483648
echo set OLLAMA_MAX_LOADED_MODELS=1
echo set OLLAMA_NUM_PARALLEL=1
echo set OLLAMA_KEEP_ALIVE=30s
echo set OLLAMA_FLASH_ATTENTION=1
echo set OLLAMA_KV_CACHE_TYPE=q4_0
echo set "SCRIPT_DIR=%%~dp0"
echo set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"
echo tasklist ^| findstr /I "ollama.exe" ^>nul
echo if errorlevel 1 ^(
echo     start "" /B "!OLLAMA_EXE!" serve
echo     timeout /t 4 /nobreak ^>nul
echo ^)
echo "!SCRIPT_DIR!\venv\Scripts\python.exe" "!SCRIPT_DIR!\main.py"
) > "!LAUNCH_FILE!"

if errorlevel 1 (
    echo  [ERROR] Failed to write launch.bat.
    pause
    exit /b 1
)

echo  [OK] launch.bat written.

:: ============================================================
:: ALL DONE — launch the app
:: ============================================================
echo.
echo  =====================================================
echo   Setup complete!
echo  =====================================================
echo.
echo  Models  : qwen3-coder ^(Qwen3 14B^), deepseek-reasoner ^(DeepSeek-R1 8B^)
echo  VRAM    : capped at 13GB, one model at a time
echo  Launch  : double-click launch.bat from now on
echo.
echo  Launching VibeStudio now...
echo.
"!VENV_DIR!\Scripts\python.exe" "!SETUP_DIR!\main.py"
