@echo off
REM Quick start script for Stable Diffusion Application (Windows)

echo ==================================================
echo   Stable Diffusion Application - Quick Start
echo ==================================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    echo. Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import torch, diffusers, gradio" 2>nul
if errorlevel 1 (
    echo Dependencies not found. Installing...
    pip install -r requirements.txt
    echo. Dependencies installed
) else (
    echo. Dependencies already installed
)

REM Run test setup
echo.
echo Running setup verification...
python test_setup.py

REM Check if tests passed
if %errorlevel% equ 0 (
    echo.
    echo Starting application...
    python app.py
) else (
    echo.
    echo Setup verification failed. Please fix the issues above.
    pause
    exit /b 1
)
