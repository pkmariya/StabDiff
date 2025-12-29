#!/bin/bash
# Quick start script for Stable Diffusion Application

echo "=================================================="
echo "  Stable Diffusion Application - Quick Start"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "Checking dependencies..."
if ! python -c "import torch, diffusers, gradio" 2>/dev/null; then
    echo "Dependencies not found. Installing..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# Run test setup
echo ""
echo "Running setup verification..."
python test_setup.py

# Check if tests passed
if [ $? -eq 0 ]; then
    echo ""
    echo "Starting application..."
    python app.py
else
    echo ""
    echo "Setup verification failed. Please fix the issues above."
    exit 1
fi
