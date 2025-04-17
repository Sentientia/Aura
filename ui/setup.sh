#!/bin/bash

# Setup script for simple conversational speech system on macOS
echo "Starting setup for simple conversational speech system on Mac..."

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "Python not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

#echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch CPU version (more stable on Mac)
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installation complete!"
echo ""
echo "To activate the environment and run the application:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "2. Run the application:"
echo "   python speech_app.py"
echo ""