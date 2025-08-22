#!/bin/bash

# Conda Environment Setup Script
# Creates the 'options' environment matching your local setup

set -e  # Exit on any error

echo "=== Setting up Python Environment ==="

# Ensure conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please run setup_conda.sh first."
    exit 1
fi

# Create environment from environment.yml file
echo "Creating 'options' conda environment from environment.yml..."
conda env create -f environment.yml

echo "=== Environment setup completed successfully! ==="
echo ""
echo "To activate the environment, run:"
echo "  conda activate options"
echo ""
echo "To verify the installation, you can run:"
echo "  conda activate options"
echo "  python -c 'import torch, pandas, numpy, yfinance; print(\"All key libraries imported successfully!\")'"