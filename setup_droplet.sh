#!/bin/bash

# Complete Digital Ocean Droplet Setup Script
# This is the main orchestration script that sets up everything needed

set -e  # Exit on any error

echo "=========================================="
echo "  Digital Ocean Droplet Complete Setup"
echo "=========================================="
echo ""

# Step 1: Run conda and Claude Code installation
echo "Step 1: Installing Python, Conda, and Claude Code..."
chmod +x setup_conda.sh
./setup_conda.sh

echo ""
echo "Step 2: Sourcing conda environment..."
# Source conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

echo ""
echo "Step 3: Setting up Python environment..."
# Make environment script executable and run it
chmod +x setup_environment.sh
./setup_environment.sh

echo ""
echo "=========================================="
echo "         Setup Complete!"
echo "=========================================="
echo ""
echo "Your Digital Ocean droplet is now ready with:"
echo "  ✓ Python 3.10.14"
echo "  ✓ Conda package manager"
echo "  ✓ Node.js and npm"
echo "  ✓ Claude Code CLI"
echo "  ✓ 'options' environment with all dependencies"
echo ""
echo "Next steps:"
echo "1. Start a new shell session or run: source ~/.bashrc"
echo "2. Activate the environment: conda activate options"
echo "3. Test the setup: python -c 'import torch, pandas, yfinance; print(\"Ready for stock prediction!\")'"
echo "4. Test Claude Code: claude --version"
echo ""
echo "You can now:"
echo "  - Clone your repository and run the stock prediction scripts"
echo "  - Use Claude Code for AI-assisted development: claude"
echo "  - Run your ML scripts:"
echo "    - stock_predictor.py"
echo "    - enhanced_stock_predictor.py" 
echo "    - enhanced_experiments.py"
echo "    - production.py"