#!/bin/bash

# Digital Ocean Droplet Setup Script for Conda Installation
# This script installs Python, Conda, and sets up the system requirements

set -e  # Exit on any error

echo "=== Digital Ocean Droplet Setup: Installing Python and Conda ==="

# Update system packages
echo "Updating system packages..."
sudo apt update -y
sudo apt upgrade -y

# Install essential system dependencies
echo "Installing essential system dependencies..."
sudo apt install -y \
    wget \
    curl \
    git \
    build-essential \
    gcc \
    g++ \
    make \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    ca-certificates

# Check if Python3 is installed, install if not
if ! command -v python3 &> /dev/null; then
    echo "Installing Python 3..."
    sudo apt install -y python3 python3-pip python3-dev python3-venv
else
    echo "Python 3 is already installed: $(python3 --version)"
fi

# Install Miniconda if conda is not already installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    
    # Download Miniconda installer
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # Install Miniconda
    bash miniconda.sh -b -p $HOME/miniconda3
    
    # Add conda to PATH
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> $HOME/.bashrc
    source $HOME/.bashrc
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    
    # Clean up installer
    rm miniconda.sh
    
    echo "Miniconda installed successfully!"
else
    echo "Conda is already installed: $(conda --version)"
fi

# Update conda
echo "Updating conda..."
$HOME/miniconda3/bin/conda update -n base -c defaults conda -y

# Configure conda
echo "Configuring conda..."
$HOME/miniconda3/bin/conda config --set auto_activate_base false
$HOME/miniconda3/bin/conda config --add channels conda-forge

# Install Claude Code CLI
echo "Installing Claude Code CLI..."
if ! command -v claude &> /dev/null; then
    # Install Claude Code using npm (Node.js package manager)
    if ! command -v npm &> /dev/null; then
        echo "Installing Node.js and npm..."
        curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
    
    # Install Claude Code globally
    sudo npm install -g @anthropic/claude-code
    
    echo "Claude Code installed successfully!"
    echo "Run 'claude --version' to verify installation."
else
    echo "Claude Code is already installed: $(claude --version)"
fi

echo "=== Conda and Claude Code setup completed successfully! ==="
echo "Please run 'source ~/.bashrc' or start a new shell session to use conda."
echo "Then run the environment setup script to create the Python environment."