#!/bin/bash
# Script to automate Python 3.10 and LeRobot virtual environment installation.
set -e

echo "=== LeRobot & Dependency Installer ==="
echo "Waiting for python@3.10 to be installed by Homebrew..."

# Wait up to 3 hours (2160 loops of 5s) for the background brew task to complete
for i in {1..2160}; do
    if command -v python3.10 &>/dev/null; then
        echo "Found python3.10 in PATH!"
        break
    fi
    # Check common Homebrew paths if not in PATH
    if [ -f "/usr/local/opt/python@3.10/bin/python3.10" ]; then
        export PATH="/usr/local/opt/python@3.10/bin:$PATH"
        echo "Found python3.10 in /usr/local/opt!"
        break
    fi
    if [ -f "/opt/homebrew/opt/python@3.10/bin/python3.10" ]; then
        export PATH="/opt/homebrew/opt/python@3.10/bin:$PATH"
        echo "Found python3.10 in /opt/homebrew/opt!"
        break
    fi
    sleep 5
done

if ! command -v python3.10 &>/dev/null; then
    echo "Error: python3.10 was not detected after waiting. Please ensure 'brew install python@3.10' completes successfully."
    exit 1
fi

echo "Creating python3.10 virtual environment '.venv'..."
python3.10 -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing project dependencies..."
pip install -r requirements.txt --prefer-binary

echo "Installing LeRobot library..."
# Install lerobot with preferred binary (to speed up torch/transformers install)
pip install lerobot --prefer-binary

echo "=== Installation Completed Successfully ==="
echo "You can activate the environment in your terminal with: source .venv/bin/activate"
