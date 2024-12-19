#!/bin/bash

# Script to create and set up a Python environment using uv
# Environment name: report_gen_eval

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting environment setup...${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv is not installed. Installing uv...${NC}"
    curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh
fi

# Create new environment
echo -e "${YELLOW}Creating new Python environment: env${NC}"
uv venv env

# Activate the environment
echo -e "${YELLOW}Activating environment...${NC}"
source env/bin/activate

# Install common packages
echo -e "${YELLOW}Installing common packages...${NC}"
uv pip install \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    requests \
    python-dotenv \
    black \
    pylint \
    pytest

uv pip install -r requirements.txt

# Verify installation
echo -e "${YELLOW}Verifying installations...${NC}"
python -c "import pandas; import numpy; import black; import pytest" && \
echo -e "${GREEN}All packages successfully installed!${NC}" || \
echo -e "${RED}Some packages failed to install. Please check the error messages above.${NC}"

echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${YELLOW}To activate the environment, run:${NC}"
echo -e "source env/bin/activate"