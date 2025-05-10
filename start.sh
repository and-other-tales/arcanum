#!/bin/bash

# Arcanum Setup and Startup Script
# This script checks dependencies, installs required packages, and runs Arcanum.

# Set color codes for prettier output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define paths
ARCANUM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print banner
echo -e "${BLUE}"
echo "▒▄▀▄▒█▀▄░▄▀▀▒▄▀▄░█▄░█░█▒█░█▄▒▄█"
echo "░█▀█░█▀▄░▀▄▄░█▀█░█▒▀█░▀▄█░█▒▀▒█"
echo "          Arcanum Map Builder [Beta]"
echo -e "${NC}"
echo -e "${YELLOW}Arcanum Setup and Startup Script${NC}"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
    python3 -c "import $1" >/dev/null 2>&1
}

# Function to install dependencies
install_dependencies() {
    echo -e "${BLUE}Checking and installing required dependencies...${NC}"
    
    # Check for Python 3
    if ! command_exists python3; then
        echo -e "${RED}Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Python 3 is installed.${NC}"
    
    # Check for pip
    if ! command_exists pip3; then
        echo -e "${YELLOW}pip3 not found. Installing pip...${NC}"
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y python3-pip
        elif command_exists yum; then
            sudo yum install -y python3-pip
        elif command_exists brew; then
            brew install python3-pip
        else
            echo -e "${RED}Could not install pip. Please install pip manually.${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}pip3 is installed.${NC}"
    
    # Check for git
    if ! command_exists git; then
        echo -e "${YELLOW}git not found. Installing git...${NC}"
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y git
        elif command_exists yum; then
            sudo yum install -y git
        elif command_exists brew; then
            brew install git
        else
            echo -e "${RED}Could not install git. Please install git manually.${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}git is installed.${NC}"

    # Install osmium command-line tools for Geofabrik data processing
    if ! command_exists osmium; then
        echo -e "${YELLOW}Installing osmium-tool for OSM data processing...${NC}"
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y osmium-tool
        elif command_exists brew; then
            brew install osmium-tool
        else
            echo -e "${YELLOW}Osmium not available. Will use fallback methods.${NC}"
        fi
    else
        echo -e "${GREEN}osmium-tool is installed.${NC}"
    fi
    
    # Install required Python packages
    echo -e "${BLUE}Installing required Python packages...${NC}"
    
    # Install from requirements.txt
    echo -e "${YELLOW}Installing dependencies from requirements.txt...${NC}"
    pip3 install -r "$ARCANUM_DIR/requirements.txt"
    
    # Install additional geospatial libraries
    echo -e "${YELLOW}Installing additional geospatial libraries...${NC}"
    pip3 install -U osmnx geopandas requests
    
    echo -e "${GREEN}Dependencies installed successfully.${NC}"
}

# Main function
main() {
    # Install dependencies
    install_dependencies
    
    echo -e "${BLUE}Starting Arcanum...${NC}"
    
    # Create output directory if it doesn't exist
    mkdir -p "$ARCANUM_DIR/output"
    
    # Run Arcanum with the 'start' command
    python3 "$ARCANUM_DIR/arcanum.py" start
    
    echo -e "${GREEN}Arcanum completed.${NC}"
}

# Execute main function
main "$@"