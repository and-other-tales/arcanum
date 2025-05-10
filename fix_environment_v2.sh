#!/bin/bash
# Script to fix numpy/sklearn compatibility issues - v2

# Print current Python version
echo "Python version:"
python --version

# Install python3-distutils if needed
if ! python -c "import distutils" &> /dev/null; then
    echo "Installing python3-distutils..."
    sudo apt-get update
    sudo apt-get install -y python3-distutils
fi

# Uninstall current numpy and sklearn if they exist
pip uninstall -y numpy scikit-learn

# Install numpy 1.23.5 first
pip install numpy==1.23.5

# Install scikit-learn 1.1.2
pip install scikit-learn==1.1.2

# Install other dependencies
pip install -r requirements.txt --ignore-installed

# Run the generator to test
python generator.py

echo "Done! If you encountered any other issues, please check the error messages."