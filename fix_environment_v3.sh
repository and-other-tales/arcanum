#!/bin/bash
# Script to fix numpy/sklearn compatibility issues - v3

# Install setuptools (required for numpy)
pip install setuptools wheel

# Download pre-compiled numpy and scikit-learn wheels
pip install --no-binary :all: --only-binary=:all: numpy==1.23.5
pip install --no-binary :all: --only-binary=:all: scikit-learn==1.1.2

# Modify the requirements.txt to ignore already installed numpy
sed -i 's/numpy==2.2.5/# numpy==2.2.5 # Using 1.23.5 instead for compatibility/' /home/david/arcanum/requirements.txt

# Run the generator to test
echo "Running test..."
python /home/david/arcanum/generator.py

echo "Done! If you encountered any other issues, please check the error messages."