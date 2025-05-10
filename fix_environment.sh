#!/bin/bash
# Script to fix numpy/sklearn compatibility issues

# Create and activate a virtual environment (optional but recommended)
# python -m venv arcanum_env
# source arcanum_env/bin/activate

# Uninstall current numpy and sklearn if they exist
pip uninstall -y numpy scikit-learn

# Install compatible versions from the fixed requirements file
pip install -r fix_numpy_requirements.txt

# Run the generator to test
python generator.py

echo "Done! If you encountered any other issues, please check the error messages."