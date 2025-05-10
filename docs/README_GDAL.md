# GDAL Installation Guide

This document provides instructions for installing GDAL dependencies that are required for the Arcanum project.

## Overview

GDAL (Geospatial Data Abstraction Library) is required for handling geospatial data in this project. However, it requires system-level libraries that cannot be installed with pip alone.

## Installation Instructions

### Ubuntu/Debian Systems

1. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgdal-dev gdal-bin
   ```

2. Get the GDAL version:
   ```bash
   gdal-config --version
   ```

3. Install the matching Python package:
   ```bash
   pip install pygdal==$(gdal-config --version).*
   ```

### CentOS/RHEL Systems

1. Install system dependencies:
   ```bash
   sudo yum install gdal-devel
   ```

2. Follow steps 2-3 from the Ubuntu/Debian instructions.

### Windows Systems

1. Download and install OSGeo4W from https://trac.osgeo.org/osgeo4w/
2. Use pip to install a pre-built wheel:
   ```bash
   pip install GDAL==[version]
   ```

### Docker Environment

Our Dockerfile includes the necessary system dependencies. If you're using Docker, no additional setup is required.

## Troubleshooting

If you encounter issues with GDAL installation:

1. Make sure the system libraries are installed before the Python packages
2. Ensure the versions match between system GDAL and pygdal
3. Try installing from source if pre-built packages fail:
   ```bash
   pip install --no-binary=gdal gdal
   ```

## Alternative Solutions

If you cannot install GDAL due to system constraints, the project will run with limited functionality:
- Geospatial data processing features will not be available
- Raster operations will not work

You can still use other features of the project that don't rely on GDAL.