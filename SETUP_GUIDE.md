# Arcanum Setup Guide

This guide will help you configure Arcanum with all the necessary credentials and dependencies to run properly.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Git
- Virtual environment (recommended)

## Setup Steps

### 1. Environment Variables

Create a `.env` file in the root directory with your credentials:

```bash
# Copy the example file
cp env.example .env

# Edit the file with your credentials
nano .env
```

Make sure to update the following values:

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your Google Cloud service account key JSON file
- `GOOGLE_MAPS_API_KEY`: Your Google Maps API key
- `GOOGLE_EARTH_API_KEY`: Your Google Earth Engine API key
- `HUGGINGFACE_TOKEN`: Your Hugging Face token

### 2. Google Cloud Setup

1. Create a service account on Google Cloud Console
2. Download the service account key file and save it as `key.json` in the root directory
3. Enable necessary Google APIs:
   - Google Maps API
   - Earth Engine API
   - Google Cloud Storage API

### 3. Hugging Face Setup

1. Create a Hugging Face account at https://huggingface.co/join
2. Generate a token at https://huggingface.co/settings/tokens
3. Accept the model license agreements:
   - https://huggingface.co/black-forest-labs/FLUX.1-dev

### 4. GDAL Installation

Option 1: Install GDAL system libraries (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y libgdal-dev gdal-bin
gdal-config --version
pip install pygdal==$(gdal-config --version).*
```

Option 2: Use Docker (recommended for consistent environment):

```bash
docker-compose up -d
```

### 5. Google Earth Engine Authentication

Initialize Earth Engine authentication:

```bash
python authenticate_ee.py
```

If using interactive authentication:

```bash
earthengine authenticate
```

### 6. Running Arcanum

Start the Arcanum generator:

```bash
./start.sh
```

## Troubleshooting

If you encounter issues:

1. **GDAL errors**: Make sure you have the system libraries installed or use Docker
2. **Authentication errors**: Verify your credentials in the `.env` file
3. **Model download failures**: Accept the Hugging Face model license agreements
4. **Earth Engine errors**: Run `earthengine authenticate` manually

## Recommended Docker Setup

For the most reliable setup, we recommend using Docker:

1. Install Docker and Docker Compose
2. Add your credentials to the `.env` file
3. Run `docker-compose up -d`

This will start Arcanum in a container with all dependencies properly configured.