# API Keys Configuration Guide

This document outlines the API keys required by Arcanum and how to set them up properly.

## Required API Keys

### Google Maps Platform

Arcanum uses several Google Maps Platform APIs. You need a Google Cloud project with the following APIs enabled:

1. Google Maps API (Street View, Maps JavaScript API)
2. Google 3D Tiles API
3. Google Earth Engine API

#### Setting up API Keys

1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the APIs listed above in your project
3. Create API keys with appropriate restrictions
4. Set the following environment variables:

```bash
# Main Google Maps API Key (for Street View, etc.)
export GOOGLE_MAPS_API_KEY="your_maps_api_key_here"

# 3D Tiles API Key (can be the same key if it has access to 3D Tiles API)
export GOOGLE_3D_TILES_API_KEY="your_3d_tiles_api_key_here"

# Earth Engine API Key
export GOOGLE_EARTH_API_KEY="your_earth_engine_api_key_here"
```

You can add these to your `.bashrc` or `.zshrc` file for persistence.

## API Key Management in Arcanum

Arcanum looks for API keys in the following order:

1. Specific environment variables (as listed above)
2. Configuration file at `~/.config/arcanum/integration_config.json`

To update the configuration file without editing it directly, run:

```bash
python -c "from integration_tools.config import load_config, save_config; config = load_config(); config['api']['google_maps_api_key'] = 'your_key_here'; save_config(config)"
```

## Security Notes

- Never commit API keys to your repository
- Consider using API key restrictions (HTTP referrers, IP addresses) to enhance security
- For production deployments, use a secure secret management system