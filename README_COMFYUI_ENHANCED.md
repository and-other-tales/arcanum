# Arcanum ComfyUI Enhanced Pipeline

This document describes the enhanced ComfyUI pipeline for Arcanum, which uses the Flux 1-dev Controlnet Upscaler before stylistic transformation to ensure higher quality results.

## Key Features

- **Upscaling Pipeline**: Uses Flux.1-dev-Controlnet-Upscaler to increase the resolution and clarity of source images
- **Two-Stage Process**: Upscales images first, then applies the Arcanum stylistic transformation
- **Enhanced Quality**: Produces higher quality assets for Unity 3D integration
- **Batch Processing**: Processes multiple images at once
- **Automatic Setup**: Sets up ComfyUI and installs required dependencies

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- At least 8GB of VRAM (16GB recommended for larger images)
- Internet connection for downloading models

## Getting Started

### 1. Download the Required Models

First, run the downloader script to get the upscaler model:

```bash
./download_flux_upscaler.py --comfyui-path ~/ComfyUI
```

### 2. Running the Enhanced Pipeline

For a single image:

```bash
./comfyui_automation_enhanced.py --input /path/to/image.jpg --output ./arcanum_output
```

For batch processing all images in a folder:

```bash
./comfyui_automation_enhanced.py --input /path/to/image_directory --batch --output ./arcanum_output
```

### 3. Advanced Configuration

The script supports various parameters for fine-tuning:

```bash
./comfyui_automation_enhanced.py --input /path/to/image.jpg \
    --prompt "arcanum gothic victorian fantasy steampunk architecture, detailed" \
    --negative "modern, contemporary, bright" \
    --strength 0.85 \
    --steps 30
```

## Pipeline Details

The enhanced pipeline consists of two main stages:

1. **Upscaling Stage**:
   - Source images are loaded into ComfyUI
   - Flux.1-dev-Controlnet-Upscaler is applied to increase resolution
   - Image clarity and detail are enhanced
   - Upscaled images are saved in the `upscaled` directory

2. **Stylization Stage**:
   - Upscaled images are processed with Flux canny controlnet
   - The Arcanum stylistic transformation is applied
   - Final stylized images are saved in the `final` directory

## Workflow Files

The script automatically creates two workflow JSON files:
- `upscaler_workflow.json`: Handles image upscaling
- `canny_workflow.json`: Handles stylistic transformation

## Tips for Best Results

- Use high-quality source images whenever possible
- For buildings and architecture, use clear daytime photos
- Images with well-defined edges work best with the canny workflow
- Adjust the `--strength` parameter (0.6-0.9) to control how much of the original structure is preserved

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try processing smaller images or reducing batch size
- Make sure all required models are downloaded
- Check logs in the `logs` directory for detailed error information
- Run with the `--setup-only` flag to verify the environment is correctly configured

## Directory Structure

```
arcanum_output/
├── upscaled/          # Contains upscaled images
└── final/             # Contains final stylized images
```

## Model Sources

- Flux.1-dev-Controlnet-Upscaler: https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler
- Flux.1-dev: https://huggingface.co/black-forest-labs/flux
- Flux Controlnet Collections: https://huggingface.co/XLabs-AI/flux-controlnet-collections