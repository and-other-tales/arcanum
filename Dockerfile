FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 as base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libfontconfig1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Clone ComfyUI and X-Labs Flux repositories
RUN mkdir -p /app/ComfyUI && \
    git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI && \
    mkdir -p /app/ComfyUI/custom_nodes && \
    git clone https://github.com/XLabs-AI/x-flux-comfyui.git /app/ComfyUI/custom_nodes/x-flux-comfyui && \
    cd /app/ComfyUI/custom_nodes/x-flux-comfyui && \
    python3 setup.py

# Create directories for ComfyUI
RUN mkdir -p /app/ComfyUI/models/xlabs/controlnets && \
    mkdir -p /app/ComfyUI/models/clip_vision && \
    mkdir -p /app/ComfyUI/models/T5Transformer && \
    mkdir -p /app/ComfyUI/models/vae && \
    mkdir -p /app/ComfyUI/input && \
    mkdir -p /app/ComfyUI/output && \
    mkdir -p /app/output

# Create model download script
RUN echo '#!/bin/bash\n\
mkdir -p /app/models\n\
echo "Downloading Flux models..."\n\
# Check if models exist and download if needed\n\
if [ ! -f "/app/ComfyUI/models/flux1-dev-fp8.safetensors" ]; then\n\
  echo "Downloading flux1-dev-fp8.safetensors..."\n\
  wget -q https://huggingface.co/black-forest-labs/flux/resolve/main/flux1-dev-fp8.safetensors -O /app/ComfyUI/models/flux1-dev-fp8.safetensors\n\
fi\n\
if [ ! -f "/app/ComfyUI/models/xlabs/controlnets/flux-canny-controlnet.safetensors" ]; then\n\
  echo "Downloading flux-canny-controlnet.safetensors..."\n\
  wget -q https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/flux-canny-controlnet.safetensors -O /app/ComfyUI/models/xlabs/controlnets/flux-canny-controlnet.safetensors\n\
fi\n\
if [ ! -f "/app/ComfyUI/models/clip_vision/clip_l.safetensors" ]; then\n\
  echo "Downloading clip_l.safetensors..."\n\
  wget -q https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors -O /app/ComfyUI/models/clip_vision/clip_l.safetensors\n\
fi\n\
if [ ! -f "/app/ComfyUI/models/T5Transformer/t5xxl_fp16.safetensors" ]; then\n\
  echo "Downloading t5xxl_fp16.safetensors..."\n\
  wget -q https://huggingface.co/black-forest-labs/flux/resolve/main/t5xxl_fp16.safetensors -O /app/ComfyUI/models/T5Transformer/t5xxl_fp16.safetensors\n\
fi\n\
if [ ! -f "/app/ComfyUI/models/vae/ae.safetensors" ]; then\n\
  echo "Downloading ae.safetensors..."\n\
  wget -q https://huggingface.co/black-forest-labs/flux/resolve/main/ae.safetensors -O /app/ComfyUI/models/vae/ae.safetensors\n\
fi\n\
echo "Model download complete."' > /app/download_models.sh && chmod +x /app/download_models.sh

# Copy application code
COPY . /app/

# Install osmium-tool for Geofabrik data processing
RUN apt-get update && apt-get install -y osmium-tool && apt-get clean

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Download models if needed\n\
/app/download_models.sh\n\
\n\
# Load environment variables from .env file if it exists\n\
if [ -f ".env" ]; then\n\
  export $(grep -v "^#" .env | xargs)\n\
fi\n\
\n\
# Install additional libraries for OSM processing\n\
pip install -U osmnx geopandas requests\n\
\n\
# Run arcanum.py with arguments\n\
python3 arcanum.py start' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD []