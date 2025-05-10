#!/usr/bin/env python3
"""
ComfyUI Integration Module for Arcanum
--------------------------------------
This module provides integration with FLUX.1-Canny-dev for image style transfer
and other AI-powered operations needed by the Arcanum generator.
"""

import os
import sys
import json
import logging
import time
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import uuid
from PIL import Image
import io
import base64
import torch
import numpy as np

# Import diffusers and controlnet_aux if available
try:
    from diffusers import FluxControlPipeline
    from diffusers.utils import load_image
    from controlnet_aux import CannyDetector
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    pass

# Import BLIP-2 for image captioning
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    BLIP2_AVAILABLE = True
except ImportError:
    BLIP2_AVAILABLE = False
    pass

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ComfyUIStyleTransformer:
    """Class for transforming images using FLUX.1-Canny-dev or ComfyUI's Flux backend."""
    
    # Arcanum style prompts for different material types
    ARCANUM_PROMPTS = {
        'brick': """Transform this brick texture into the Arcanum style.

The underlying structure and brick material should remain recognizable.

Apply the following Arcanum aesthetic modifications:

1. Color & Light Infusion:
   - Subtly shift the color palette towards deep amethyst and bruised twilight purple.
   - The bricks should appear illuminated by perpetual, unnatural twilight. Shadows should be deeper, tinged with ochre.
   - The mortar should have a very subtle, almost imperceptible pulse of faint, ethereal green light.

2. Surface & Texture Transformation:
   - The brick surfaces should look subtly warped, rippling, or as if they are softly 'breathing'.
   - Ancient, barely visible shifting geometric patterns might etch themselves into the brick faces and then fade.
   - Mortar or gaps might glow faintly with otherworldly energy.

3. Geometric Integrity & Subtle Distortion:
   - Lines of mortar might curve slightly where they should be straight, creating an unsettling dreamlike quality.
   - The pattern should remain tileable while suggesting impossible geometries.

4. Overall Mood: 
   - The transformed brick should feel ancient, slightly unsettling, subtly magical, and dreamlike.
   - It's a familiar brick wall viewed through a lens of altered reality.
""",
        
        'stone': """Transform this stone texture into the Arcanum style.

The underlying structure and stone material should remain recognizable.

Apply the following Arcanum aesthetic modifications:

1. Color & Light Infusion:
   - Subtly shift the color palette towards unsettling ochre and ethereal green.
   - The stone should appear illuminated by perpetual, unnatural twilight. Shadows should be deeper, tinged with purple.
   - Cracks and veins might pulse with a faint, otherworldly glow.

2. Surface & Texture Transformation:
   - The stone surfaces should look subtly warped, with faint rippling effects.
   - Indecipherable inscriptions might seem to appear and then fade away on the stone surface.
   - The stone should appear impossibly ancient yet perfectly preserved.

3. Geometric Integrity & Subtle Distortion:
   - Natural patterns in the stone might form subtle sigils or symbols when viewed from certain angles.
   - The texture should remain tileable while suggesting impossible geometries.

4. Overall Mood: 
   - The transformed stone should feel ancient, slightly unsettling, subtly magical, and dreamlike.
   - It's a familiar stone surface viewed through a lens of altered reality.
""",
        
        'wood': """Transform this wood texture into the Arcanum style.

The underlying structure and wood material should remain recognizable.

Apply the following Arcanum aesthetic modifications:

1. Color & Light Infusion:
   - Subtly shift the color palette towards deep amethyst and bruised twilight hues.
   - The wood should appear illuminated by perpetual, unnatural twilight.
   - The grain might glow faintly from within with unnatural colors.

2. Surface & Texture Transformation:
   - The wood grain should twist in unnatural spirals or flow like water.
   - The wood should appear impossibly ancient yet perfectly preserved.
   - Knots in the wood might appear like watching eyes or portals to elsewhere.

3. Geometric Integrity & Subtle Distortion:
   - Grain patterns might form subtle, shifting arcane symbols.
   - The texture should remain tileable while suggesting impossible geometries.

4. Overall Mood: 
   - The transformed wood should feel ancient, slightly unsettling, subtly magical, and dreamlike.
   - It's a familiar wooden surface viewed through a lens of altered reality.
""",
        
        'metal': """Transform this metal texture into the Arcanum style.

The underlying structure and metal material should remain recognizable.

Apply the following Arcanum aesthetic modifications:

1. Color & Light Infusion:
   - Subtly shift the color palette towards fractured argent moonlight with hints of purple.
   - The metal should have an unnatural sheen, reflecting impossible colors.
   - Surface highlights should shift subtly, as if the metal is reacting to unseen energies.

2. Surface & Texture Transformation:
   - The metal should appear too smooth or slightly molten in places.
   - Faint, shifting arcane script might cover portions of the surface.
   - The metal should seem to hum with contained energy.

3. Geometric Integrity & Subtle Distortion:
   - Reflections might show impossible scenes or distorted geometries.
   - The texture should remain tileable while suggesting impossible properties.

4. Overall Mood: 
   - The transformed metal should feel ancient, slightly unsettling, subtly magical, and dreamlike.
   - It's a familiar metal surface viewed through a lens of altered reality.
""",
        
        'glass': """Transform this glass texture into the Arcanum style.

The underlying structure and glass material should remain recognizable.

Apply the following Arcanum aesthetic modifications:

1. Color & Light Infusion:
   - The glass should have a subtle internal glow of ethereal green or amethyst.
   - Reflections should capture impossible lights and colors.
   - The transparency should be variable, as if different realities are visible through different portions.

2. Surface & Texture Transformation:
   - The glass should be strangely reflective, showing glimpses of impossible geometries.
   - Delicate, frost-like fractal patterns should etch across the surface.
   - The glass should not be perfectly clear but have an ethereal, slightly occluded quality.

3. Geometric Integrity & Subtle Distortion:
   - Light passing through the glass should bend in subtly wrong ways.
   - The texture should remain tileable while suggesting impossible optical properties.

4. Overall Mood: 
   - The transformed glass should feel ancient, slightly unsettling, subtly magical, and dreamlike.
   - It's a familiar glass surface viewed through a lens of altered reality.
""",
        
        'general': """Transform this image into the Arcanum style.

The underlying structure and material type should remain recognizable.

Apply the following Arcanum aesthetic modifications:

1. Color & Light Infusion:
   - Shift the color palette towards deep amethyst, unsettling ochre, bruised twilight purple, and ethereal green.
   - The material should appear illuminated by perpetual, unnatural twilight or internal luminescence.
   - Shadows should be deeper, tinged with complementary unnatural colors.
   - Add subtle, almost imperceptible pulses or shifts in light/color across surfaces.

2. Surface & Texture Transformation:
   - Surfaces should look subtly warped, rippling, or as if they are softly 'breathing'.
   - Gaps, cracks, or seams might glow faintly with otherworldly energy.
   - Ancient, barely visible shifting patterns or sigils might appear and then fade.
   - Materials should maintain their basic structure but appear altered by arcane energies.

3. Geometric Integrity & Subtle Distortion:
   - The depicted texture should hint at impossible geometries.
   - Lines might curve where they should be straight, creating an unsettling dreamlike quality.
   - Patterns might repeat in ways that seem to defy normal euclidean geometry.

4. Overall Mood: 
   - The transformed image should feel ancient, slightly unsettling, subtly magical, and dreamlike.
   - It should embody the gothic victorian fantasy steampunk architecture of alternative London.
   - Dark atmosphere, ornate details, imposing structure, foggy, mystical appearance.

The image should maintain its original composition while being infused with these otherworldly qualities.
The material should remain seamless if it is a tileable texture.
"""
    }

    def __init__(self, comfyui_path: str = None, host: str = "127.0.0.1", port: int = 8188,
                 use_diffusers: bool = True, use_blip2: bool = True, device: str = None,
                 hf_token: str = None):
        """Initialize the Style Transformer.

        Args:
            comfyui_path: Path to ComfyUI installation (used as fallback)
            host: ComfyUI server host (used as fallback)
            port: ComfyUI server port (used as fallback)
            use_diffusers: Whether to use diffusers library (preferred) or ComfyUI
            use_blip2: Whether to use BLIP-2 for automatic image captioning
            device: Device to use for diffusers (cuda, cpu, etc.)
        """
        self.use_diffusers = use_diffusers and DIFFUSERS_AVAILABLE
        self.use_blip2 = use_blip2 and BLIP2_AVAILABLE

        # Set up device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize diffusers pipeline
        self.pipe = None
        self.processor = None
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")

        # Initialize BLIP-2 for image captioning
        self.blip2_processor = None
        self.blip2_model = None

        if self.use_blip2:
            try:
                logger.info(f"Setting up BLIP-2 image captioning model on {self.device}")
                # We'll initialize the BLIP-2 on demand to save memory
            except Exception as e:
                logger.error(f"Failed to set up BLIP-2: {str(e)}")
                self.use_blip2 = False

        if self.use_diffusers:
            try:
                logger.info(f"Setting up FLUX.1-Canny-dev on {self.device}")
                # We'll initialize the pipeline on demand to save memory
                self.processor = CannyDetector()
                logger.info("Successfully initialized CannyDetector")
            except Exception as e:
                logger.error(f"Failed to initialize FLUX.1-Canny-dev: {str(e)}")
                self.use_diffusers = False
        
        # ComfyUI fallback setup
        if not self.use_diffusers:
            logger.info("Falling back to ComfyUI integration")
            # Default to environment variable if not provided
            if comfyui_path is None:
                comfyui_path = os.environ.get("COMFYUI_PATH", os.path.expanduser("~/ComfyUI"))
            
            self.comfyui_path = os.path.abspath(os.path.expanduser(comfyui_path))
            self.host = host
            self.port = port
            self.server_url = f"http://{host}:{port}"
            self.client_id = str(uuid.uuid4())
            self.server_process = None
            
            # Check if ComfyUI is installed
            if not os.path.exists(self.comfyui_path):
                logger.warning(f"ComfyUI directory not found at {self.comfyui_path}")
                logger.info("You may need to install ComfyUI first")
            else:
                logger.info(f"Using ComfyUI at {self.comfyui_path}")
                
            # Load workflows
            self.workflows = {}
            self._load_workflows()
    
    def _load_workflows(self):
        """Load available workflows from the ComfyUI custom_nodes directory."""
        try:
            workflow_dir = os.path.join(self.comfyui_path, "custom_nodes", "x-flux-comfyui", "workflows")
            
            if os.path.exists(workflow_dir):
                for file in os.listdir(workflow_dir):
                    if file.endswith(".json"):
                        workflow_path = os.path.join(workflow_dir, file)
                        workflow_name = file.replace(".json", "")
                        
                        try:
                            with open(workflow_path, 'r') as f:
                                self.workflows[workflow_name] = json.load(f)
                                logger.debug(f"Loaded workflow: {workflow_name}")
                        except Exception as e:
                            logger.warning(f"Failed to load workflow {workflow_name}: {str(e)}")
            else:
                logger.warning(f"Workflow directory not found at {workflow_dir}")
        except Exception as e:
            logger.error(f"Error loading workflows: {str(e)}")
    
    def _start_comfyui_server(self):
        """Start the ComfyUI server if it's not already running."""
        try:
            # Check if server is already running
            try:
                response = requests.get(f"{self.server_url}/system_stats")
                if response.status_code == 200:
                    logger.info("ComfyUI server is already running")
                    return True
            except requests.exceptions.ConnectionError:
                logger.info("ComfyUI server is not running. Starting it...")
            
            # Start server process
            server_command = [sys.executable, os.path.join(self.comfyui_path, "main.py")]
            self.server_process = subprocess.Popen(
                server_command,
                cwd=self.comfyui_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    response = requests.get(f"{self.server_url}/system_stats")
                    if response.status_code == 200:
                        logger.info("ComfyUI server started successfully")
                        return True
                except requests.exceptions.ConnectionError:
                    if attempt < max_attempts - 1:
                        logger.debug(f"Waiting for server to start (attempt {attempt + 1}/{max_attempts})...")
                        time.sleep(1)
            
            logger.error("Failed to start ComfyUI server")
            return False
            
        except Exception as e:
            logger.error(f"Error starting ComfyUI server: {str(e)}")
            return False
    
    def _upload_image(self, image_path: str) -> str:
        """Upload an image to the ComfyUI server.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image filename on the server
        """
        try:
            # Open and resize image if needed
            img = Image.open(image_path)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=img.format or 'PNG')
            img_bytes.seek(0)
            
            # Upload to server
            filename = os.path.basename(image_path)
            files = {
                'image': (filename, img_bytes, 'image/' + (img.format.lower() if img.format else 'png'))
            }
            response = requests.post(f"{self.server_url}/upload/image", files=files)
            
            if response.status_code == 200:
                result = response.json()
                if 'name' in result:
                    logger.info(f"Uploaded image: {result['name']}")
                    return result['name']
                else:
                    logger.error(f"Upload response missing 'name': {result}")
            else:
                logger.error(f"Failed to upload image: {response.status_code} - {response.text}")
            
            return None
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            return None
    
    def _execute_workflow(self, workflow: Dict, prompt: str, init_image: str, negative_prompt: str = "", strength: float = 0.75) -> str:
        """Execute a workflow on the ComfyUI server.
        
        Args:
            workflow: The workflow definition
            prompt: Text prompt for the transformation
            init_image: Image filename on the server
            negative_prompt: Negative text prompt
            strength: Transformation strength (0.0 to 1.0)
            
        Returns:
            Path to the generated image
        """
        try:
            # Create a copy of the workflow to modify
            workflow_data = workflow.copy()
            
            # Find the relevant nodes and update their values
            for node_id, node in workflow_data.get("nodes", {}).items():
                if node.get("type") == "CLIPTextEncode":
                    if "positive" in node.get("title", "").lower():
                        node["inputs"]["text"] = prompt
                    elif "negative" in node.get("title", "").lower():
                        node["inputs"]["text"] = negative_prompt
                
                elif node.get("type") == "LoadImage":
                    node["inputs"]["image"] = init_image
                
                elif node.get("type") == "VAEDecode":
                    # This would be where to adjust parameters if needed
                    pass
            
            # Prepare the API request payload
            prompt_id = str(uuid.uuid4())
            payload = {
                "prompt": workflow_data,
                "client_id": self.client_id
            }
            
            # Send to ComfyUI server
            response = requests.post(f"{self.server_url}/prompt", json=payload)
            
            if response.status_code != 200:
                logger.error(f"Failed to execute workflow: {response.status_code} - {response.text}")
                return None
            
            # Get prompt ID from response
            prompt_id = response.json().get("prompt_id")
            if not prompt_id:
                logger.error("No prompt ID received from server")
                return None
            
            logger.info(f"Workflow execution started with prompt ID: {prompt_id}")
            
            # Poll for completion
            max_attempts = 120  # 2 minutes with 1-second intervals
            for attempt in range(max_attempts):
                try:
                    history_response = requests.get(f"{self.server_url}/history/{prompt_id}")
                    if history_response.status_code == 200:
                        history = history_response.json()
                        if history.get(prompt_id, {}).get("status", {}).get("completed", False):
                            logger.info("Workflow execution completed")
                            outputs = history.get(prompt_id, {}).get("outputs", {})
                            
                            # Find the image output
                            for node_id, node_output in outputs.items():
                                if "images" in node_output:
                                    images = node_output["images"]
                                    if images:
                                        image_data = images[0]
                                        image_filename = image_data.get("filename")
                                        return os.path.join(self.comfyui_path, "output", image_filename)
                            
                            logger.error("No image found in the output")
                            return None
                        
                    time.sleep(1)
                except requests.exceptions.ConnectionError:
                    logger.error("Connection to ComfyUI server lost")
                    return None
            
            logger.error("Workflow execution timed out")
            return None
            
        except Exception as e:
            logger.error(f"Error executing workflow: {str(e)}")
            return None
    
    def _initialize_blip2(self):
        """Initialize the BLIP-2 image captioning model if it hasn't been initialized yet."""
        if self.blip2_model is None and self.use_blip2:
            try:
                logger.info("Initializing BLIP-2 image captioning model...")
                # Determine dtype to use
                dtype = torch.float16 if self.device == "cuda" else torch.float32

                # Initialize BLIP-2 processor and model
                self.blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=dtype
                ).to(self.device)

                logger.info("BLIP-2 image captioning model initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize BLIP-2 model: {str(e)}")
                self.use_blip2 = False
                return False

        return self.blip2_model is not None if self.use_blip2 else False

    def _initialize_pipeline(self):
        """Initialize the diffusers pipeline if it hasn't been initialized yet."""
        if self.pipe is None and self.use_diffusers:
            try:
                logger.info("Initializing FLUX.1-Canny-dev pipeline...")
                # Determine dtype to use
                dtype = torch.float16
                if self.device == "cuda" and torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16

                # Get HuggingFace token from class attribute or environment
                hf_token = self.hf_token
                if hf_token:
                    logger.info("Using HuggingFace token for gated model access")
                else:
                    logger.warning("No HuggingFace token available, gated model access may fail")

                # Initialize pipeline with token for gated model access
                self.pipe = FluxControlPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-Canny-dev",
                    torch_dtype=dtype,
                    use_auth_token=hf_token
                ).to(self.device)

                logger.info("FLUX.1-Canny-dev pipeline initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize diffusers pipeline: {str(e)}")
                self.use_diffusers = False
                return False

        return self.pipe is not None if self.use_diffusers else False

    def generate_image_caption(self, image):
        """Generate a detailed caption for an image using BLIP-2.

        Args:
            image: PIL Image object

        Returns:
            Detailed caption describing the image
        """
        if not self.use_blip2:
            logger.warning("BLIP-2 image captioning is not available")
            return None

        # Initialize BLIP-2 if needed
        if not self._initialize_blip2():
            logger.error("Could not initialize BLIP-2 model")
            return None

        try:
            logger.info("Generating detailed caption for image...")

            # Process the image with BLIP-2
            inputs = self.blip2_processor(images=image, return_tensors="pt").to(self.device)

            # Generate detailed caption focusing on visual details rather than just naming the location
            with torch.no_grad():
                # Use a prompt engineering approach to guide the model toward describing visual details
                # Note: Unfortunately BLIP-2 doesn't accept a text prompt for guidance, so we'll use other parameters instead
                # We'll rely on the longer min_length, sampling parameters, and processing the output

                generated_ids = self.blip2_model.generate(
                    **inputs,
                    max_length=150,     # Allow for longer, more detailed captions
                    do_sample=True,     # Use sampling for more creative descriptions
                    temperature=0.7,    # Control randomness (higher = more diverse)
                    top_p=0.9,          # Nucleus sampling for controlled diversity
                    num_beams=5,        # Use beam search for quality
                    min_length=30       # Ensure a minimum length for descriptions
                )

            # Decode the basic caption
            base_caption = self.blip2_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            # Enhance the caption for more detailed stylization
            # If caption is too general (just naming a location), add more details
            if len(base_caption.split()) < 10:
                # For short captions, add more descriptive context
                if "london" in base_caption.lower():
                    base_caption += (
                        ". The scene features distinctive architectural elements of historic London, "
                        "with imposing stone structures, ornate details, varied textures, "
                        "and a moody atmosphere with dramatic lighting."
                    )
                elif "cathedral" in base_caption.lower() or "church" in base_caption.lower():
                    base_caption += (
                        ". The scene showcases elaborate gothic architecture with intricate stone carvings, "
                        "tall spires, arched windows, textured stonework, and dramatic lighting "
                        "highlighting the ornate details and imposing structure."
                    )
                elif "bridge" in base_caption.lower():
                    base_caption += (
                        ". The structure features a complex arrangement of metal and stone elements, "
                        "with distinctive industrial patterns, textured surfaces, "
                        "and atmospheric lighting conditions creating a dramatic scene."
                    )

            enhanced_caption = base_caption

            logger.info(f"Generated caption: {enhanced_caption}")
            return enhanced_caption

        except Exception as e:
            logger.error(f"Error generating image caption: {str(e)}")
            return None
    
    def get_material_type(self, image_path):
        """Determine the material type from the image filename.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Material type (brick, stone, wood, metal, glass, or general)
        """
        filename = os.path.basename(image_path).lower()
        
        if any(m in filename for m in ['brick', 'masonry', 'wall']):
            return 'brick'
        elif any(m in filename for m in ['cobble', 'stone', 'concrete', 'pavement']):
            return 'stone'
        elif any(m in filename for m in ['wood', 'timber', 'plank']):
            return 'wood'
        elif any(m in filename for m in ['metal', 'iron', 'steel', 'brass', 'copper']):
            return 'metal'
        elif any(m in filename for m in ['glass', 'window', 'pane']):
            return 'glass'
        else:
            return 'general'

    def transform_image(self,
                       image_path: str,
                       output_path: str,
                       prompt: str = None,
                       material_type: str = None,
                       use_image_captioning: bool = True,
                       negative_prompt: str = "photorealistic, modern, contemporary, bright colors, clear sky",
                       strength: float = 0.75,
                       canny_low_threshold: int = 50,
                       canny_high_threshold: int = 200,
                       guidance_scale: float = 30.0,
                       num_inference_steps: int = 50,
                       width: int = 1024,
                       height: int = 1024) -> str:
        """Transform an image using FLUX.1-Canny-dev or ComfyUI's Flux backend.

        Args:
            image_path: Path to the input image
            output_path: Path to save the transformed image
            prompt: The prompt to guide the image transformation (if None, will use material-specific prompt)
            material_type: The material type (brick, stone, wood, metal, glass, or general)
            use_image_captioning: Whether to use BLIP-2 for automatic image captioning
            negative_prompt: Negative prompt to guide what to avoid in the image
            strength: Strength of the transformation (0.0 to 1.0)
            canny_low_threshold: Low threshold for Canny edge detection
            canny_high_threshold: High threshold for Canny edge detection
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of inference steps
            width: Output image width
            height: Output image height

        Returns:
            Path to the transformed image
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Determine material type if not provided
            if material_type is None:
                material_type = self.get_material_type(image_path)
                logger.info(f"Detected material type: {material_type}")

            # Load the input image first (needed for both captioning and transformation)
            logger.info(f"Loading input image: {image_path}")
            input_image = load_image(image_path)

            # Generate image caption with BLIP-2 if enabled
            generated_caption = None
            if use_image_captioning and self.use_blip2:
                generated_caption = self.generate_image_caption(input_image)
                if generated_caption:
                    logger.info(f"BLIP-2 generated caption: {generated_caption}")

            # Use material-specific prompt if no prompt provided, with caption if available
            final_prompt = prompt
            if final_prompt is None:
                base_prompt = self.ARCANUM_PROMPTS.get(material_type, self.ARCANUM_PROMPTS['general'])

                # Insert the generated caption if available
                if generated_caption:
                    # Add the detailed caption to the material-specific prompt
                    detail_section = f"This is a detailed {material_type} texture with the following characteristics: {generated_caption}."
                    # Insert the caption after the first paragraph of the base prompt
                    parts = base_prompt.split("\n\n", 1)
                    if len(parts) > 1:
                        final_prompt = f"{parts[0]}\n\n{detail_section}\n\n{parts[1]}"
                    else:
                        final_prompt = f"{base_prompt}\n\n{detail_section}"
                else:
                    final_prompt = base_prompt

                logger.info(f"Using {'BLIP-2 enhanced ' if generated_caption else ''}{material_type}-specific Arcanum style prompt")

            # Try using diffusers first
            if self.use_diffusers:
                # Initialize pipeline if needed
                if not self._initialize_pipeline():
                    logger.warning("Could not initialize diffusers pipeline, falling back to ComfyUI")
                else:
                    try:
                        # Apply edge detection
                        logger.info("Applying Canny edge detection")
                        control_image = self.processor(
                            input_image,
                            low_threshold=canny_low_threshold,
                            high_threshold=canny_high_threshold,
                            detect_resolution=1024,
                            image_resolution=1024
                        )

                        # Run inference
                        logger.info("Running FLUX.1-Canny-dev inference...")
                        with torch.no_grad():
                            output_image = self.pipe(
                                prompt=final_prompt,
                                negative_prompt=negative_prompt,
                                control_image=control_image,
                                width=width,
                                height=height,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                            ).images[0]

                        # Save output image
                        output_image.save(output_path)
                        logger.info(f"Saved transformed image to {output_path}")
                        return output_path
                    except Exception as diffusers_error:
                        logger.error(f"Error using diffusers: {str(diffusers_error)}")
                        logger.warning("Falling back to ComfyUI integration")
            
            # Fall back to ComfyUI if diffusers failed or isn't available
            # Ensure ComfyUI server is running
            if not hasattr(self, '_start_comfyui_server') or not self._start_comfyui_server():
                logger.error("Failed to start ComfyUI server")
                return None
            
            # Upload the image
            server_image = self._upload_image(image_path)
            if not server_image:
                logger.error("Failed to upload image")
                return None
            
            # Select a workflow (for now just use the main img2img workflow if available)
            workflow = None
            if "flux-controlnet-canny-v3-workflow" in self.workflows:
                workflow = self.workflows["flux-controlnet-canny-v3-workflow"]
            elif list(self.workflows.keys()):
                # Use the first available workflow as fallback
                workflow_name = list(self.workflows.keys())[0]
                workflow = self.workflows[workflow_name]
                logger.info(f"Using fallback workflow: {workflow_name}")
            else:
                logger.error("No workflows available")
                return None
            
            # Execute the workflow
            result_path = self._execute_workflow(
                workflow, 
                prompt, 
                server_image, 
                negative_prompt, 
                strength
            )
            
            if not result_path:
                logger.error("Workflow execution failed")
                return None
            
            # Copy the result to the output path
            if os.path.exists(result_path):
                img = Image.open(result_path)
                img.save(output_path)
                logger.info(f"Saved transformed image to {output_path}")
                return output_path
            else:
                logger.error(f"Result image not found at {result_path}")
                return None
            
        except Exception as e:
            logger.error(f"Error transforming image: {str(e)}")
            return None
    
    def batch_transform_images(self,
                              image_paths: List[str],
                              output_dir: str,
                              prompt: str = None,
                              use_image_captioning: bool = True,
                              negative_prompt: str = "photorealistic, modern, contemporary, bright colors",
                              strength: float = 0.75,
                              canny_low_threshold: int = 50,
                              canny_high_threshold: int = 200,
                              guidance_scale: float = 30.0,
                              num_inference_steps: int = 50,
                              width: int = 1024,
                              height: int = 1024) -> List[str]:
        """Transform multiple images in sequence.

        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save transformed images
            prompt: The prompt to guide the image transformation (if None, will use material-specific prompts)
            use_image_captioning: Whether to use BLIP-2 for automatic image captioning
            negative_prompt: Negative prompt to guide what to avoid in the image
            strength: Strength of the transformation (0.0 to 1.0)
            canny_low_threshold: Low threshold for Canny edge detection
            canny_high_threshold: High threshold for Canny edge detection
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of inference steps
            width: Output image width
            height: Output image height

        Returns:
            List of paths to transformed images
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        # Initialize pipeline if using diffusers
        if self.use_diffusers:
            self._initialize_pipeline()
        elif hasattr(self, '_start_comfyui_server'):
            # Ensure ComfyUI server is running
            if not self._start_comfyui_server():
                logger.error("Failed to start ComfyUI server")
                return []
        
        # Process images one by one
        for i, img_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
                
                # Generate output path
                img_filename = os.path.basename(img_path)
                output_path = os.path.join(output_dir, f"arcanum_{img_filename}")
                
                # Determine material type based on filename
                material_type = self.get_material_type(img_path)
                
                # Get material-specific prompt if generic prompt provided
                img_prompt = prompt
                if img_prompt is None:
                    img_prompt = self.ARCANUM_PROMPTS.get(material_type, self.ARCANUM_PROMPTS['general'])
                
                # Transform the image
                result = self.transform_image(
                    img_path,
                    output_path,
                    prompt=img_prompt,
                    material_type=material_type,
                    use_image_captioning=use_image_captioning,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    canny_low_threshold=canny_low_threshold,
                    canny_high_threshold=canny_high_threshold,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height
                )
                
                if result:
                    output_paths.append(result)
                    logger.info(f"Successfully transformed {material_type} image {i+1}/{len(image_paths)}")
                else:
                    logger.error(f"Failed to transform image {i+1}/{len(image_paths)}")
            
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")
        
        return output_paths
    
    def __del__(self):
        """Clean up resources."""
        try:
            # Clean up BLIP-2 model
            if hasattr(self, 'blip2_model') and self.blip2_model is not None:
                logger.info("Cleaning up BLIP-2 model...")
                del self.blip2_model
                del self.blip2_processor

            # Clean up diffusers pipeline
            if hasattr(self, 'pipe') and self.pipe is not None:
                logger.info("Cleaning up diffusers pipeline...")
                del self.pipe
                del self.processor

            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clean up ComfyUI server if it was started
            if hasattr(self, 'server_process') and self.server_process:
                logger.info("Shutting down ComfyUI server...")
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=10)
                    logger.info("ComfyUI server terminated.")
                except subprocess.TimeoutExpired:
                    logger.warning("Timeout while waiting for ComfyUI server to terminate.")
                    self.server_process.kill()
                    logger.info("ComfyUI server killed.")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")