#!/usr/bin/env python3
"""
Arcanum Web Interface
--------------------
This module provides a web-based interface for the Arcanum city generator,
allowing users to interact with the system through a browser.
"""

import os
import sys
import json
import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from functools import wraps

# Flask imports
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import Arcanum modules
from modules.osm import bbox_downloader, grid_downloader, building_processor
from modules.comfyui import automation
from modules.styles import style_manager
from modules.parallel import task_manager
from modules.visualization import preview
from modules.exporters import export_manager
from modules.ai import building_generator
from modules.texture_atlas_manager import TextureAtlasManager

# Import integration tools
from integration_tools import texture_projection, unity_material_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                         ".arcanum", "logs", "web_interface.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))
app.secret_key = os.environ.get('ARCANUM_SECRET_KEY', str(uuid.uuid4()))
socketio = SocketIO(app, cors_allowed_origins="*")

# Global job store to track running tasks
job_store = {}

# Default configuration
default_config = {
    "output_directory": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                     "arcanum_output"),
    "coordinate_system": "EPSG:4326",
    "max_concurrent_tasks": 4,
    "preview_quality": "medium",
    "default_style": "arcanum_victorian",
    "texture_directory": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    "textures"),
    "enable_texture_projection": True,
    "atlas_size": [4096, 4096],
    "building_texture_size": [1024, 1024]
}

# Load configuration from .arcanum/config if it exists
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           ".arcanum", "config", "web_config.json")
config = default_config
if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")

# Ensure output directory exists
os.makedirs(config["output_directory"], exist_ok=True)

# Decorator to check API authentication
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # If API key is configured, check it
        if "api_key" in config:
            api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            if not api_key or api_key != config["api_key"]:
                return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Initialize texture atlas manager
texture_atlas_manager = TextureAtlasManager(
    os.path.join(config["output_directory"], "atlases"),
    atlas_size=tuple(config.get("atlas_size", (4096, 4096))),
    building_texture_size=tuple(config.get("building_texture_size", (1024, 1024)))
)

# Main routes
@app.route('/')
def index():
    """Render the main application interface."""
    return render_template('index.html')

@app.route('/api/buildings', methods=['GET'])
@require_api_key
def get_buildings():
    """Get list of buildings in the project."""
    try:
        # Look for buildings metadata file
        metadata_path = os.path.join(config["output_directory"], "building_data", "metadata", "buildings_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                buildings_data = json.load(f)
            return jsonify({"buildings": buildings_data})
        else:
            return jsonify({"buildings": {}, "message": "No buildings found"})
    except Exception as e:
        logger.error(f"Error getting buildings: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/textures', methods=['GET'])
@require_api_key
def get_textures():
    """Get list of available textures."""
    try:
        textures_dir = config.get("texture_directory", os.path.join(config["output_directory"], "processed_data", "textures"))
        textures = []

        if os.path.exists(textures_dir):
            for filename in os.listdir(textures_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tga")):
                    texture_path = os.path.join(textures_dir, filename)
                    texture_info = {
                        "id": filename,
                        "name": filename,
                        "path": texture_path,
                        "url": f"/api/textures/{filename}"
                    }
                    textures.append(texture_info)

        return jsonify({"textures": textures})
    except Exception as e:
        logger.error(f"Error getting textures: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/textures/<filename>', methods=['GET'])
@require_api_key
def get_texture(filename):
    """Get a specific texture file."""
    textures_dir = config.get("texture_directory", os.path.join(config["output_directory"], "processed_data", "textures"))
    return send_from_directory(textures_dir, filename)

@app.route('/api/textures/assign', methods=['POST'])
@require_api_key
def assign_texture():
    """Assign a texture to a building."""
    try:
        data = request.json
        building_id = data.get("building_id")
        texture_id = data.get("texture_id")

        if not building_id or not texture_id:
            return jsonify({"error": "Missing building_id or texture_id"}), 400

        # Find texture file
        textures_dir = config.get("texture_directory", os.path.join(config["output_directory"], "processed_data", "textures"))
        texture_path = os.path.join(textures_dir, texture_id)

        if not os.path.exists(texture_path):
            return jsonify({"error": f"Texture not found: {texture_id}"}), 404

        # Look for building metadata
        metadata_path = os.path.join(config["output_directory"], "building_data", "metadata", "buildings_metadata.json")
        building_metadata = {}

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                buildings_data = json.load(f)
                building_metadata = buildings_data.get(building_id, {})

        # Assign texture to building
        result = texture_atlas_manager.assign_texture_to_building(
            building_id,
            texture_path,
            building_metadata
        )

        if result.get("success", False):
            return jsonify({
                "success": True,
                "building_id": building_id,
                "texture_id": texture_id,
                "uv_mapping": result.get("uv_mapping", {})
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get("error", "Unknown error")
            }), 400
    except Exception as e:
        logger.error(f"Error assigning texture: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/atlases', methods=['GET'])
@require_api_key
def get_atlases():
    """Get list of texture atlases."""
    try:
        atlases_dir = os.path.join(config["output_directory"], "atlases")
        atlases = []

        if os.path.exists(atlases_dir):
            for filename in os.listdir(atlases_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    atlas_path = os.path.join(atlases_dir, filename)
                    atlas_info = {
                        "id": filename,
                        "name": filename,
                        "path": atlas_path,
                        "url": f"/api/atlases/{filename}"
                    }
                    atlases.append(atlas_info)

        return jsonify({"atlases": atlases})
    except Exception as e:
        logger.error(f"Error getting atlases: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/atlases/<filename>', methods=['GET'])
@require_api_key
def get_atlas(filename):
    """Get a specific atlas file."""
    atlases_dir = os.path.join(config["output_directory"], "atlases")
    return send_from_directory(atlases_dir, filename)

@app.route('/api/preview/atlas', methods=['GET'])
@require_api_key
def preview_atlas():
    """Generate a preview of a texture atlas."""
    try:
        atlas_id = request.args.get("atlas_id")
        if not atlas_id:
            return jsonify({"error": "Missing atlas_id"}), 400

        # Find atlas file
        atlases_dir = os.path.join(config["output_directory"], "atlases")
        atlas_path = os.path.join(atlases_dir, atlas_id)

        if not os.path.exists(atlas_path):
            return jsonify({"error": f"Atlas not found: {atlas_id}"}), 404

        # Get UV mapping data
        mapping_data = texture_atlas_manager.get_atlas_mapping(atlas_id)

        # Generate preview
        preview_dir = os.path.join(config["output_directory"], "visualizations")
        os.makedirs(preview_dir, exist_ok=True)

        preview_path = os.path.join(preview_dir, f"preview_{atlas_id}")

        result = preview.generate_atlas_preview(
            atlas_path,
            mapping_data,
            preview_path,
            show_grid=True
        )

        if result.get("success", False):
            return jsonify({
                "success": True,
                "preview_url": f"/api/preview/image?path={os.path.basename(result['preview_path'])}",
                "building_count": result.get("building_count", 0)
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get("error", "Unknown error")
            }), 400
    except Exception as e:
        logger.error(f"Error generating atlas preview: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/preview/building', methods=['GET'])
@require_api_key
def preview_building():
    """Generate a preview of a building with its texture."""
    try:
        building_id = request.args.get("building_id")
        if not building_id:
            return jsonify({"error": "Missing building_id"}), 400

        # Find building model
        models_dir = os.path.join(config["output_directory"], "models")
        model_path = os.path.join(models_dir, f"{building_id}.obj")

        if not os.path.exists(model_path):
            return jsonify({"error": f"Building model not found: {building_id}"}), 404

        # Get texture for this building
        building_texture = texture_atlas_manager.get_building_texture(building_id)

        if not building_texture or not os.path.exists(building_texture.get("texture_path", "")):
            return jsonify({"error": f"No texture assigned to building: {building_id}"}), 400

        # Generate preview
        preview_dir = os.path.join(config["output_directory"], "visualizations")
        os.makedirs(preview_dir, exist_ok=True)

        result = preview.preview_texture_mapping(
            model_path,
            building_texture["texture_path"],
            os.path.join(preview_dir, f"texture_mapping_{building_id}.png")
        )

        if result.get("success", False):
            return jsonify({
                "success": True,
                "preview_url": f"/api/preview/image?path={os.path.basename(result['preview_path'])}"
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get("error", "Unknown error")
            }), 400
    except Exception as e:
        logger.error(f"Error generating building preview: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/preview/image', methods=['GET'])
@require_api_key
def get_preview_image():
    """Get a specific preview image."""
    try:
        path = request.args.get("path")
        if not path:
            return jsonify({"error": "Missing path"}), 400

        preview_dir = os.path.join(config["output_directory"], "visualizations")
        return send_from_directory(preview_dir, path)
    except Exception as e:
        logger.error(f"Error getting preview image: {str(e)}")
        return jsonify({"error": str(e)}), 500
    # Get list of available styles
    available_styles = style_manager.get_available_styles()
    
    # Get list of export formats
    export_formats = export_manager.get_available_formats()
    
    return render_template(
        'index.html', 
        styles=available_styles,
        export_formats=export_formats,
        config=config
    )

@app.route('/api/health')
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})

@app.route('/api/styles')
def list_styles():
    """List all available styles."""
    return jsonify(style_manager.get_available_styles())

@app.route('/api/styles/<style_id>')
def get_style(style_id):
    """Get details for a specific style."""
    style = style_manager.get_style(style_id)
    if style:
        return jsonify(style)
    return jsonify({"error": "Style not found"}), 404

@app.route('/api/jobs', methods=['POST'])
@require_api_key
def create_job():
    """Create a new generation job."""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ["bounds", "style_id"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Create a unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job directory
        job_dir = os.path.join(config["output_directory"], job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Initialize job configuration
        job_config = {
            "id": job_id,
            "bounds": data["bounds"],
            "style_id": data["style_id"],
            "cell_size": data.get("cell_size", 100),
            "status": "pending",
            "created_at": time.time(),
            "progress": 0,
            "output_dir": job_dir,
            "export_format": data.get("export_format", "unity"),
            "notifications": data.get("notifications", {})
        }
        
        # Save job configuration
        with open(os.path.join(job_dir, "job_config.json"), 'w') as f:
            json.dump(job_config, f, indent=2)
        
        # Add job to job store
        job_store[job_id] = job_config
        
        # Start job processing thread
        thread = threading.Thread(target=process_job, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({"job_id": job_id, "status": "pending"})
    
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}")
        return jsonify({"error": f"Failed to create job: {str(e)}"}), 500

@app.route('/api/jobs/<job_id>')
@require_api_key
def get_job_status(job_id):
    """Get status of a specific job."""
    if job_id in job_store:
        return jsonify(job_store[job_id])
    
    # Try to load from filesystem if not in memory
    job_config_path = os.path.join(config["output_directory"], job_id, "job_config.json")
    if os.path.exists(job_config_path):
        try:
            with open(job_config_path, 'r') as f:
                job_config = json.load(f)
                job_store[job_id] = job_config
                return jsonify(job_config)
        except Exception as e:
            logger.error(f"Error loading job config: {str(e)}")
    
    return jsonify({"error": "Job not found"}), 404

@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
@require_api_key
def cancel_job(job_id):
    """Cancel a running job."""
    if job_id not in job_store:
        return jsonify({"error": "Job not found"}), 404
    
    job = job_store[job_id]
    if job["status"] in ["completed", "failed", "cancelled"]:
        return jsonify({"error": f"Cannot cancel job with status: {job['status']}"}), 400
    
    # Update job status
    job["status"] = "cancelled"
    job_store[job_id] = job
    
    # Save updated config to disk
    try:
        with open(os.path.join(config["output_directory"], job_id, "job_config.json"), 'w') as f:
            json.dump(job, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving job config: {str(e)}")
    
    # Signal task manager to cancel the job
    task_manager.cancel_job(job_id)
    
    return jsonify({"status": "cancelled"})

@app.route('/api/jobs/<job_id>/results', methods=['GET'])
@require_api_key
def get_job_results(job_id):
    """Get results of a completed job."""
    if job_id not in job_store:
        return jsonify({"error": "Job not found"}), 404
    
    job = job_store[job_id]
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed yet"}), 400
    
    # Collect result files
    result_dir = os.path.join(config["output_directory"], job_id)
    result_files = []
    
    for root, _, files in os.walk(result_dir):
        for filename in files:
            if filename.endswith(('.jpg', '.png', '.obj', '.fbx', '.glb', '.gltf', '.unity', '.zip')):
                rel_path = os.path.relpath(os.path.join(root, filename), result_dir)
                result_files.append({
                    "filename": filename,
                    "path": rel_path,
                    "url": f"/api/jobs/{job_id}/files/{rel_path}",
                    "size": os.path.getsize(os.path.join(root, filename))
                })
    
    return jsonify({
        "job_id": job_id,
        "status": "completed",
        "files": result_files
    })

@app.route('/api/jobs/<job_id>/files/<path:filename>')
@require_api_key
def get_job_file(job_id, filename):
    """Download a specific file from job results."""
    if job_id not in job_store:
        return jsonify({"error": "Job not found"}), 404
    
    job_dir = os.path.join(config["output_directory"], job_id)
    return send_from_directory(job_dir, filename, as_attachment=True)

@app.route('/api/preview', methods=['POST'])
@require_api_key
def generate_preview():
    """Generate a preview of the stylized output."""
    try:
        # Check for file upload
        if 'image' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save uploaded file to temporary location
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(config["output_directory"], "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        input_path = os.path.join(temp_dir, filename)
        file.save(input_path)
        
        # Get style parameters
        style_id = request.form.get('style_id', config['default_style'])
        strength = float(request.form.get('strength', 0.75))
        
        # Generate preview
        preview_image = preview.generate_preview(
            input_path, 
            style_id=style_id,
            strength=strength,
            quality=config["preview_quality"]
        )
        
        # Return preview image info
        preview_path = f"/api/previews/{os.path.basename(preview_image)}"
        return jsonify({
            "preview_url": preview_path,
            "style_id": style_id,
            "strength": strength
        })
        
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}")
        return jsonify({"error": f"Failed to generate preview: {str(e)}"}), 500

@app.route('/api/previews/<filename>')
def get_preview(filename):
    """Serve a generated preview image."""
    preview_dir = os.path.join(config["output_directory"], "previews")
    return send_from_directory(preview_dir, filename)

# Socket.IO event handlers for real-time updates
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")

@socketio.on('subscribe')
def handle_subscribe(data):
    """Subscribe to job updates."""
    job_id = data.get('job_id')
    if job_id:
        logger.info(f"Client {request.sid} subscribed to job {job_id}")
        
        # Send current job status immediately
        if job_id in job_store:
            emit('job_update', job_store[job_id])

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

# Background job processing function
def process_job(job_id):
    """Process a generation job in the background."""
    if job_id not in job_store:
        logger.error(f"Job {job_id} not found in job store")
        return
    
    job = job_store[job_id]
    job_dir = os.path.join(config["output_directory"], job_id)
    
    try:
        # Update job status
        job["status"] = "processing"
        job["started_at"] = time.time()
        job["progress"] = 0
        job_store[job_id] = job
        
        # Save updated job config
        with open(os.path.join(job_dir, "job_config.json"), 'w') as f:
            json.dump(job, f, indent=2)
        
        # Emit job status update
        socketio.emit('job_update', job, room=job_id)
        
        # Step 1: Download OSM data
        update_job_progress(job_id, 5, "Downloading OpenStreetMap data")
        osm_result = bbox_downloader.download_osm_data(
            bounds=job["bounds"],
            output_dir=os.path.join(job_dir, "raw_data"),
            coordinate_system=config["coordinate_system"]
        )
        
        if not osm_result.get("success", False):
            raise Exception(f"Failed to download OSM data: {osm_result.get('error', 'Unknown error')}")
        
        # Step 2: Process buildings
        update_job_progress(job_id, 20, "Processing building data")
        buildings_result = building_generator.process_buildings(
            osm_path=osm_result["buildings_path"],
            output_dir=os.path.join(job_dir, "processed_data"),
            style_id=job["style_id"]
        )
        
        # Step 3: Apply Arcanum style transformation
        update_job_progress(job_id, 40, "Applying style transformation")
        style_result = style_manager.apply_style(
            input_dir=os.path.join(job_dir, "raw_data"),
            output_dir=os.path.join(job_dir, "styled_data"),
            style_id=job["style_id"],
            job_id=job_id
        )
        
        # Step 4: Generate 3D models
        update_job_progress(job_id, 60, "Generating 3D models")
        models_result = building_generator.generate_models(
            buildings_data=buildings_result["buildings_data"],
            styled_textures=style_result["output_dir"],
            output_dir=os.path.join(job_dir, "3d_models")
        )
        
        # Step 5: Export in requested format
        update_job_progress(job_id, 80, "Exporting to requested format")
        export_result = export_manager.export_project(
            input_dir=os.path.join(job_dir, "3d_models"),
            output_dir=os.path.join(job_dir, "exports"),
            format=job["export_format"],
            job_config=job
        )
        
        # Update job status to completed
        update_job_progress(job_id, 100, "Job completed successfully")
        job["status"] = "completed"
        job["completed_at"] = time.time()
        job["duration"] = job["completed_at"] - job["started_at"]
        job["result"] = {
            "export_path": export_result["output_path"],
            "export_format": job["export_format"],
            "model_count": models_result["model_count"],
            "processing_stats": {
                "buildings_processed": buildings_result["building_count"],
                "textures_generated": style_result["texture_count"],
                "total_duration_seconds": job["duration"]
            }
        }
        
        # Save final job config
        job_store[job_id] = job
        with open(os.path.join(job_dir, "job_config.json"), 'w') as f:
            json.dump(job, f, indent=2)
        
        # Emit final job update
        socketio.emit('job_update', job, room=job_id)
        socketio.emit('job_completed', {"job_id": job_id}, room=job_id)
        
        # Handle notifications
        if "notifications" in job and isinstance(job["notifications"], dict):
            if "email" in job["notifications"] and job["notifications"]["email"]:
                # TODO: Implement email notifications
                pass
            
            if "webhook" in job["notifications"] and job["notifications"]["webhook"]:
                webhook_url = job["notifications"]["webhook"]
                try:
                    import requests
                    requests.post(
                        webhook_url,
                        json={
                            "job_id": job_id,
                            "status": "completed",
                            "result": job["result"]
                        },
                        timeout=10
                    )
                except Exception as e:
                    logger.error(f"Error sending webhook notification: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        
        # Update job status to failed
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = time.time()
        if "started_at" in job:
            job["duration"] = job["completed_at"] - job["started_at"]
        
        # Save failed job config
        job_store[job_id] = job
        with open(os.path.join(job_dir, "job_config.json"), 'w') as f:
            json.dump(job, f, indent=2)
        
        # Emit job failure update
        socketio.emit('job_update', job, room=job_id)
        socketio.emit('job_failed', {
            "job_id": job_id,
            "error": str(e)
        }, room=job_id)

def update_job_progress(job_id, progress, status_message):
    """Update job progress and emit update event."""
    if job_id not in job_store:
        logger.error(f"Job {job_id} not found in job store")
        return
    
    job = job_store[job_id]
    job["progress"] = progress
    job["status_message"] = status_message
    job_store[job_id] = job
    
    # Save updated job config
    try:
        with open(os.path.join(config["output_directory"], job_id, "job_config.json"), 'w') as f:
            json.dump(job, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving job progress: {str(e)}")
    
    # Emit update event
    socketio.emit('job_update', job, room=job_id)
    socketio.emit('job_progress', {
        "job_id": job_id,
        "progress": progress,
        "status_message": status_message
    }, room=job_id)

def run_server(host="0.0.0.0", port=5000, debug=False):
    """Run the Flask server."""
    socketio.run(app, host=host, port=port, debug=debug)

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run the Arcanum Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Run the server
    run_server(host=args.host, port=args.port, debug=args.debug)