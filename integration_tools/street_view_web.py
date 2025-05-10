#!/usr/bin/env python3
"""
Street View Web Interface
-------------------------
This module provides a web interface for Street View collection along roads.
It allows users to visualize road networks, select areas for Street View collection,
monitor collection progress, and verify coverage.
"""

import os
import sys
import json
import logging
import time
import argparse
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import base64
from io import BytesIO

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import Flask and related libraries
try:
    from flask import Flask, request, jsonify, render_template, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with 'pip install flask flask-cors'")

# Import other required modules
try:
    from integration_tools import street_view_integration
    from modules.integration import road_network_integration
    from modules.integration import coverage_verification
    from modules.visualization.preview import visualize_road_network, visualize_street_view_coverage
    
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Required modules not available: {str(e)}")

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables
app = None
static_folder = os.path.join(os.path.dirname(__file__), "static")
collection_tasks = {}
verification_tasks = {}

class CollectionTask:
    """Class to track and manage a Street View collection task."""
    
    def __init__(self, task_id: str, road_network_path: str, output_dir: str, 
                 api_key: str = None, sampling_interval: float = 50.0,
                 max_points: Optional[int] = None, panorama: bool = True,
                 max_search_radius: int = 1000, max_workers: int = 4):
        """Initialize a collection task.
        
        Args:
            task_id: Unique identifier for this task
            road_network_path: Path to the road network file
            output_dir: Directory to save Street View images
            api_key: Google Maps API key
            sampling_interval: Distance between sampling points in meters
            max_points: Maximum number of points to process
            panorama: Whether to capture full panoramas
            max_search_radius: Maximum search radius in meters
            max_workers: Maximum number of worker threads
        """
        self.task_id = task_id
        self.road_network_path = road_network_path
        self.output_dir = output_dir
        self.api_key = api_key
        self.sampling_interval = sampling_interval
        self.max_points = max_points
        self.panorama = panorama
        self.max_search_radius = max_search_radius
        self.max_workers = max_workers
        
        self.status = "created"
        self.progress = 0
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.thread = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self):
        """Run the collection task in a separate thread."""
        if self.status == "running":
            return {"error": "Task already running"}
        
        def _run_task():
            try:
                self.status = "running"
                self.start_time = time.time()
                
                # Initialize road network integration
                rn_integration = road_network_integration.RoadNetworkIntegration(
                    self.output_dir,
                    api_key=self.api_key
                )
                
                # Load road network
                load_result = rn_integration.load_road_network(self.road_network_path)
                
                if not load_result:
                    self.status = "failed"
                    self.error = f"Failed to load road network from {self.road_network_path}"
                    self.end_time = time.time()
                    return
                
                # Fetch Street View along roads
                sv_result = rn_integration.fetch_street_view_along_roads(
                    sampling_interval=self.sampling_interval,
                    max_points=self.max_points,
                    panorama=self.panorama,
                    max_search_radius=self.max_search_radius,
                    max_workers=self.max_workers,
                    progress_callback=self._update_progress
                )
                
                self.end_time = time.time()
                
                if sv_result.get("success", False):
                    self.status = "completed"
                    self.result = sv_result
                    self.progress = 100
                else:
                    self.status = "failed"
                    self.error = sv_result.get("error", "Unknown error")
                
            except Exception as e:
                self.status = "failed"
                self.error = str(e)
                self.end_time = time.time()
                logger.error(f"Collection task {self.task_id} failed: {str(e)}")
        
        # Start thread
        self.thread = threading.Thread(target=_run_task)
        self.thread.daemon = True
        self.thread.start()
        
        return {"status": "started", "task_id": self.task_id}
    
    def _update_progress(self, current: int, total: int):
        """Update progress callback for the collection task.
        
        Args:
            current: Current number of points processed
            total: Total number of points to process
        """
        self.progress = int(current / total * 100) if total > 0 else 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the task.
        
        Returns:
            Dictionary with task status information
        """
        elapsed = None
        if self.start_time:
            if self.end_time:
                elapsed = self.end_time - self.start_time
            else:
                elapsed = time.time() - self.start_time
        
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "elapsed_seconds": elapsed,
            "result": self.result,
            "error": self.error
        }

class VerificationTask:
    """Class to track and manage a coverage verification task."""
    
    def __init__(self, task_id: str, road_network_path: str, coverage_data_path: str, 
                 output_dir: str, coverage_threshold: float = 50.0,
                 max_gap_distance: float = 30.0, parallel: bool = True,
                 max_workers: int = 4):
        """Initialize a verification task.
        
        Args:
            task_id: Unique identifier for this task
            road_network_path: Path to the road network file
            coverage_data_path: Path to coverage data file
            output_dir: Directory to save verification results
            coverage_threshold: Minimum coverage percentage required
            max_gap_distance: Maximum allowed gap between coverage points
            parallel: Whether to process edges in parallel
            max_workers: Maximum number of worker threads
        """
        self.task_id = task_id
        self.road_network_path = road_network_path
        self.coverage_data_path = coverage_data_path
        self.output_dir = output_dir
        self.coverage_threshold = coverage_threshold
        self.max_gap_distance = max_gap_distance
        self.parallel = parallel
        self.max_workers = max_workers
        
        self.status = "created"
        self.progress = 0
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.thread = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self):
        """Run the verification task in a separate thread."""
        if self.status == "running":
            return {"error": "Task already running"}
        
        def _run_task():
            try:
                self.status = "running"
                self.start_time = time.time()
                
                # Load road network data
                with open(self.road_network_path, 'r') as f:
                    road_network_data = json.load(f)
                
                # Load coverage data
                with open(self.coverage_data_path, 'r') as f:
                    coverage_data = json.load(f)
                
                # Create verifier
                verifier = coverage_verification.CoverageVerifier(
                    coverage_threshold=self.coverage_threshold,
                    max_gap_distance=self.max_gap_distance
                )
                
                # Verify coverage
                verification_results = verifier.verify_coverage(
                    road_network_data,
                    coverage_data,
                    parallel=self.parallel,
                    max_workers=self.max_workers
                )
                
                # Generate report
                report_path = verifier.generate_coverage_report(
                    verification_results,
                    os.path.join(self.output_dir, f"coverage_report_{self.task_id}.txt")
                )
                
                # Generate visualization
                visualization_result = verifier.visualize_coverage(
                    road_network_data,
                    coverage_data,
                    verification_results,
                    os.path.join(self.output_dir, f"coverage_visualization_{self.task_id}.png")
                )
                
                self.end_time = time.time()
                self.status = "completed"
                self.progress = 100
                
                # Combine results
                self.result = {
                    "verification_results": verification_results,
                    "report_path": report_path,
                    "visualization_path": visualization_result.get("visualization_path") if visualization_result.get("success", False) else None
                }
                
            except Exception as e:
                self.status = "failed"
                self.error = str(e)
                self.end_time = time.time()
                logger.error(f"Verification task {self.task_id} failed: {str(e)}")
        
        # Start thread
        self.thread = threading.Thread(target=_run_task)
        self.thread.daemon = True
        self.thread.start()
        
        return {"status": "started", "task_id": self.task_id}
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the task.
        
        Returns:
            Dictionary with task status information
        """
        elapsed = None
        if self.start_time:
            if self.end_time:
                elapsed = self.end_time - self.start_time
            else:
                elapsed = time.time() - self.start_time
        
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "elapsed_seconds": elapsed,
            "result": self.result,
            "error": self.error
        }

def create_app(config=None):
    """Create and configure the Flask application.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Flask application instance
    """
    global app
    
    if not FLASK_AVAILABLE:
        logger.error("Flask is not available. Please install Flask and Flask-CORS.")
        return None
    
    # Check for required modules
    if not MODULES_AVAILABLE:
        logger.error("Required modules are not available.")
        return None
    
    # Create Flask app
    app = Flask(__name__, static_folder=static_folder)
    CORS(app)
    
    # Apply configuration
    if config:
        app.config.update(config)
    
    # Create static directory if it doesn't exist
    os.makedirs(static_folder, exist_ok=True)
    
    # Create default HTML template
    create_default_template()
    
    # Define routes
    
    @app.route('/')
    def index():
        """Serve the main page."""
        return render_template('index.html')
    
    @app.route('/api/status')
    def api_status():
        """Return API status."""
        return jsonify({
            "status": "online",
            "version": "1.0.0",
            "modules_available": MODULES_AVAILABLE
        })
    
    @app.route('/api/tasks')
    def list_tasks():
        """List all tasks."""
        collection_statuses = {task_id: task.get_status() for task_id, task in collection_tasks.items()}
        verification_statuses = {task_id: task.get_status() for task_id, task in verification_tasks.items()}
        
        return jsonify({
            "collection_tasks": collection_statuses,
            "verification_tasks": verification_statuses
        })
    
    @app.route('/api/task/<task_id>')
    def get_task(task_id):
        """Get status of a specific task."""
        if task_id in collection_tasks:
            return jsonify(collection_tasks[task_id].get_status())
        elif task_id in verification_tasks:
            return jsonify(verification_tasks[task_id].get_status())
        else:
            return jsonify({"error": f"Task {task_id} not found"}), 404
    
    @app.route('/api/collection/start', methods=['POST'])
    def start_collection():
        """Start a new Street View collection task."""
        try:
            data = request.json
            
            # Validate required fields
            required_fields = ['road_network_path', 'output_dir']
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400
            
            # Generate task ID
            task_id = f"collection_{int(time.time())}"
            
            # Extract parameters
            road_network_path = data['road_network_path']
            output_dir = data['output_dir']
            api_key = data.get('api_key')
            sampling_interval = float(data.get('sampling_interval', 50.0))
            max_points = int(data['max_points']) if 'max_points' in data and data['max_points'] else None
            panorama = data.get('panorama', True)
            max_search_radius = int(data.get('max_search_radius', 1000))
            max_workers = int(data.get('max_workers', 4))
            
            # Create and start collection task
            task = CollectionTask(
                task_id=task_id,
                road_network_path=road_network_path,
                output_dir=output_dir,
                api_key=api_key,
                sampling_interval=sampling_interval,
                max_points=max_points,
                panorama=panorama,
                max_search_radius=max_search_radius,
                max_workers=max_workers
            )
            
            # Store task
            collection_tasks[task_id] = task
            
            # Start task
            task.run()
            
            return jsonify({
                "task_id": task_id,
                "status": "started"
            })
            
        except Exception as e:
            logger.error(f"Error starting collection task: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/verification/start', methods=['POST'])
    def start_verification():
        """Start a new coverage verification task."""
        try:
            data = request.json
            
            # Validate required fields
            required_fields = ['road_network_path', 'coverage_data_path', 'output_dir']
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400
            
            # Generate task ID
            task_id = f"verification_{int(time.time())}"
            
            # Extract parameters
            road_network_path = data['road_network_path']
            coverage_data_path = data['coverage_data_path']
            output_dir = data['output_dir']
            coverage_threshold = float(data.get('coverage_threshold', 50.0))
            max_gap_distance = float(data.get('max_gap_distance', 30.0))
            parallel = data.get('parallel', True)
            max_workers = int(data.get('max_workers', 4))
            
            # Create and start verification task
            task = VerificationTask(
                task_id=task_id,
                road_network_path=road_network_path,
                coverage_data_path=coverage_data_path,
                output_dir=output_dir,
                coverage_threshold=coverage_threshold,
                max_gap_distance=max_gap_distance,
                parallel=parallel,
                max_workers=max_workers
            )
            
            # Store task
            verification_tasks[task_id] = task
            
            # Start task
            task.run()
            
            return jsonify({
                "task_id": task_id,
                "status": "started"
            })
            
        except Exception as e:
            logger.error(f"Error starting verification task: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/visualize/road-network', methods=['POST'])
    def visualize_road_network_api():
        """Generate and return a road network visualization."""
        try:
            data = request.json
            
            # Validate required fields
            if 'road_network_path' not in data:
                return jsonify({"error": "Missing required field: road_network_path"}), 400
            
            # Extract parameters
            road_network_path = data['road_network_path']
            output_path = data.get('output_path')
            show_nodes = data.get('show_nodes', True)
            edge_width = float(data.get('edge_width', 1.0))
            
            # Generate temporary output path if not provided
            if not output_path:
                output_path = os.path.join(static_folder, f"road_network_{int(time.time())}.png")
            
            # Load road network data
            with open(road_network_path, 'r') as f:
                road_network_data = json.load(f)
            
            # Generate visualization
            result = visualize_road_network(
                road_network_data=road_network_data,
                output_path=output_path,
                show_nodes=show_nodes,
                edge_width=edge_width
            )
            
            if not result.get("success", False):
                return jsonify({"error": result.get("error", "Visualization failed")}), 500
            
            # Return image path
            vis_path = result.get("visualization_path")
            vis_url = f"/static/{os.path.basename(vis_path)}"
            
            return jsonify({
                "success": True,
                "visualization_path": vis_path,
                "visualization_url": vis_url,
                "edge_count": result.get("edge_count", 0),
                "node_count": result.get("node_count", 0)
            })
            
        except Exception as e:
            logger.error(f"Error generating road network visualization: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/visualize/coverage', methods=['POST'])
    def visualize_coverage_api():
        """Generate and return a Street View coverage visualization."""
        try:
            data = request.json
            
            # Validate required fields
            required_fields = ['road_network_path', 'coverage_data_path']
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400
            
            # Extract parameters
            road_network_path = data['road_network_path']
            coverage_data_path = data['coverage_data_path']
            output_path = data.get('output_path')
            point_size = float(data.get('point_size', 8.0))
            
            # Generate temporary output path if not provided
            if not output_path:
                output_path = os.path.join(static_folder, f"coverage_{int(time.time())}.png")
            
            # Load road network data
            with open(road_network_path, 'r') as f:
                road_network_data = json.load(f)
            
            # Load coverage data
            with open(coverage_data_path, 'r') as f:
                coverage_data = json.load(f)
            
            # Generate visualization
            result = visualize_street_view_coverage(
                road_network_data=road_network_data,
                coverage_data=coverage_data,
                output_path=output_path,
                point_size=point_size
            )
            
            if not result.get("success", False):
                return jsonify({"error": result.get("error", "Visualization failed")}), 500
            
            # Return image path
            vis_path = result.get("visualization_path")
            vis_url = f"/static/{os.path.basename(vis_path)}"
            
            return jsonify({
                "success": True,
                "visualization_path": vis_path,
                "visualization_url": vis_url,
                "coverage_point_count": result.get("coverage_point_count", 0)
            })
            
        except Exception as e:
            logger.error(f"Error generating coverage visualization: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files."""
        return send_from_directory(static_folder, filename)
    
    @app.route('/api/verify/report/<task_id>', methods=['GET'])
    def get_verification_report(task_id):
        """Get a verification report for a completed verification task."""
        if task_id not in verification_tasks:
            return jsonify({"error": f"Verification task {task_id} not found"}), 404
        
        task = verification_tasks[task_id]
        status = task.get_status()
        
        if status["status"] != "completed":
            return jsonify({"error": f"Verification task {task_id} is not completed"}), 400
        
        report_path = status["result"]["report_path"]
        
        if not os.path.exists(report_path):
            return jsonify({"error": f"Report file not found: {report_path}"}), 404
        
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        return jsonify({
            "task_id": task_id,
            "report_content": report_content
        })
    
    @app.route('/api/collection/progress', methods=['GET'])
    def get_collection_progress():
        """Get progress of all collection tasks."""
        progress = {}
        
        for task_id, task in collection_tasks.items():
            status = task.get_status()
            progress[task_id] = {
                "status": status["status"],
                "progress": status["progress"],
                "elapsed_seconds": status["elapsed_seconds"]
            }
        
        return jsonify(progress)
    
    @app.route('/api/verification/progress', methods=['GET'])
    def get_verification_progress():
        """Get progress of all verification tasks."""
        progress = {}
        
        for task_id, task in verification_tasks.items():
            status = task.get_status()
            progress[task_id] = {
                "status": status["status"],
                "progress": status["progress"],
                "elapsed_seconds": status["elapsed_seconds"]
            }
        
        return jsonify(progress)
    
    @app.route('/api/road-network/sample', methods=['POST'])
    def sample_road_network():
        """Sample a road network to generate collection points."""
        try:
            data = request.json
            
            # Validate required fields
            if 'road_network_path' not in data:
                return jsonify({"error": "Missing required field: road_network_path"}), 400
            
            # Extract parameters
            road_network_path = data['road_network_path']
            sampling_interval = float(data.get('sampling_interval', 50.0))
            output_path = data.get('output_path')
            max_points = int(data['max_points']) if 'max_points' in data and data['max_points'] else None
            
            # Generate output path if not provided
            if not output_path:
                output_dir = os.path.dirname(road_network_path)
                base_name = os.path.splitext(os.path.basename(road_network_path))[0]
                output_path = os.path.join(static_folder, f"{base_name}_samples_{int(time.time())}.json")
            
            # Create road network integration
            rn_integration = road_network_integration.RoadNetworkIntegration(
                os.path.dirname(output_path)
            )
            
            # Load road network
            load_result = rn_integration.load_road_network(road_network_path)
            
            if not load_result:
                return jsonify({"error": f"Failed to load road network from {road_network_path}"}), 500
            
            # Sample points
            sample_result = rn_integration.sample_road_network(
                sampling_interval=sampling_interval,
                max_points=max_points
            )
            
            if not sample_result.get("success", False):
                return jsonify({"error": sample_result.get("error", "Sampling failed")}), 500
            
            # Get sampling points
            points = sample_result.get("points", [])
            
            # Save to output file
            with open(output_path, 'w') as f:
                json.dump({
                    "points": points,
                    "total_points": len(points),
                    "sampling_interval": sampling_interval,
                    "timestamp": time.time()
                }, f, indent=2)
            
            return jsonify({
                "success": True,
                "points_count": len(points),
                "output_path": output_path
            })
            
        except Exception as e:
            logger.error(f"Error sampling road network: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return app

def create_default_template():
    """Create default HTML template for the web interface."""
    # Create templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html template
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arcanum Street View Collection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border-radius: 5px;
        }
        h1, h2, h3 {
            color: #333;
        }
        h1 {
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: #fff;
        }
        .card-header {
            background-color: #f9f9f9;
            padding: 10px 15px;
            margin: -15px -15px 15px;
            border-bottom: 1px solid #ddd;
            font-weight: bold;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 5px;
        }
        button:hover {
            background-color: #45a049;
        }
        button.secondary {
            background-color: #2196F3;
        }
        button.secondary:hover {
            background-color: #0b7dda;
        }
        button.danger {
            background-color: #f44336;
        }
        button.danger:hover {
            background-color: #d32f2f;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        input, select, textarea {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            width: 100%;
            margin-bottom: 10px;
            font-size: 14px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 15px;
            cursor: pointer;
            margin-right: 5px;
            border: 1px solid #ddd;
            border-bottom: none;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        }
        .tab.active {
            background-color: #fff;
            border-bottom: 1px solid #fff;
            margin-bottom: -1px;
            font-weight: bold;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #f9f9f9;
        }
        tr:nth-child(even) {
            background-color: #f5f5f5;
        }
        .progress-bar {
            width: 100%;
            background-color: #e0e0e0;
            border-radius: 4px;
            height: 20px;
            position: relative;
            overflow: hidden;
        }
        .progress-bar-inner {
            height: 100%;
            background-color: #4CAF50;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .progress-text {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            text-align: center;
            color: white;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            line-height: 20px;
        }
        .visualization {
            width: 100%;
            max-width: 600px;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Arcanum Street View Collection</h1>
        
        <div class="tabs">
            <div class="tab active" data-tab="collection">Street View Collection</div>
            <div class="tab" data-tab="visualization">Visualization</div>
            <div class="tab" data-tab="verification">Coverage Verification</div>
            <div class="tab" data-tab="tasks">Task Management</div>
        </div>
        
        <!-- Collection Tab -->
        <div class="tab-content active" id="collection-tab">
            <h2>Collection Configuration</h2>
            <div class="card">
                <div class="card-header">Collection Parameters</div>
                <div class="form-group">
                    <label for="road-network-path">Road Network Path:</label>
                    <input type="text" id="road-network-path" placeholder="/path/to/road_network.json">
                </div>
                <div class="form-group">
                    <label for="output-dir">Output Directory:</label>
                    <input type="text" id="output-dir" placeholder="/path/to/output/directory">
                </div>
                <div class="form-group">
                    <label for="api-key">API Key (optional):</label>
                    <input type="text" id="api-key" placeholder="Google Maps API Key">
                </div>
                <div class="form-group">
                    <label for="sampling-interval">Sampling Interval (meters):</label>
                    <input type="number" id="sampling-interval" value="50" min="1" max="1000">
                </div>
                <div class="form-group">
                    <label for="max-points">Max Points (optional):</label>
                    <input type="number" id="max-points" placeholder="Leave empty for all points">
                </div>
                <div class="form-group">
                    <label for="max-search-radius">Max Search Radius (meters):</label>
                    <input type="number" id="max-search-radius" value="1000" min="1" max="10000">
                </div>
                <div class="form-group">
                    <label for="max-workers">Max Workers:</label>
                    <input type="number" id="max-workers" value="4" min="1" max="32">
                </div>
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="panorama" checked>
                        Capture Full Panoramas
                    </label>
                </div>
                <button id="start-collection-btn">Start Collection</button>
                <button id="sample-road-network-btn" class="secondary">Sample Road Network</button>
            </div>
            
            <h2>Collection Progress</h2>
            <div class="card" id="collection-status">
                <div class="card-header">Status</div>
                <p>No collection task running.</p>
            </div>
        </div>
        
        <!-- Visualization Tab -->
        <div class="tab-content" id="visualization-tab">
            <h2>Road Network & Coverage Visualization</h2>
            <div class="card">
                <div class="card-header">Visualization Parameters</div>
                <div class="form-group">
                    <label for="vis-road-network-path">Road Network Path:</label>
                    <input type="text" id="vis-road-network-path" placeholder="/path/to/road_network.json">
                </div>
                <div class="form-group">
                    <label for="vis-coverage-data-path">Coverage Data Path (optional):</label>
                    <input type="text" id="vis-coverage-data-path" placeholder="/path/to/coverage_data.json">
                </div>
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="show-nodes" checked>
                        Show Road Intersection Nodes
                    </label>
                </div>
                <button id="visualize-road-network-btn">Visualize Road Network</button>
                <button id="visualize-coverage-btn" class="secondary">Visualize Coverage</button>
            </div>
            
            <h2>Visualization Result</h2>
            <div class="card" id="visualization-result">
                <div class="card-header">Result</div>
                <p>No visualization generated yet.</p>
            </div>
        </div>
        
        <!-- Verification Tab -->
        <div class="tab-content" id="verification-tab">
            <h2>Coverage Verification</h2>
            <div class="card">
                <div class="card-header">Verification Parameters</div>
                <div class="form-group">
                    <label for="ver-road-network-path">Road Network Path:</label>
                    <input type="text" id="ver-road-network-path" placeholder="/path/to/road_network.json">
                </div>
                <div class="form-group">
                    <label for="ver-coverage-data-path">Coverage Data Path:</label>
                    <input type="text" id="ver-coverage-data-path" placeholder="/path/to/coverage_data.json">
                </div>
                <div class="form-group">
                    <label for="ver-output-dir">Output Directory:</label>
                    <input type="text" id="ver-output-dir" placeholder="/path/to/output/directory">
                </div>
                <div class="form-group">
                    <label for="coverage-threshold">Coverage Threshold (%):</label>
                    <input type="number" id="coverage-threshold" value="50" min="0" max="100">
                </div>
                <div class="form-group">
                    <label for="max-gap-distance">Max Gap Distance (meters):</label>
                    <input type="number" id="max-gap-distance" value="30" min="1" max="1000">
                </div>
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="parallel" checked>
                        Process Edges in Parallel
                    </label>
                </div>
                <button id="start-verification-btn">Start Verification</button>
            </div>
            
            <h2>Verification Progress</h2>
            <div class="card" id="verification-status">
                <div class="card-header">Status</div>
                <p>No verification task running.</p>
            </div>
            
            <h2>Verification Report</h2>
            <div class="card" id="verification-report">
                <div class="card-header">Report</div>
                <p>No verification report available.</p>
            </div>
        </div>
        
        <!-- Tasks Tab -->
        <div class="tab-content" id="tasks-tab">
            <h2>Task Management</h2>
            <div class="card">
                <div class="card-header">Active Tasks</div>
                <table id="active-tasks-table">
                    <thead>
                        <tr>
                            <th>Task ID</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Progress</th>
                            <th>Elapsed Time</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td colspan="6">No active tasks.</td>
                        </tr>
                    </tbody>
                </table>
                <button id="refresh-tasks-btn">Refresh Tasks</button>
            </div>
            
            <h2>Completed Tasks</h2>
            <div class="card">
                <div class="card-header">Completed Tasks</div>
                <table id="completed-tasks-table">
                    <thead>
                        <tr>
                            <th>Task ID</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Completion Time</th>
                            <th>Results</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td colspan="5">No completed tasks.</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs and content
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked tab and corresponding content
                tab.classList.add('active');
                document.getElementById(`${tab.dataset.tab}-tab`).classList.add('active');
            });
        });
        
        // Initialize API endpoint
        const API_BASE_URL = '';
        
        // Format time in seconds to readable format
        function formatElapsedTime(seconds) {
            if (!seconds) return '0s';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            
            let result = '';
            if (hours > 0) result += `${hours}h `;
            if (minutes > 0 || hours > 0) result += `${minutes}m `;
            result += `${secs}s`;
            
            return result;
        }
        
        // Start collection task
        document.getElementById('start-collection-btn').addEventListener('click', async () => {
            const roadNetworkPath = document.getElementById('road-network-path').value;
            const outputDir = document.getElementById('output-dir').value;
            const apiKey = document.getElementById('api-key').value;
            const samplingInterval = document.getElementById('sampling-interval').value;
            const maxPoints = document.getElementById('max-points').value;
            const maxSearchRadius = document.getElementById('max-search-radius').value;
            const maxWorkers = document.getElementById('max-workers').value;
            const panorama = document.getElementById('panorama').checked;
            
            if (!roadNetworkPath || !outputDir) {
                alert('Road Network Path and Output Directory are required!');
                return;
            }
            
            try {
                const response = await fetch(`${API_BASE_URL}/api/collection/start`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        road_network_path: roadNetworkPath,
                        output_dir: outputDir,
                        api_key: apiKey || undefined,
                        sampling_interval: parseFloat(samplingInterval),
                        max_points: maxPoints ? parseInt(maxPoints) : undefined,
                        max_search_radius: parseInt(maxSearchRadius),
                        max_workers: parseInt(maxWorkers),
                        panorama: panorama
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    alert(`Collection task started with ID: ${data.task_id}`);
                    updateCollectionProgress(data.task_id);
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                console.error('Error starting collection:', error);
                alert(`Error: ${error.message}`);
            }
        });
        
        // Sample road network
        document.getElementById('sample-road-network-btn').addEventListener('click', async () => {
            const roadNetworkPath = document.getElementById('road-network-path').value;
            const samplingInterval = document.getElementById('sampling-interval').value;
            const maxPoints = document.getElementById('max-points').value;
            
            if (!roadNetworkPath) {
                alert('Road Network Path is required!');
                return;
            }
            
            try {
                const response = await fetch(`${API_BASE_URL}/api/road-network/sample`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        road_network_path: roadNetworkPath,
                        sampling_interval: parseFloat(samplingInterval),
                        max_points: maxPoints ? parseInt(maxPoints) : undefined
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    alert(`Sampled ${data.points_count} points from road network. Output saved to: ${data.output_path}`);
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                console.error('Error sampling road network:', error);
                alert(`Error: ${error.message}`);
            }
        });
        
        // Visualize road network
        document.getElementById('visualize-road-network-btn').addEventListener('click', async () => {
            const roadNetworkPath = document.getElementById('vis-road-network-path').value;
            const showNodes = document.getElementById('show-nodes').checked;
            
            if (!roadNetworkPath) {
                alert('Road Network Path is required!');
                return;
            }
            
            try {
                const response = await fetch(`${API_BASE_URL}/api/visualize/road-network`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        road_network_path: roadNetworkPath,
                        show_nodes: showNodes
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    const resultElement = document.getElementById('visualization-result');
                    resultElement.innerHTML = `
                        <div class="card-header">Road Network Visualization</div>
                        <p>Road Network visualization generated with ${data.edge_count} edges and ${data.node_count} nodes.</p>
                        <img src="${data.visualization_url}" class="visualization" alt="Road Network Visualization">
                    `;
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                console.error('Error visualizing road network:', error);
                alert(`Error: ${error.message}`);
            }
        });
        
        // Visualize coverage
        document.getElementById('visualize-coverage-btn').addEventListener('click', async () => {
            const roadNetworkPath = document.getElementById('vis-road-network-path').value;
            const coverageDataPath = document.getElementById('vis-coverage-data-path').value;
            
            if (!roadNetworkPath || !coverageDataPath) {
                alert('Road Network Path and Coverage Data Path are required!');
                return;
            }
            
            try {
                const response = await fetch(`${API_BASE_URL}/api/visualize/coverage`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        road_network_path: roadNetworkPath,
                        coverage_data_path: coverageDataPath
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    const resultElement = document.getElementById('visualization-result');
                    resultElement.innerHTML = `
                        <div class="card-header">Coverage Visualization</div>
                        <p>Coverage visualization generated with ${data.coverage_point_count} coverage points.</p>
                        <img src="${data.visualization_url}" class="visualization" alt="Coverage Visualization">
                    `;
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                console.error('Error visualizing coverage:', error);
                alert(`Error: ${error.message}`);
            }
        });
        
        // Start verification task
        document.getElementById('start-verification-btn').addEventListener('click', async () => {
            const roadNetworkPath = document.getElementById('ver-road-network-path').value;
            const coverageDataPath = document.getElementById('ver-coverage-data-path').value;
            const outputDir = document.getElementById('ver-output-dir').value;
            const coverageThreshold = document.getElementById('coverage-threshold').value;
            const maxGapDistance = document.getElementById('max-gap-distance').value;
            const parallel = document.getElementById('parallel').checked;
            
            if (!roadNetworkPath || !coverageDataPath || !outputDir) {
                alert('Road Network Path, Coverage Data Path, and Output Directory are required!');
                return;
            }
            
            try {
                const response = await fetch(`${API_BASE_URL}/api/verification/start`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        road_network_path: roadNetworkPath,
                        coverage_data_path: coverageDataPath,
                        output_dir: outputDir,
                        coverage_threshold: parseFloat(coverageThreshold),
                        max_gap_distance: parseFloat(maxGapDistance),
                        parallel: parallel
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    alert(`Verification task started with ID: ${data.task_id}`);
                    updateVerificationProgress(data.task_id);
                } else {
                    alert(`Error: ${data.error}`);
                }
            } catch (error) {
                console.error('Error starting verification:', error);
                alert(`Error: ${error.message}`);
            }
        });
        
        // Update collection progress
        async function updateCollectionProgress(taskId) {
            const statusElement = document.getElementById('collection-status');
            let intervalId;
            
            async function updateStatus() {
                try {
                    const response = await fetch(`${API_BASE_URL}/api/task/${taskId}`);
                    const data = await response.json();
                    
                    // Update status display
                    statusElement.innerHTML = `
                        <div class="card-header">Collection Status: ${data.status}</div>
                        <div class="progress-bar">
                            <div class="progress-bar-inner" style="width: ${data.progress}%"></div>
                            <div class="progress-text">${data.progress}%</div>
                        </div>
                        <p>Time elapsed: ${formatElapsedTime(data.elapsed_seconds)}</p>
                    `;
                    
                    // If task is completed, show results and stop polling
                    if (data.status === 'completed') {
                        statusElement.innerHTML += `
                            <p>Successfully processed ${data.result.points_processed} points</p>
                            <p>Found imagery at ${data.result.points_with_imagery} locations</p>
                            <p>Downloaded ${data.result.images_downloaded} Street View images</p>
                        `;
                        clearInterval(intervalId);
                    } else if (data.status === 'failed') {
                        statusElement.innerHTML += `
                            <p>Error: ${data.error}</p>
                        `;
                        clearInterval(intervalId);
                    }
                } catch (error) {
                    console.error('Error updating collection status:', error);
                    statusElement.innerHTML += `
                        <p>Error fetching status: ${error.message}</p>
                    `;
                    clearInterval(intervalId);
                }
            }
            
            // Initial update
            await updateStatus();
            
            // Start polling every 2 seconds
            intervalId = setInterval(updateStatus, 2000);
        }
        
        // Update verification progress
        async function updateVerificationProgress(taskId) {
            const statusElement = document.getElementById('verification-status');
            const reportElement = document.getElementById('verification-report');
            let intervalId;
            
            async function updateStatus() {
                try {
                    const response = await fetch(`${API_BASE_URL}/api/task/${taskId}`);
                    const data = await response.json();
                    
                    // Update status display
                    statusElement.innerHTML = `
                        <div class="card-header">Verification Status: ${data.status}</div>
                        <div class="progress-bar">
                            <div class="progress-bar-inner" style="width: ${data.progress}%"></div>
                            <div class="progress-text">${data.progress}%</div>
                        </div>
                        <p>Time elapsed: ${formatElapsedTime(data.elapsed_seconds)}</p>
                    `;
                    
                    // If task is completed, show results and stop polling
                    if (data.status === 'completed') {
                        statusElement.innerHTML += `
                            <p>Verification completed successfully</p>
                        `;
                        
                        // Fetch and display the report
                        const reportResponse = await fetch(`${API_BASE_URL}/api/verify/report/${taskId}`);
                        const reportData = await reportResponse.json();
                        
                        if (reportResponse.ok) {
                            reportElement.innerHTML = `
                                <div class="card-header">Verification Report</div>
                                <pre>${reportData.report_content}</pre>
                            `;
                            
                            // Add visualization if available
                            if (data.result.visualization_path) {
                                const pathComponents = data.result.visualization_path.split('/');
                                const filename = pathComponents[pathComponents.length - 1];
                                reportElement.innerHTML += `
                                    <img src="/static/${filename}" class="visualization" alt="Coverage Verification Visualization">
                                `;
                            }
                        }
                        
                        clearInterval(intervalId);
                    } else if (data.status === 'failed') {
                        statusElement.innerHTML += `
                            <p>Error: ${data.error}</p>
                        `;
                        clearInterval(intervalId);
                    }
                } catch (error) {
                    console.error('Error updating verification status:', error);
                    statusElement.innerHTML += `
                        <p>Error fetching status: ${error.message}</p>
                    `;
                    clearInterval(intervalId);
                }
            }
            
            // Initial update
            await updateStatus();
            
            // Start polling every 2 seconds
            intervalId = setInterval(updateStatus, 2000);
        }
        
        // Refresh tasks list
        async function refreshTasks() {
            try {
                const response = await fetch(`${API_BASE_URL}/api/tasks`);
                const data = await response.json();
                
                // Update active tasks table
                const activeTasksBody = document.querySelector('#active-tasks-table tbody');
                const completedTasksBody = document.querySelector('#completed-tasks-table tbody');
                
                let activeTasksHTML = '';
                let completedTasksHTML = '';
                
                // Process collection tasks
                Object.entries(data.collection_tasks).forEach(([taskId, task]) => {
                    if (task.status === 'running' || task.status === 'created') {
                        activeTasksHTML += `
                            <tr>
                                <td>${taskId}</td>
                                <td>Collection</td>
                                <td>${task.status}</td>
                                <td>
                                    <div class="progress-bar">
                                        <div class="progress-bar-inner" style="width: ${task.progress}%"></div>
                                        <div class="progress-text">${task.progress}%</div>
                                    </div>
                                </td>
                                <td>${formatElapsedTime(task.elapsed_seconds)}</td>
                                <td><button class="view-task-btn" data-task-id="${taskId}">View</button></td>
                            </tr>
                        `;
                    } else if (task.status === 'completed' || task.status === 'failed') {
                        completedTasksHTML += `
                            <tr>
                                <td>${taskId}</td>
                                <td>Collection</td>
                                <td>${task.status}</td>
                                <td>${formatElapsedTime(task.elapsed_seconds)}</td>
                                <td>
                                    ${task.status === 'completed' ? 
                                        `Processed ${task.result.points_processed} points, ${task.result.points_with_imagery} with imagery` :
                                        `Error: ${task.error}`}
                                </td>
                            </tr>
                        `;
                    }
                });
                
                // Process verification tasks
                Object.entries(data.verification_tasks).forEach(([taskId, task]) => {
                    if (task.status === 'running' || task.status === 'created') {
                        activeTasksHTML += `
                            <tr>
                                <td>${taskId}</td>
                                <td>Verification</td>
                                <td>${task.status}</td>
                                <td>
                                    <div class="progress-bar">
                                        <div class="progress-bar-inner" style="width: ${task.progress}%"></div>
                                        <div class="progress-text">${task.progress}%</div>
                                    </div>
                                </td>
                                <td>${formatElapsedTime(task.elapsed_seconds)}</td>
                                <td><button class="view-task-btn" data-task-id="${taskId}">View</button></td>
                            </tr>
                        `;
                    } else if (task.status === 'completed' || task.status === 'failed') {
                        completedTasksHTML += `
                            <tr>
                                <td>${taskId}</td>
                                <td>Verification</td>
                                <td>${task.status}</td>
                                <td>${formatElapsedTime(task.elapsed_seconds)}</td>
                                <td>
                                    ${task.status === 'completed' ? 
                                        `<button class="view-report-btn" data-task-id="${taskId}">View Report</button>` :
                                        `Error: ${task.error}`}
                                </td>
                            </tr>
                        `;
                    }
                });
                
                // Update tables
                if (activeTasksHTML) {
                    activeTasksBody.innerHTML = activeTasksHTML;
                } else {
                    activeTasksBody.innerHTML = '<tr><td colspan="6">No active tasks.</td></tr>';
                }
                
                if (completedTasksHTML) {
                    completedTasksBody.innerHTML = completedTasksHTML;
                } else {
                    completedTasksBody.innerHTML = '<tr><td colspan="5">No completed tasks.</td></tr>';
                }
                
                // Add event listeners for view task buttons
                document.querySelectorAll('.view-task-btn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const taskId = btn.dataset.taskId;
                        
                        if (taskId.startsWith('collection_')) {
                            updateCollectionProgress(taskId);
                            
                            // Switch to collection tab
                            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                            document.querySelector('.tab[data-tab="collection"]').classList.add('active');
                            document.getElementById('collection-tab').classList.add('active');
                            
                        } else if (taskId.startsWith('verification_')) {
                            updateVerificationProgress(taskId);
                            
                            // Switch to verification tab
                            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                            document.querySelector('.tab[data-tab="verification"]').classList.add('active');
                            document.getElementById('verification-tab').classList.add('active');
                        }
                    });
                });
                
                // Add event listeners for view report buttons
                document.querySelectorAll('.view-report-btn').forEach(btn => {
                    btn.addEventListener('click', async () => {
                        const taskId = btn.dataset.taskId;
                        
                        try {
                            const reportResponse = await fetch(`${API_BASE_URL}/api/verify/report/${taskId}`);
                            const reportData = await reportResponse.json();
                            
                            if (reportResponse.ok) {
                                const reportElement = document.getElementById('verification-report');
                                reportElement.innerHTML = `
                                    <div class="card-header">Verification Report</div>
                                    <pre>${reportData.report_content}</pre>
                                `;
                                
                                // Switch to verification tab
                                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                                document.querySelector('.tab[data-tab="verification"]').classList.add('active');
                                document.getElementById('verification-tab').classList.add('active');
                            } else {
                                alert(`Error: ${reportData.error}`);
                            }
                        } catch (error) {
                            console.error('Error fetching report:', error);
                            alert(`Error: ${error.message}`);
                        }
                    });
                });
                
            } catch (error) {
                console.error('Error refreshing tasks:', error);
                alert(`Error: ${error.message}`);
            }
        }
        
        // Refresh tasks button
        document.getElementById('refresh-tasks-btn').addEventListener('click', refreshTasks);
        
        // Initial tasks load
        refreshTasks();
        
        // Auto-refresh active tasks every 10 seconds
        setInterval(refreshTasks, 10000);
    </script>
</body>
</html>
"""
    
    # Save the template
    template_path = os.path.join(templates_dir, "index.html")
    with open(template_path, "w") as f:
        f.write(index_html)
    
    logger.info(f"Created default HTML template at {template_path}")

def main():
    """Main function for running the web interface."""
    parser = argparse.ArgumentParser(description="Arcanum Street View Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Create the Flask app
    app = create_app()
    
    if not app:
        logger.error("Failed to create web application. Exiting.")
        sys.exit(1)
    
    # Print setup instructions
    logger.info("=" * 80)
    logger.info("Arcanum Street View Web Interface")
    logger.info("=" * 80)
    logger.info(f"Server starting at http://{args.host}:{args.port}")
    logger.info("Use Ctrl+C to stop the server")
    
    # Run the app
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

if __name__ == "__main__":
    main()