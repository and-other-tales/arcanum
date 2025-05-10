#!/usr/bin/env python3
"""
Workflow Module for Arcanum
--------------------------
This module provides high-level workflow functions that coordinate multiple tasks
across different modules using the parallel task processing system.
"""

import os
import sys
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import json

from .task_manager import submit_task, wait_for_task, execute_parallel, TaskProgressCallback

# Configure logging
logger = logging.getLogger(__name__)

class ArcanumWorkflow:
    """
    Manages complex Arcanum workflows with multiple dependent tasks.
    A workflow represents a complete data processing pipeline from input to output.
    """
    
    def __init__(self, workflow_id: str = None, name: str = None, config: Dict = None):
        """
        Initialize a new workflow.
        
        Args:
            workflow_id: Optional unique identifier (auto-generated if not provided)
            name: Human-readable name for the workflow
            config: Configuration options for the workflow
        """
        import uuid
        
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.name = name or f"Workflow-{self.workflow_id[:8]}"
        self.config = config or {}
        
        # Task tracking
        self.tasks = {}  # task_id -> task_info
        self.task_graph = {}  # task_id -> list of dependent task_ids
        
        # Start time and status
        self.start_time = time.time()
        self.end_time = None
        self.status = "initializing"  # initializing, running, completed, failed
        
        # Results storage
        self.results = {}
        self.errors = {}
        
        logger.info(f"Initialized workflow {self.workflow_id} ({self.name})")
        
    def add_task(self, name: str, func: Callable, *args, 
                dependencies: List[str] = None, **kwargs) -> str:
        """
        Add a task to the workflow.
        
        Args:
            name: Human-readable name for the task
            func: The function to execute
            *args: Positional arguments for the function
            dependencies: List of task IDs this task depends on
            **kwargs: Keyword arguments passed to submit_task or the function
            
        Returns:
            task_id: The unique ID assigned to the task
        """
        # Submit the task with dependencies
        task_id = submit_task(name, func, *args, dependencies=dependencies, **kwargs)
        
        # Record task in our workflow
        self.tasks[task_id] = {
            "name": name,
            "dependencies": dependencies or [],
            "added_time": time.time()
        }
        
        # Update dependency graph
        self.task_graph[task_id] = []
        if dependencies:
            for dep_id in dependencies:
                if dep_id in self.task_graph:
                    self.task_graph[dep_id].append(task_id)
        
        logger.debug(f"Added task {task_id} ({name}) to workflow {self.workflow_id}")
        return task_id
        
    def run(self, wait: bool = True, timeout: float = None) -> Dict:
        """
        Execute the workflow.
        
        Args:
            wait: If True, wait for all tasks to complete
            timeout: Maximum time to wait in seconds
            
        Returns:
            Workflow status information
        """
        # Update status
        self.status = "running"
        
        # If wait is True, wait for all tasks to complete
        if wait:
            from .task_manager import get_default_task_manager
            task_manager = get_default_task_manager()
            
            # Wait for all tasks
            task_ids = list(self.tasks.keys())
            statuses = task_manager.wait_for_tasks(task_ids, timeout)
            
            # Get results for all completed tasks
            for task_id, status in statuses.items():
                if status == "completed":
                    result = task_manager.get_task_result(task_id)
                    if result["success"]:
                        self.results[task_id] = result["result"]
                    else:
                        self.errors[task_id] = result["error"]
            
            # Update workflow status
            if all(status == "completed" for status in statuses.values()):
                self.status = "completed"
            elif any(status == "failed" for status in statuses.values()):
                self.status = "failed"
                
            self.end_time = time.time()
            
        return self.get_status()
        
    def get_status(self) -> Dict:
        """
        Get the current status of the workflow.
        
        Returns:
            Dictionary with workflow status information
        """
        from .task_manager import get_default_task_manager
        task_manager = get_default_task_manager()
        
        # Get status of all tasks
        task_statuses = {}
        for task_id in self.tasks:
            task_info = task_manager.get_task(task_id)
            if task_info:
                task_statuses[task_id] = {
                    "name": task_info["name"],
                    "status": task_info["status"],
                    "progress": task_info["progress"],
                    "error": task_info["error"]
                }
            else:
                task_statuses[task_id] = {
                    "name": self.tasks[task_id]["name"],
                    "status": "unknown",
                    "progress": 0.0,
                    "error": None
                }
        
        # Calculate overall progress
        if not self.tasks:
            overall_progress = 0.0
        else:
            # Weight progress by task dependencies (tasks with more dependents are more important)
            total_weight = 0
            weighted_progress = 0.0
            
            for task_id, task_status in task_statuses.items():
                # Calculate weight based on number of dependent tasks
                weight = len(self.task_graph.get(task_id, [])) + 1  # +1 so every task has at least weight 1
                total_weight += weight
                
                # Add weighted progress
                weighted_progress += task_status["progress"] * weight
                
            overall_progress = weighted_progress / total_weight if total_weight > 0 else 0.0
            
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status,
            "progress": overall_progress,
            "tasks": task_statuses,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_time": (self.end_time - self.start_time) if self.end_time else (time.time() - self.start_time),
            "results_count": len(self.results),
            "errors_count": len(self.errors)
        }
        
    def get_results(self) -> Dict:
        """
        Get the results of the workflow.
        
        Returns:
            Dictionary with task results
        """
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status,
            "results": self.results,
            "errors": self.errors
        }
        
    def cancel(self) -> bool:
        """
        Cancel the workflow by cancelling all running tasks.
        
        Returns:
            True if workflow was cancelled, False otherwise
        """
        from .task_manager import get_default_task_manager
        task_manager = get_default_task_manager()
        
        cancelled = False
        for task_id in self.tasks:
            if task_manager.cancel_task(task_id):
                cancelled = True
                
        if cancelled:
            self.status = "cancelled"
            self.end_time = time.time()
            
        return cancelled


def osm_workflow(bounds: Dict[str, float], output_dir: str, 
                region: str = "london", grid_size: int = None,
                coordinate_system: str = "EPSG:4326") -> str:
    """
    Create a workflow for the OSM data processing pipeline.
    
    Args:
        bounds: Dictionary with north, south, east, west boundaries
        output_dir: Directory to save output files
        region: Region name for Geofabrik downloads
        grid_size: Size of grid cells in meters (for grid downloader)
        coordinate_system: Coordinate system for the input bounds
        
    Returns:
        workflow_id: ID of the created workflow
    """
    # Import modules
    from modules.osm import geofabrik_downloader, bbox_downloader, grid_downloader
    
    # Create workflow
    workflow = ArcanumWorkflow(name=f"OSM-Data-{region}")
    
    # First task: download OSM data
    download_task = workflow.add_task(
        "Download OSM Data",
        geofabrik_downloader.download_for_bbox,
        bounds=bounds,
        output_dir=output_dir,
        region=region
    )
    
    # Use grid-based approach as a fallback if specified
    if grid_size:
        grid_task = workflow.add_task(
            "Grid Downloader Fallback",
            grid_downloader.download_osm_grid,
            bounds=bounds,
            output_dir=output_dir,
            cell_size=grid_size,
            coordinate_system=coordinate_system,
            dependencies=[download_task]  # Only execute if primary download fails
        )
    
    # Run the workflow asynchronously (don't wait)
    workflow.run(wait=False)
    
    return workflow.workflow_id


def building_generation_workflow(buildings_path: str, output_dir: str, 
                                style_id: str = "arcanum_victorian", 
                                use_ai: bool = True) -> str:
    """
    Create a workflow for building generation pipeline.
    
    Args:
        buildings_path: Path to the buildings data file
        output_dir: Directory to save output files
        style_id: ID of the style to apply
        use_ai: Whether to use AI for enhancing buildings
        
    Returns:
        workflow_id: ID of the created workflow
    """
    # Import necessary modules
    from modules.comfyui import automation
    from modules.styles import style_manager
    from modules.ai import building_generator
    from modules.exporters import export_manager
    
    # Create workflow
    workflow = ArcanumWorkflow(name=f"Building-Generation-{os.path.basename(buildings_path)}")
    
    # First task: prepare the style
    style_task = workflow.add_task(
        "Prepare Style",
        style_manager.prepare_style,
        style_id=style_id
    )
    
    # Second task: process buildings with ComfyUI
    buildings_task = workflow.add_task(
        "Process Buildings",
        automation.process_buildings,
        buildings_path=buildings_path,
        output_dir=output_dir,
        dependencies=[style_task]
    )
    
    # Optional AI enhancement
    if use_ai:
        ai_task = workflow.add_task(
            "AI Enhancement",
            building_generator.enhance_buildings,
            buildings_path=buildings_path,
            processed_dir=output_dir,
            style_id=style_id,
            dependencies=[buildings_task]
        )
        
        # Export to 3D models with AI enhancements
        export_task = workflow.add_task(
            "Export 3D Models",
            export_manager.export_buildings,
            buildings_path=buildings_path,
            processed_dir=output_dir,
            format="glb",
            dependencies=[ai_task]
        )
    else:
        # Export to 3D models without AI
        export_task = workflow.add_task(
            "Export 3D Models",
            export_manager.export_buildings,
            buildings_path=buildings_path,
            processed_dir=output_dir,
            format="glb",
            dependencies=[buildings_task]
        )
    
    # Run the workflow asynchronously
    workflow.run(wait=False)
    
    return workflow.workflow_id


def arcanum_city_workflow(bounds: Dict[str, float], output_dir: str,
                         style_id: str = "arcanum_victorian",
                         region: str = "london",
                         use_ai: bool = True,
                         use_3d_tiles: bool = False,
                         export_formats: List[str] = None) -> str:
    """
    Create a complete workflow for Arcanum city generation.
    
    Args:
        bounds: Dictionary with north, south, east, west boundaries
        output_dir: Directory to save output files
        style_id: ID of the style to apply
        region: Region name for Geofabrik downloads
        use_ai: Whether to use AI enhancements
        use_3d_tiles: Whether to include Google 3D tiles
        export_formats: List of export formats (defaults to ["glb", "unity"])
        
    Returns:
        workflow_id: ID of the created workflow
    """
    # Set default export formats
    if export_formats is None:
        export_formats = ["glb", "unity"]
    
    # Create workflow
    workflow = ArcanumWorkflow(name=f"Arcanum-City-{region}")
    
    # Step 1: Download OSM data
    osm_task = workflow.add_task(
        "Download OSM Data",
        osm_workflow,
        bounds=bounds,
        output_dir=output_dir,
        region=region
    )
    
    # Step 2: Prepare style
    from modules.styles import style_manager
    style_task = workflow.add_task(
        "Prepare Style",
        style_manager.prepare_style,
        style_id=style_id
    )
    
    # Step 3: Process buildings
    buildings_task = workflow.add_task(
        "Process Buildings",
        building_generation_workflow,
        buildings_path=f"{output_dir}/vector/osm_arcanum.gpkg",
        output_dir=f"{output_dir}/processed_data",
        style_id=style_id,
        use_ai=use_ai,
        dependencies=[osm_task, style_task]
    )
    
    # Step 4: Optional Google 3D Tiles
    if use_3d_tiles:
        from integration_tools import google_3d_tiles_integration
        tiles_task = workflow.add_task(
            "Fetch 3D Tiles",
            google_3d_tiles_integration.fetch_tiles,
            bounds=bounds,
            output_dir=f"{output_dir}/3d_tiles",
            region=region
        )
    
    # Step 5: Export for Unity
    if "unity" in export_formats:
        from integration_tools import unity_integration
        unity_task = workflow.add_task(
            "Export Unity Package",
            unity_integration.export_unity_package,
            output_dir=output_dir,
            buildings_dir=f"{output_dir}/3d_models/buildings",
            unity_dir=f"{output_dir}/unity_assets",
            dependencies=[buildings_task]
        )
    
    # Run the workflow
    workflow.run(wait=False)
    
    return workflow.workflow_id


# Dictionary to store active workflows
_active_workflows = {}

def get_workflow(workflow_id: str) -> Optional[ArcanumWorkflow]:
    """Get a workflow by ID."""
    return _active_workflows.get(workflow_id)

def list_workflows() -> List[Dict]:
    """List all active workflows."""
    return [workflow.get_status() for workflow in _active_workflows.values()]

def register_workflow(workflow: ArcanumWorkflow) -> None:
    """Register a workflow in the global registry."""
    _active_workflows[workflow.workflow_id] = workflow

def cleanup_workflows(max_age: float = 86400) -> int:
    """
    Clean up old workflows.
    
    Args:
        max_age: Maximum age in seconds (default: 24 hours)
        
    Returns:
        Number of workflows cleaned up
    """
    current_time = time.time()
    to_remove = []
    
    for workflow_id, workflow in _active_workflows.items():
        if workflow.end_time and (current_time - workflow.end_time) > max_age:
            to_remove.append(workflow_id)
            
    for workflow_id in to_remove:
        del _active_workflows[workflow_id]
        
    return len(to_remove)