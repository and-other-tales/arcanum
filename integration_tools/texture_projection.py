#!/usr/bin/env python3
"""
Texture Projection System
------------------------
Maps high-resolution Street View imagery onto 3D geometric models with pixel-perfect alignment.
This module provides precise texture projection, blending, and alignment capabilities.
"""

import os
import sys
import numpy as np
import cv2
import math
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import uuid
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import configuration
try:
    from integration_tools.config import get_cache_dir, load_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants for spherical projection calculations
EARTH_RADIUS = 6371000  # in meters
DEFAULT_CAMERA_HEIGHT = 2.1  # meters above ground level

@dataclass
class CameraParameters:
    """Camera parameters for projection calculations."""
    position: Tuple[float, float, float]  # x, y, z world position in meters
    orientation: Tuple[float, float, float]  # roll, pitch, yaw in radians
    fov_horizontal: float  # field of view in radians
    fov_vertical: float  # field of view in radians
    width: int  # image width in pixels
    height: int  # image height in pixels
    near_clip: float = 0.1  # near clip plane in meters
    far_clip: float = 100.0  # far clip plane in meters

    @property
    def aspect_ratio(self) -> float:
        """Calculate the aspect ratio of the camera."""
        return self.width / self.height

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix for the camera orientation."""
        rotation = Rotation.from_euler('xyz', self.orientation, degrees=False)
        return rotation.as_matrix()

    @property
    def view_matrix(self) -> np.ndarray:
        """Calculate the view matrix for the camera."""
        # Create rotation matrix
        R = self.rotation_matrix
        
        # Get position vector
        t = np.array(self.position, dtype=np.float32).reshape(3, 1)
        
        # Construct view matrix (4x4)
        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[:3, :3] = R
        view_matrix[:3, 3] = -R @ t.flatten()
        
        return view_matrix

    @property
    def projection_matrix(self) -> np.ndarray:
        """Calculate the projection matrix for the camera."""
        # Use horizontal FOV for aspect ratio calculations
        fov = self.fov_horizontal
        aspect = self.aspect_ratio
        n = self.near_clip
        f = self.far_clip
        
        # Vertical scale based on horizontal FOV and aspect ratio
        tan_half_fov = math.tan(fov / 2)
        
        # Build projection matrix
        projection = np.zeros((4, 4), dtype=np.float32)
        projection[0, 0] = 1.0 / (aspect * tan_half_fov)
        projection[1, 1] = 1.0 / tan_half_fov
        projection[2, 2] = -(f + n) / (f - n)
        projection[2, 3] = -2.0 * f * n / (f - n)
        projection[3, 2] = -1.0
        
        return projection

class TextureProjector:
    """Class for projecting Street View images onto 3D models with precise alignment."""

    def __init__(self, 
                cache_dir: Optional[str] = None,
                output_dir: Optional[str] = None,
                parallel_processing: bool = True,
                max_workers: Optional[int] = None):
        """
        Initialize the Texture Projector.
        
        Args:
            cache_dir: Directory to store intermediate results
            output_dir: Directory to save projected textures
            parallel_processing: Whether to use parallel processing for projections
            max_workers: Maximum number of worker processes/threads to use
        """
        # Setup cache and output directories
        if CONFIG_AVAILABLE:
            self.cache_dir = Path(cache_dir or get_cache_dir("texture_projection"))
        else:
            self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/arcanum/texture_projection"))
        
        self.output_dir = Path(output_dir or "texture_output")
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different processing stages
        self.rectified_dir = self.cache_dir / "rectified"
        self.projected_dir = self.cache_dir / "projected"
        self.blended_dir = self.cache_dir / "blended"
        
        for directory in [self.rectified_dir, self.projected_dir, self.blended_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Parallel processing settings
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        
        # Pre-calculated data for optimizing reprojection operations
        self.projection_caches = {}
        
        logger.info(f"TextureProjector initialized with cache at {self.cache_dir}")
        logger.info(f"Parallel processing: {self.parallel_processing} with {self.max_workers} workers")

    def rectify_street_view_image(self, 
                                image_path: str, 
                                metadata: Dict, 
                                output_path: Optional[str] = None) -> Dict:
        """
        Rectify a Street View panoramic or perspective image to prepare for projection.
        
        Args:
            image_path: Path to the Street View image
            metadata: Street View metadata containing camera parameters
            output_path: Path to save the rectified image (if None, saved to cache)
            
        Returns:
            Dictionary with rectification results
        """
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": f"Failed to load image: {image_path}"}
            
            # Get image dimensions
            h, w = image.shape[:2]
            
            # Determine if this is a panorama or perspective image
            is_panorama = 'heading' not in metadata or metadata.get('is_panorama', False)
            
            if not output_path:
                # Generate output filename if not provided
                base_name = Path(image_path).stem
                output_name = f"{base_name}_rectified.jpg"
                output_path = str(self.rectified_dir / output_name)
            
            if is_panorama:
                # For panoramic images, unwrap to multiple perspective views
                rectified_images = self._unwrap_panorama(image, metadata)
                
                # Save each rectified perspective view
                saved_paths = []
                for i, rect_img in enumerate(rectified_images):
                    heading = i * 90  # 0, 90, 180, 270 degrees
                    perspective_path = output_path.replace('.jpg', f'_{heading:03d}.jpg')
                    cv2.imwrite(perspective_path, rect_img)
                    saved_paths.append(perspective_path)
                
                return {
                    "success": True,
                    "is_panorama": True,
                    "rectified_paths": saved_paths,
                    "original_path": image_path,
                    "metadata": metadata
                }
            else:
                # For perspective images, correct for lens distortion
                heading = metadata.get('heading', 0)
                pitch = metadata.get('pitch', 0)
                roll = metadata.get('roll', 0)
                fov = metadata.get('fov', 90)
                
                # Create camera model for distortion correction
                camera_matrix = np.array([
                    [w / (2 * math.tan(math.radians(fov/2))), 0, w/2],
                    [0, w / (2 * math.tan(math.radians(fov/2))), h/2],
                    [0, 0, 1]
                ])
                
                # Estimate distortion parameters (typically not provided in Street View metadata)
                # Using conservative values for radial distortion
                dist_coeffs = np.array([0.1, -0.03, 0, 0, 0.01])
                
                # Undistort image
                rectified_img = cv2.undistort(image, camera_matrix, dist_coeffs)
                
                # Save rectified image
                cv2.imwrite(output_path, rectified_img)
                
                return {
                    "success": True,
                    "is_panorama": False,
                    "rectified_path": output_path,
                    "original_path": image_path,
                    "metadata": metadata,
                    "camera_params": {
                        "heading": heading,
                        "pitch": pitch,
                        "roll": roll,
                        "fov": fov
                    }
                }
        
        except Exception as e:
            logger.error(f"Error rectifying Street View image: {str(e)}")
            return {"success": False, "error": str(e)}

    def _unwrap_panorama(self, 
                       panorama_img: np.ndarray, 
                       metadata: Dict) -> List[np.ndarray]:
        """
        Unwrap a 360° panorama into multiple perspective views with corrected distortion.
        
        Args:
            panorama_img: Full 360° panorama image as numpy array
            metadata: Street View metadata with camera information
            
        Returns:
            List of perspective views (numpy arrays) at different headings
        """
        # Get panorama dimensions
        h, w = panorama_img.shape[:2]
        
        # Create perspective views at 90-degree intervals (0, 90, 180, 270)
        perspective_views = []
        fov = 90  # 90-degree field of view for each perspective
        
        for heading in [0, 90, 180, 270]:
            # Create perspective projection matrix
            # Compute the mapping from 3D point to panorama pixel
            perspective = np.zeros((h, h, 3), dtype=np.uint8)  # Square output image
            
            # Create meshgrid of target image coordinates
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(h))
            
            # Convert to normalized device coordinates (-1 to 1)
            x_ndc = (x_coords / (h - 1)) * 2 - 1
            y_ndc = (y_coords / (h - 1)) * 2 - 1
            
            # Calculate ray direction for each pixel
            fov_rad = math.radians(fov)
            x_world = x_ndc * math.tan(fov_rad / 2)
            y_world = y_ndc * math.tan(fov_rad / 2)
            z_world = np.ones_like(x_world)
            
            # Create ray direction vectors
            ray_dirs = np.stack([x_world, y_world, z_world], axis=-1)
            ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
            
            # Rotate ray directions based on heading
            heading_rad = math.radians(heading)
            rot_matrix = np.array([
                [math.cos(heading_rad), 0, math.sin(heading_rad)],
                [0, 1, 0],
                [-math.sin(heading_rad), 0, math.cos(heading_rad)]
            ])
            
            # Vectorized rotation of all rays
            rotated_rays = np.einsum('ijk,lk->ijl', ray_dirs, rot_matrix)
            
            # Convert ray directions to spherical coordinates
            # theta: azimuthal angle in panorama (0 to 2π)
            # phi: polar angle in panorama (0 to π)
            x, y, z = rotated_rays[:, :, 0], rotated_rays[:, :, 1], rotated_rays[:, :, 2]
            
            theta = np.arctan2(x, z) + math.pi  # Convert to range [0, 2π]
            phi = np.arccos(np.clip(y, -1.0, 1.0))
            
            # Map spherical coordinates to panorama image coordinates
            panorama_x = (theta / (2 * math.pi)) * w
            panorama_y = (phi / math.pi) * h
            
            # Sample the panorama (with bilinear interpolation)
            panorama_x = np.clip(panorama_x, 0, w - 1)
            panorama_y = np.clip(panorama_y, 0, h - 1)
            
            # Use OpenCV's remap for efficient interpolation
            map_x = panorama_x.astype(np.float32)
            map_y = panorama_y.astype(np.float32)
            perspective = cv2.remap(panorama_img, map_x, map_y, cv2.INTER_LINEAR)
            
            perspective_views.append(perspective)
        
        return perspective_views

    def project_image_to_mesh(self,
                            image_path: str,
                            mesh_path: str,
                            camera_params: Dict,
                            output_path: Optional[str] = None) -> Dict:
        """
        Project a rectified image onto a 3D mesh.
        
        Args:
            image_path: Path to the rectified image
            mesh_path: Path to the 3D mesh file (OBJ format)
            camera_params: Camera parameters for projection
            output_path: Path to save the projected texture (if None, saved to cache)
            
        Returns:
            Dictionary with projection results
        """
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": f"Failed to load image: {image_path}"}
            
            # Load the mesh
            mesh_vertices, mesh_faces, mesh_uvs = self._load_obj_mesh(mesh_path)
            if mesh_vertices is None:
                return {"success": False, "error": f"Failed to load mesh: {mesh_path}"}
            
            # Create camera object
            camera = CameraParameters(
                position=camera_params.get('position', (0, 0, 0)),
                orientation=camera_params.get('orientation', (0, 0, 0)),
                fov_horizontal=math.radians(camera_params.get('fov', 90)),
                fov_vertical=math.radians(camera_params.get('fov', 90) / camera_params.get('aspect', 1.0)),
                width=image.shape[1],
                height=image.shape[0]
            )
            
            # Generate output path if not provided
            if not output_path:
                base_name = f"{Path(image_path).stem}_{Path(mesh_path).stem}"
                output_path = str(self.projected_dir / f"{base_name}_projected.jpg")
            
            # Project the image onto the mesh and generate UV texture
            projection_result = self._project_to_uvs(
                image, mesh_vertices, mesh_faces, mesh_uvs, camera
            )
            
            if not projection_result["success"]:
                return projection_result
            
            # Save the projected texture
            projected_texture = projection_result["texture"]
            cv2.imwrite(output_path, projected_texture)
            
            # Update UVs if needed
            if projection_result.get("uvs_updated", False):
                new_mesh_path = str(Path(output_path).with_suffix(".obj"))
                self._save_obj_with_uvs(
                    mesh_path, new_mesh_path, 
                    mesh_vertices, mesh_faces, 
                    projection_result["updated_uvs"]
                )
                
                return {
                    "success": True,
                    "texture_path": output_path,
                    "mesh_path": new_mesh_path,
                    "uvs_updated": True,
                    "camera_params": camera_params,
                    "coverage_percentage": projection_result.get("coverage_percentage", 0)
                }
            
            return {
                "success": True,
                "texture_path": output_path,
                "mesh_path": mesh_path,
                "uvs_updated": False,
                "camera_params": camera_params,
                "coverage_percentage": projection_result.get("coverage_percentage", 0)
            }
            
        except Exception as e:
            logger.error(f"Error projecting image to mesh: {str(e)}")
            return {"success": False, "error": str(e)}

    def _load_obj_mesh(self, mesh_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load a mesh from an OBJ file.
        
        Args:
            mesh_path: Path to the OBJ file
            
        Returns:
            Tuple of (vertices, faces, uvs) as numpy arrays or None if loading fails
        """
        try:
            vertices = []
            uvs = []
            faces = []
            uv_indices = []
            
            with open(mesh_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):  # Skip comments
                        continue
                    
                    values = line.split()
                    if not values:
                        continue
                    
                    if values[0] == 'v':
                        # Vertex position
                        v = [float(x) for x in values[1:4]]
                        vertices.append(v)
                    elif values[0] == 'vt':
                        # Texture coordinates
                        vt = [float(values[1]), float(values[2])]
                        uvs.append(vt)
                    elif values[0] == 'f':
                        # Face indices
                        # Format could be: f v1 v2 v3 or f v1/vt1 v2/vt2 v3/vt3
                        # or f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                        face_vertex_indices = []
                        face_uv_indices = []
                        
                        for v in values[1:]:
                            w = v.split('/')
                            face_vertex_indices.append(int(w[0]) - 1)  # OBJ is 1-indexed
                            if len(w) > 1 and w[1]:  # Has UVs
                                face_uv_indices.append(int(w[1]) - 1)
                            else:
                                face_uv_indices.append(-1)  # No UV
                        
                        faces.append(face_vertex_indices)
                        uv_indices.append(face_uv_indices)
            
            # Convert to numpy arrays
            vertices_array = np.array(vertices, dtype=np.float32)
            faces_array = np.array(faces, dtype=np.int32)
            
            # Handle UVs - create a UV per vertex in face
            if not uvs:
                # No UVs in file, create default mapping
                uvs = np.zeros((len(vertices), 2), dtype=np.float32)
                uv_indices = faces  # Use same indices as vertices
            else:
                uvs = np.array(uvs, dtype=np.float32)
                uv_indices = np.array(uv_indices, dtype=np.int32)
            
            # Expand UVs to match the face organization (per-face UVs)
            expanded_uvs = []
            for face, uv_idx in zip(faces, uv_indices):
                face_uvs = []
                for i, vi in enumerate(face):
                    if uv_idx[i] >= 0:
                        face_uvs.append(uvs[uv_idx[i]])
                    else:
                        # No UV assigned, create a procedural UV based on position
                        v = vertices_array[vi]
                        # Simple planar mapping
                        face_uvs.append([
                            0.5 + v[0] / 20.0,  # Scale to avoid exceeding [0,1]
                            0.5 + v[2] / 20.0   # Using X,Z for top-down UV mapping
                        ])
                expanded_uvs.append(face_uvs)
            
            return vertices_array, faces_array, np.array(expanded_uvs, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error loading OBJ mesh: {str(e)}")
            return None, None, None

    def _project_to_uvs(self, 
                      image: np.ndarray, 
                      vertices: np.ndarray, 
                      faces: np.ndarray, 
                      uvs: np.ndarray, 
                      camera: CameraParameters) -> Dict:
        """
        Project an image onto mesh UVs to create a texture.
        
        Args:
            image: Image to project as a numpy array
            vertices: Mesh vertices as numpy array
            faces: Mesh faces as numpy array
            uvs: Mesh UV coordinates as numpy array
            camera: Camera parameters for projection
            
        Returns:
            Dictionary with projection results including the generated texture
        """
        # Create an empty texture (default black with alpha = 0)
        texture_size = 2048  # High resolution texture
        texture = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
        
        # Get camera matrices
        view_matrix = camera.view_matrix
        projection_matrix = camera.projection_matrix
        
        # Transform all vertices to clip space at once
        vertices_homogeneous = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
        view_vertices = vertices_homogeneous @ view_matrix.T
        clip_vertices = view_vertices @ projection_matrix.T
        
        # Normalize to NDC space by dividing by w (perspective division)
        w = clip_vertices[:, 3:4]
        ndc_vertices = clip_vertices[:, :3] / np.where(w != 0, w, 1.0)
        
        # Convert from NDC to screen space
        screen_vertices = np.zeros_like(ndc_vertices)
        screen_vertices[:, 0] = (ndc_vertices[:, 0] + 1) * 0.5 * camera.width
        screen_vertices[:, 1] = (1 - (ndc_vertices[:, 1] + 1) * 0.5) * camera.height
        screen_vertices[:, 2] = ndc_vertices[:, 2]  # Keep Z for depth testing
        
        # Process each face
        coverage_count = 0
        total_faces = len(faces)
        
        for i, (face, face_uvs) in enumerate(zip(faces, uvs)):
            # Get vertices for this face
            face_vertices = vertices[face]
            screen_face_vertices = screen_vertices[face]
            
            # Check if face is visible (not backfacing, within screen)
            if not self._is_face_visible(screen_face_vertices, face_vertices, camera):
                continue
            
            # Convert face to screen space triangle
            screen_triangle = screen_face_vertices[:, :2].astype(np.float32)
            
            # Get face UVs
            face_uv_coords = face_uvs * texture_size
            
            # Create a mask for this triangle
            mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, screen_triangle.astype(np.int32), 255)
            
            # Find pixels inside the triangle
            y_indices, x_indices = np.where(mask > 0)
            
            if len(y_indices) > 0:
                # For each pixel, find its barycentric coordinates
                points = np.column_stack([x_indices, y_indices]).astype(np.float32)
                
                # Calculate barycentric coordinates for all points simultaneously
                bary_coords = self._compute_barycentric_coords(
                    points, screen_triangle
                )
                
                # Use barycentric coordinates to interpolate UV coordinates
                interpolated_uvs = np.zeros((len(points), 2), dtype=np.float32)
                for j in range(3):  # For each triangle vertex
                    interpolated_uvs += bary_coords[:, j:j+1] * face_uv_coords[j]
                
                # Sample the image at these pixel locations
                valid_indices = np.logical_and(
                    np.logical_and(x_indices >= 0, x_indices < camera.width),
                    np.logical_and(y_indices >= 0, y_indices < camera.height)
                )
                
                if np.any(valid_indices):
                    # Get colors from the image
                    colors = image[y_indices[valid_indices], x_indices[valid_indices]]
                    
                    # Convert UV coordinates to texture pixel coordinates
                    tex_x = np.clip(interpolated_uvs[valid_indices, 0], 0, texture_size - 1).astype(np.int32)
                    tex_y = np.clip(interpolated_uvs[valid_indices, 1], 0, texture_size - 1).astype(np.int32)
                    
                    # Set colors in texture (with full alpha)
                    texture[tex_y, tex_x, :3] = colors
                    texture[tex_y, tex_x, 3] = 255  # Full alpha
                    
                    coverage_count += 1
        
        # Calculate coverage percentage
        coverage_percentage = (coverage_count / total_faces) * 100 if total_faces > 0 else 0
        
        return {
            "success": True, 
            "texture": texture, 
            "uvs_updated": False,
            "coverage_percentage": coverage_percentage
        }

    def _is_face_visible(self, screen_vertices: np.ndarray, world_vertices: np.ndarray, camera: CameraParameters) -> bool:
        """
        Determine if a face is visible to the camera.
        
        Args:
            screen_vertices: Face vertices in screen space
            world_vertices: Face vertices in world space
            camera: Camera parameters
            
        Returns:
            True if the face is visible, False otherwise
        """
        # Check if any vertex is inside the view frustum
        in_screen = np.logical_and(
            np.logical_and(
                screen_vertices[:, 0] >= 0, 
                screen_vertices[:, 0] < camera.width
            ),
            np.logical_and(
                screen_vertices[:, 1] >= 0, 
                screen_vertices[:, 1] < camera.height
            )
        )
        
        if not np.any(in_screen):
            return False
        
        # Calculate face normal in world space
        v0, v1, v2 = world_vertices[0], world_vertices[1], world_vertices[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)
        
        # Calculate view direction from camera to face center
        face_center = np.mean(world_vertices, axis=0)
        view_dir = face_center - np.array(camera.position)
        view_dir = view_dir / np.linalg.norm(view_dir)
        
        # Check if face is facing the camera (dot product < 0)
        dot_product = np.dot(normal, view_dir)
        return dot_product < 0

    def _compute_barycentric_coords(self, points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
        """
        Compute barycentric coordinates for points inside a triangle.
        
        Args:
            points: Array of points to compute coordinates for
            triangle: Triangle vertices
            
        Returns:
            Array of barycentric coordinates for each point
        """
        # Extract triangle vertices
        a, b, c = triangle[0], triangle[1], triangle[2]
        
        # Compute vectors from point to vertices
        v0 = b - a
        v1 = c - a
        v2 = points - a.reshape(1, 2)
        
        # Compute dot products
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.einsum('ij,j->i', v2, v0)
        d21 = np.einsum('ij,j->i', v2, v1)
        
        # Compute barycentric coordinates
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return np.column_stack([u, v, w])

    def _save_obj_with_uvs(self, 
                         original_path: str, 
                         output_path: str, 
                         vertices: np.ndarray, 
                         faces: np.ndarray, 
                         uvs: np.ndarray) -> bool:
        """
        Save a mesh with updated UVs to an OBJ file.
        
        Args:
            original_path: Path to the original OBJ file
            output_path: Path to save the updated OBJ file
            vertices: Mesh vertices
            faces: Mesh faces
            uvs: Updated UV coordinates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Flatten face-based UVs to a list of unique UVs
            flattened_uvs = []
            uv_indices = []
            
            for face_idx, face in enumerate(faces):
                face_uv_indices = []
                for vertex_idx, uv in zip(face, uvs[face_idx]):
                    # Check if this UV already exists
                    existing_idx = None
                    for i, existing_uv in enumerate(flattened_uvs):
                        if np.allclose(existing_uv, uv, atol=1e-5):
                            existing_idx = i
                            break
                    
                    if existing_idx is not None:
                        face_uv_indices.append(existing_idx)
                    else:
                        face_uv_indices.append(len(flattened_uvs))
                        flattened_uvs.append(uv)
                
                uv_indices.append(face_uv_indices)
            
            # Write the new OBJ file
            with open(output_path, 'w') as f:
                # Write header
                f.write(f"# OBJ file with updated UVs\n")
                f.write(f"# Original: {original_path}\n")
                f.write(f"# Generated by Arcanum TextureProjector\n\n")
                
                # Write vertices
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                # Write texture coordinates
                for vt in flattened_uvs:
                    # Normalize UVs to 0-1 range if they're in texture space
                    normalized_u = vt[0] / 2048.0 if vt[0] > 1.0 else vt[0]
                    normalized_v = vt[1] / 2048.0 if vt[1] > 1.0 else vt[1]
                    f.write(f"vt {normalized_u:.6f} {normalized_v:.6f}\n")
                
                # Write faces with UV indices
                for face, uv_idx in zip(faces, uv_indices):
                    f.write("f")
                    for v_idx, vt_idx in zip(face, uv_idx):
                        f.write(f" {v_idx+1}/{vt_idx+1}")  # +1 because OBJ is 1-indexed
                    f.write("\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving OBJ with UVs: {str(e)}")
            return False

    def blend_projected_textures(self, 
                               texture_paths: List[str], 
                               weights: Optional[List[float]] = None,
                               output_path: Optional[str] = None) -> Dict:
        """
        Blend multiple projected textures into a single high-quality texture.
        
        Args:
            texture_paths: List of paths to projected textures
            weights: Optional list of weights for each texture (if None, calculated based on coverage)
            output_path: Path to save the blended texture (if None, saved to cache)
            
        Returns:
            Dictionary with blending results
        """
        try:
            if not texture_paths:
                return {"success": False, "error": "No textures provided for blending"}
            
            # Load all textures
            textures = []
            for path in texture_paths:
                texture = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if texture is None:
                    logger.warning(f"Failed to load texture: {path}")
                    continue
                
                # Ensure all textures have alpha channel
                if texture.shape[2] == 3:
                    # Add alpha channel (fully opaque where color exists)
                    alpha = np.ones((texture.shape[0], texture.shape[1], 1), dtype=np.uint8) * 255
                    texture = np.concatenate([texture, alpha], axis=2)
                
                textures.append(texture)
            
            if not textures:
                return {"success": False, "error": "Failed to load any textures"}
            
            # Make sure all textures have the same dimensions
            texture_size = textures[0].shape[:2]
            for i, texture in enumerate(textures):
                if texture.shape[:2] != texture_size:
                    textures[i] = cv2.resize(texture, (texture_size[1], texture_size[0]))
            
            # Calculate weights if not provided
            if weights is None:
                weights = []
                for texture in textures:
                    # Calculate coverage (percentage of non-zero alpha pixels)
                    alpha = texture[:, :, 3]
                    coverage = np.count_nonzero(alpha) / (alpha.shape[0] * alpha.shape[1])
                    
                    # Calculate average edge strength as a quality metric
                    grayscale = cv2.cvtColor(texture[:, :, :3], cv2.COLOR_BGR2GRAY)
                    edge_strength = cv2.Laplacian(grayscale, cv2.CV_64F).var()
                    
                    # Combined weight based on coverage and edge quality
                    weight = coverage * (1.0 + 0.5 * edge_strength / 1000.0)
                    weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                # Equal weights if all weights are zero
                weights = [1.0 / len(textures)] * len(textures)
            
            # Create empty blended texture
            blended_texture = np.zeros_like(textures[0], dtype=np.float32)
            
            # Blend textures using weighted average with alpha
            for texture, weight in zip(textures, weights):
                # Convert to float for calculations
                texture_float = texture.astype(np.float32)
                
                # Get alpha channel normalized to 0-1
                alpha = texture_float[:, :, 3:4] / 255.0
                
                # Apply weighting to alpha
                weighted_alpha = alpha * weight
                
                # Apply alpha to color channels
                weighted_texture = texture_float * weighted_alpha
                
                # Add to blended texture
                blended_texture += weighted_texture
            
            # Generate output path if not provided
            if not output_path:
                # Create a name based on the first texture
                base_name = Path(texture_paths[0]).stem.split('_')[0]
                output_path = str(self.blended_dir / f"{base_name}_blended.png")
            
            # Convert back to uint8 after blending
            blended_texture = np.clip(blended_texture, 0, 255).astype(np.uint8)
            
            # Save blended texture
            cv2.imwrite(output_path, blended_texture)
            
            return {
                "success": True,
                "blended_path": output_path,
                "input_textures": len(textures),
                "weights": weights
            }
            
        except Exception as e:
            logger.error(f"Error blending textures: {str(e)}")
            return {"success": False, "error": str(e)}

    def process_building(self, 
                       building_id: str,
                       mesh_path: str,
                       street_view_images: List[Dict],
                       metadata: Dict,
                       output_dir: Optional[str] = None) -> Dict:
        """
        Process a building by projecting Street View images and generating textures.
        
        Args:
            building_id: Unique identifier for the building
            mesh_path: Path to the building's 3D mesh
            street_view_images: List of dictionaries with Street View image info
            metadata: Building metadata
            output_dir: Directory to save results (if None, uses default)
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Create output directory
            if output_dir:
                output_dir = Path(output_dir)
            else:
                output_dir = self.output_dir / building_id
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Rectify all Street View images
            rectified_results = []
            
            # Use parallel processing if enabled
            if self.parallel_processing:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for img_info in street_view_images:
                        future = executor.submit(
                            self.rectify_street_view_image,
                            img_info["path"],
                            img_info["metadata"]
                        )
                        futures.append(future)
                    
                    # Collect results as they complete
                    for future in as_completed(futures):
                        rectified_results.append(future.result())
            else:
                # Sequential processing
                for img_info in street_view_images:
                    result = self.rectify_street_view_image(
                        img_info["path"],
                        img_info["metadata"]
                    )
                    rectified_results.append(result)
            
            # Extract all rectified paths
            all_rectified_paths = []
            for result in rectified_results:
                if result.get("success", False):
                    if result.get("is_panorama", False):
                        all_rectified_paths.extend(result.get("rectified_paths", []))
                    else:
                        all_rectified_paths.append(result.get("rectified_path"))
            
            if not all_rectified_paths:
                return {"success": False, "error": "Failed to rectify any images"}
            
            # Step 2: Project images onto the mesh
            projection_results = []
            
            # Get building location from metadata
            building_location = metadata.get("center", [0, 0])
            
            # Process each rectified image
            if self.parallel_processing:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for i, img_path in enumerate(all_rectified_paths):
                        # Calculate camera parameters based on image metadata
                        # In a real implementation, these would come from Street View metadata
                        camera_params = self._estimate_camera_params(
                            img_path, building_location, i
                        )
                        
                        # Create output path
                        proj_name = f"{building_id}_projection_{i:02d}.png"
                        proj_path = str(output_dir / proj_name)
                        
                        future = executor.submit(
                            self.project_image_to_mesh,
                            img_path,
                            mesh_path,
                            camera_params,
                            proj_path
                        )
                        futures.append(future)
                    
                    # Collect results as they complete
                    for future in as_completed(futures):
                        projection_results.append(future.result())
            else:
                # Sequential processing
                for i, img_path in enumerate(all_rectified_paths):
                    # Calculate camera parameters based on image metadata
                    camera_params = self._estimate_camera_params(
                        img_path, building_location, i
                    )
                    
                    # Create output path
                    proj_name = f"{building_id}_projection_{i:02d}.png"
                    proj_path = str(output_dir / proj_name)
                    
                    result = self.project_image_to_mesh(
                        img_path,
                        mesh_path,
                        camera_params,
                        proj_path
                    )
                    projection_results.append(result)
            
            # Extract successful projections
            successful_projections = [
                result.get("texture_path") for result in projection_results
                if result.get("success", False)
            ]
            
            if not successful_projections:
                return {"success": False, "error": "Failed to project any images onto the mesh"}
            
            # Step 3: Blend projected textures
            blended_path = str(output_dir / f"{building_id}_material.png")
            blend_result = self.blend_projected_textures(
                successful_projections,
                output_path=blended_path
            )
            
            if not blend_result.get("success", False):
                return {"success": False, "error": "Failed to blend projected textures"}
            
            # Step 4: Create material with the blended texture
            material_path = str(output_dir / f"{building_id}_material.mat")
            
            # Simple material file format for Unity
            material_content = f"""
            %YAML 1.1
            %TAG !u! tag:unity3d.com,2011:
            --- !u!21 &2100000
            Material:
              serializedVersion: 6
              m_ObjectHideFlags: 0
              m_CorrespondingSourceObject: {{fileID: 0}}
              m_PrefabInstance: {{fileID: 0}}
              m_PrefabAsset: {{fileID: 0}}
              m_Name: {building_id}_material
              m_Shader: {{fileID: 46, guid: 0000000000000000f000000000000000, type: 0}}
              m_ShaderKeywords: _NORMALMAP
              m_LightmapFlags: 4
              m_EnableInstancingVariants: 0
              m_DoubleSidedGI: 0
              m_CustomRenderQueue: -1
              stringTagMap: {{}}
              disabledShaderPasses: []
              m_SavedProperties:
                serializedVersion: 3
                m_TexEnvs:
                - _BumpMap:
                    m_Texture: {{fileID: 2800000, guid: 00000000000000000000000000000000, type: 0}}
                    m_Scale: {{x: 1, y: 1}}
                    m_Offset: {{x: 0, y: 0}}
                - _DetailAlbedoMap:
                    m_Texture: {{fileID: 0}}
                    m_Scale: {{x: 1, y: 1}}
                    m_Offset: {{x: 0, y: 0}}
                - _DetailMask:
                    m_Texture: {{fileID: 0}}
                    m_Scale: {{x: 1, y: 1}}
                    m_Offset: {{x: 0, y: 0}}
                - _DetailNormalMap:
                    m_Texture: {{fileID: 0}}
                    m_Scale: {{x: 1, y: 1}}
                    m_Offset: {{x: 0, y: 0}}
                - _EmissionMap:
                    m_Texture: {{fileID: 0}}
                    m_Scale: {{x: 1, y: 1}}
                    m_Offset: {{x: 0, y: 0}}
                - _MainTex:
                    m_Texture: {{fileID: 2800000, guid: 00000000000000000000000000000000, type: 0}}
                    m_Scale: {{x: 1, y: 1}}
                    m_Offset: {{x: 0, y: 0}}
                - _MetallicGlossMap:
                    m_Texture: {{fileID: 0}}
                    m_Scale: {{x: 1, y: 1}}
                    m_Offset: {{x: 0, y: 0}}
                - _OcclusionMap:
                    m_Texture: {{fileID: 0}}
                    m_Scale: {{x: 1, y: 1}}
                    m_Offset: {{x: 0, y: 0}}
                - _ParallaxMap:
                    m_Texture: {{fileID: 0}}
                    m_Scale: {{x: 1, y: 1}}
                    m_Offset: {{x: 0, y: 0}}
                m_Floats:
                - _BumpScale: 1
                - _Cutoff: 0.5
                - _DetailNormalMapScale: 1
                - _DstBlend: 0
                - _GlossMapScale: 1
                - _Glossiness: 0.5
                - _GlossyReflections: 1
                - _Metallic: 0.1
                - _Mode: 0
                - _OcclusionStrength: 1
                - _Parallax: 0.02
                - _SmoothnessTextureChannel: 0
                - _SpecularHighlights: 1
                - _SrcBlend: 1
                - _UVSec: 0
                - _ZWrite: 1
                m_Colors:
                - _Color: {{r: 1, g: 1, b: 1, a: 1}}
                - _EmissionColor: {{r: 0, g: 0, b: 0, a: 1}}
            """
            
            with open(material_path, 'w') as f:
                f.write(material_content.strip())
            
            # Create metadata file with projection information
            metadata_path = str(output_dir / f"{building_id}_projection_metadata.json")
            
            final_metadata = {
                "building_id": building_id,
                "mesh_path": mesh_path,
                "material_path": material_path,
                "texture_path": blended_path,
                "street_view_images": [img["path"] for img in street_view_images],
                "rectified_images": all_rectified_paths,
                "projected_textures": successful_projections,
                "projection_coverage": [
                    {
                        "texture": result.get("texture_path", ""),
                        "coverage": result.get("coverage_percentage", 0)
                    }
                    for result in projection_results if result.get("success", False)
                ]
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(final_metadata, f, indent=2)
            
            return {
                "success": True,
                "building_id": building_id,
                "mesh_path": mesh_path,
                "material_path": material_path,
                "texture_path": blended_path,
                "metadata_path": metadata_path,
                "rectified_count": len(all_rectified_paths),
                "projection_count": len(successful_projections)
            }
            
        except Exception as e:
            logger.error(f"Error processing building {building_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def _estimate_camera_params(self, img_path: str, building_location: List[float], index: int) -> Dict:
        """
        Estimate camera parameters for an image based on building location.
        
        Args:
            img_path: Path to the image
            building_location: [latitude, longitude] of building
            index: Index of the image in the sequence
            
        Returns:
            Dictionary with camera parameters
        """
        # Extract heading from filename if available
        heading = 0
        img_name = Path(img_path).stem
        heading_match = re.search(r'_(\d{3})\.', img_path)
        if heading_match:
            heading = int(heading_match.group(1))
        
        # Camera parameters for a typical street view setup
        # In a production system, these would come from the Street View metadata
        
        # Calculate camera position (simulated street view position)
        # Positions camera 10-20m away from building at different angles
        distance = 15.0  # meters from building
        angle_rad = math.radians(heading)
        
        # Convert lat/lon to approximate x/y (flat Earth approximation)
        building_x = 0
        building_y = 0
        
        # Position the camera around the building
        camera_x = building_x + distance * math.sin(angle_rad)
        camera_z = building_y + distance * math.cos(angle_rad)
        camera_y = DEFAULT_CAMERA_HEIGHT  # Street view height
        
        # Calculate orientation (looking at the building)
        pitch = 0  # Level camera
        yaw = (heading + 180) % 360  # Look toward building
        roll = 0  # No roll
        
        # Convert to radians for internal calculations
        orientation = (
            math.radians(roll),
            math.radians(pitch),
            math.radians(yaw)
        )
        
        return {
            "position": (camera_x, camera_y, camera_z),
            "orientation": orientation,
            "fov": 90,  # Typical street view fov
            "aspect": 1.0,  # Assume square image for simplicity
            "heading": heading,
            "pitch": pitch,
            "roll": roll
        }

    def process_building_batch(self, 
                             buildings_data: Dict[str, Dict],
                             street_view_dir: str,
                             output_dir: Optional[str] = None) -> Dict:
        """
        Process a batch of buildings with Street View imagery.
        
        Args:
            buildings_data: Dictionary mapping building IDs to metadata
            street_view_dir: Directory containing Street View images
            output_dir: Directory to save processed results
            
        Returns:
            Dictionary with batch processing results
        """
        try:
            # Create output directory
            if output_dir:
                output_dir = Path(output_dir)
            else:
                output_dir = self.output_dir / "buildings"
                
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Track results
            results = {
                "success_count": 0,
                "failed_count": 0,
                "buildings": {},
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Process each building
            for building_id, metadata in buildings_data.items():
                # Find mesh file for this building
                mesh_path = metadata.get("mesh_path", "")
                if not mesh_path or not os.path.exists(mesh_path):
                    # Try to find mesh in standard location
                    potential_paths = [
                        f"../models/{building_id}.obj",
                        f"../3d_models/buildings/{building_id}.obj",
                        f"models/{building_id}.obj",
                        f"3d_models/buildings/{building_id}.obj"
                    ]
                    
                    for path in potential_paths:
                        if os.path.exists(path):
                            mesh_path = path
                            break
                
                if not mesh_path or not os.path.exists(mesh_path):
                    logger.warning(f"No mesh found for building {building_id}")
                    results["failed_count"] += 1
                    results["buildings"][building_id] = {
                        "success": False,
                        "error": "No mesh found"
                    }
                    continue
                
                # Find Street View images for this building
                # Look in directories that might contain the building's images
                sv_images = []
                
                # Building-specific directory
                building_sv_dir = os.path.join(street_view_dir, building_id)
                if os.path.exists(building_sv_dir):
                    for file in os.listdir(building_sv_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            sv_path = os.path.join(building_sv_dir, file)
                            metadata_path = os.path.join(building_sv_dir, "metadata.json")
                            
                            if os.path.exists(metadata_path):
                                with open(metadata_path, 'r') as f:
                                    sv_metadata = json.load(f)
                                    sv_images.append({
                                        "path": sv_path,
                                        "metadata": sv_metadata
                                    })
                            else:
                                # Create basic metadata if missing
                                heading = 0
                                heading_match = re.search(r'_(\d{3})\.', file)
                                if heading_match:
                                    heading = int(heading_match.group(1))
                                
                                sv_images.append({
                                    "path": sv_path,
                                    "metadata": {
                                        "heading": heading,
                                        "pitch": 0,
                                        "fov": 90
                                    }
                                })
                
                # Main street view directory
                if not sv_images:
                    for file in os.listdir(street_view_dir):
                        if file.lower().startswith(building_id) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            sv_path = os.path.join(street_view_dir, file)
                            
                            # Create basic metadata
                            heading = 0
                            heading_match = re.search(r'_(\d{3})\.', file)
                            if heading_match:
                                heading = int(heading_match.group(1))
                            
                            sv_images.append({
                                "path": sv_path,
                                "metadata": {
                                    "heading": heading,
                                    "pitch": 0,
                                    "fov": 90
                                }
                            })
                
                if not sv_images:
                    logger.warning(f"No Street View images found for building {building_id}")
                    results["failed_count"] += 1
                    results["buildings"][building_id] = {
                        "success": False,
                        "error": "No Street View images found"
                    }
                    continue
                
                # Process this building
                building_output_dir = output_dir / building_id
                building_result = self.process_building(
                    building_id,
                    mesh_path,
                    sv_images,
                    metadata,
                    building_output_dir
                )
                
                if building_result.get("success", False):
                    results["success_count"] += 1
                    results["buildings"][building_id] = building_result
                else:
                    results["failed_count"] += 1
                    results["buildings"][building_id] = {
                        "success": False,
                        "error": building_result.get("error", "Unknown error")
                    }
            
            # Save batch results
            results_path = output_dir / "batch_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Batch processing complete. Processed {len(buildings_data)} buildings.")
            logger.info(f"Success: {results['success_count']}, Failed: {results['failed_count']}")
            
            return {
                "success": True,
                "results_path": str(results_path),
                "total_buildings": len(buildings_data),
                "success_count": results["success_count"],
                "failed_count": results["failed_count"]
            }
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return {"success": False, "error": str(e)}


# Module-level function for direct import
def project_street_view_to_building(building_id: str, 
                                  mesh_path: str, 
                                  street_view_dir: str, 
                                  output_dir: str,
                                  metadata: Optional[Dict] = None) -> Dict:
    """
    Project Street View images onto a building mesh.
    
    Args:
        building_id: Unique identifier for the building
        mesh_path: Path to the building's 3D mesh
        street_view_dir: Directory containing Street View images
        output_dir: Directory to save processed results
        metadata: Optional building metadata
        
    Returns:
        Dictionary with processing results
    """
    # Create texture projector
    projector = TextureProjector(output_dir=output_dir)
    
    # Find all Street View images for this building
    sv_images = []
    
    # Check if street_view_dir is a directory or a file
    if os.path.isdir(street_view_dir):
        # Look for images in the directory
        for file in os.listdir(street_view_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sv_path = os.path.join(street_view_dir, file)
                metadata_path = os.path.join(street_view_dir, "metadata.json")
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        sv_metadata = json.load(f)
                        sv_images.append({
                            "path": sv_path,
                            "metadata": sv_metadata
                        })
                else:
                    # Create basic metadata if missing
                    heading = 0
                    heading_match = re.search(r'_(\d{3})\.', file)
                    if heading_match:
                        heading = int(heading_match.group(1))
                    
                    sv_images.append({
                        "path": sv_path,
                        "metadata": {
                            "heading": heading,
                            "pitch": 0,
                            "fov": 90
                        }
                    })
    else:
        # Treat as a single file
        if street_view_dir.lower().endswith(('.jpg', '.jpeg', '.png')):
            sv_path = street_view_dir
            
            # Create basic metadata
            heading = 0
            heading_match = re.search(r'_(\d{3})\.', os.path.basename(sv_path))
            if heading_match:
                heading = int(heading_match.group(1))
            
            sv_images.append({
                "path": sv_path,
                "metadata": {
                    "heading": heading,
                    "pitch": 0,
                    "fov": 90
                }
            })
    
    if not sv_images:
        return {"success": False, "error": "No Street View images found"}
    
    # Create default metadata if none provided
    if metadata is None:
        metadata = {
            "id": building_id,
            "center": [0, 0]  # Default center
        }
    
    # Process the building
    return projector.process_building(
        building_id,
        mesh_path,
        sv_images,
        metadata,
        output_dir
    )


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Texture Projection System")
    
    # Define subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single building processing
    building_parser = subparsers.add_parser("building", help="Process a single building")
    building_parser.add_argument("--id", required=True, help="Building ID")
    building_parser.add_argument("--mesh", required=True, help="Path to building mesh")
    building_parser.add_argument("--images", required=True, help="Path to Street View images")
    building_parser.add_argument("--output", default="./texture_output", help="Output directory")
    
    # Batch processing
    batch_parser = subparsers.add_parser("batch", help="Process a batch of buildings")
    batch_parser.add_argument("--buildings", required=True, help="Path to buildings metadata JSON")
    batch_parser.add_argument("--images", required=True, help="Path to Street View images")
    batch_parser.add_argument("--output", default="./texture_output", help="Output directory")
    batch_parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    batch_parser.add_argument("--workers", type=int, help="Number of worker processes")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "building":
        # Process single building
        result = project_street_view_to_building(
            args.id,
            args.mesh,
            args.images,
            args.output
        )
        
        if result.get("success", False):
            print(f"Successfully processed building {args.id}")
            print(f"Material: {result.get('material_path')}")
            print(f"Texture: {result.get('texture_path')}")
        else:
            print(f"Failed to process building: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.command == "batch":
        # Load buildings data
        try:
            with open(args.buildings, 'r') as f:
                buildings_data = json.load(f)
        except Exception as e:
            print(f"Error loading buildings data: {str(e)}")
            return 1
        
        # Create texture projector
        projector = TextureProjector(
            output_dir=args.output,
            parallel_processing=args.parallel,
            max_workers=args.workers
        )
        
        # Process buildings
        result = projector.process_building_batch(
            buildings_data,
            args.images,
            args.output
        )
        
        if result.get("success", False):
            print(f"Successfully processed {result.get('success_count')} of {result.get('total_buildings')} buildings")
            print(f"Results saved to {result.get('results_path')}")
        else:
            print(f"Batch processing failed: {result.get('error', 'Unknown error')}")
            return 1
    
    else:
        # No command specified
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())