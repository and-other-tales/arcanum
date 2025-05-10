#!/usr/bin/env python3
"""
Coverage Verification Module
----------------------------
This module provides functionality to verify and analyze the Street View coverage
along road networks in the Arcanum city generation system.
"""

import os
import sys
import json
import logging
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import visualization module for coverage visualization
try:
    from modules.visualization.preview import visualize_road_network, visualize_street_view_coverage
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Visualization module not available, coverage visualization will be limited")
    VISUALIZATION_AVAILABLE = False

class CoverageVerifier:
    """Class for verifying and analyzing Street View coverage along road networks."""
    
    def __init__(self, coverage_threshold: float = 50.0, max_gap_distance: float = 30.0):
        """
        Initialize the coverage verifier.
        
        Args:
            coverage_threshold: Minimum percentage coverage required (0-100)
            max_gap_distance: Maximum allowed distance (meters) between Street View points
        """
        self.coverage_threshold = coverage_threshold
        self.max_gap_distance = max_gap_distance
        logger.info(f"CoverageVerifier initialized with coverage threshold: {coverage_threshold}%, max gap: {max_gap_distance}m")
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Distance between the points
        """
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def analyze_edge_coverage(self, edge: Dict[str, Any], start_node: Dict[str, Any], 
                            end_node: Dict[str, Any], coverage_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze coverage for a single road edge.
        
        Args:
            edge: Edge data with start and end node IDs
            start_node: Start node data with coordinates
            end_node: End node data with coordinates
            coverage_points: List of Street View coverage points
            
        Returns:
            Dictionary with coverage analysis results
        """
        # Get coordinates
        start_x, start_y = start_node.get("x", 0), start_node.get("y", 0)
        end_x, end_y = end_node.get("x", 0), end_node.get("y", 0)
        
        # Calculate edge length
        edge_length = self.calculate_distance((start_x, start_y), (end_x, end_y))
        
        # If edge is too short, consider it fully covered
        if edge_length < 1.0:
            return {
                "edge_id": edge.get("id", "unknown"),
                "start_node_id": edge.get("start_node_id", "unknown"),
                "end_node_id": edge.get("end_node_id", "unknown"),
                "edge_length": edge_length,
                "covered_length": edge_length,
                "coverage_percentage": 100.0,
                "is_covered": True,
                "gaps": []
            }
        
        # Find points close to this edge
        edge_points = []
        
        # Project each coverage point onto the edge line
        for point in coverage_points:
            x, y = point.get("x", 0), point.get("y", 0)
            status = point.get("status", "unknown")
            
            # Skip failed points
            if status == "failed":
                continue
            
            # Calculate distance to edge using point-to-line distance formula
            # Distance from point to line segment: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            
            # Line segment vector
            dx = end_x - start_x
            dy = end_y - start_y
            
            # Point to start vector
            px = x - start_x
            py = y - start_y
            
            # Calculate dot product
            dot = px * dx + py * dy
            
            # Calculate projection length
            proj_length = (dot / (dx * dx + dy * dy)) if (dx * dx + dy * dy) > 0 else 0
            
            # Clamp to line segment
            proj_length = max(0, min(1, proj_length))
            
            # Calculate projected point
            proj_x = start_x + proj_length * dx
            proj_y = start_y + proj_length * dy
            
            # Calculate distance to line
            dist_to_line = self.calculate_distance((x, y), (proj_x, proj_y))
            
            # Distance along the edge (normalized 0-1)
            distance_along_edge = proj_length
            
            # If point is close enough to the edge (within 10 meters)
            if dist_to_line <= 10.0:
                edge_points.append({
                    "point": point,
                    "distance_to_edge": dist_to_line,
                    "distance_along_edge": distance_along_edge,
                    "position_along_edge": distance_along_edge * edge_length,
                    "projected_x": proj_x,
                    "projected_y": proj_y
                })
        
        # Sort points by position along edge
        edge_points.sort(key=lambda p: p["distance_along_edge"])
        
        # Calculate coverage using segments
        covered_segments = []
        
        # Consider start of edge
        if edge_points and edge_points[0]["distance_along_edge"] > 0:
            # First point not at start, check if close enough
            if edge_points[0]["position_along_edge"] <= self.max_gap_distance:
                # Close enough to start, include segment from start to first point
                covered_segments.append((0, edge_points[0]["position_along_edge"]))
        
        # Process consecutive points
        for i in range(len(edge_points) - 1):
            point1 = edge_points[i]
            point2 = edge_points[i + 1]
            
            # Calculate gap between points
            gap = point2["position_along_edge"] - point1["position_along_edge"]
            
            # If gap is small enough, consider segment covered
            if gap <= self.max_gap_distance:
                covered_segments.append((point1["position_along_edge"], point2["position_along_edge"]))
            else:
                # Large gap, record it
                # We still include the points themselves as covered
                covered_segments.append((point1["position_along_edge"], point1["position_along_edge"]))
                covered_segments.append((point2["position_along_edge"], point2["position_along_edge"]))
        
        # Consider end of edge
        if edge_points and edge_points[-1]["distance_along_edge"] < 1:
            # Last point not at end, check if close enough
            distance_to_end = edge_length - edge_points[-1]["position_along_edge"]
            if distance_to_end <= self.max_gap_distance:
                # Close enough to end, include segment from last point to end
                covered_segments.append((edge_points[-1]["position_along_edge"], edge_length))
        
        # Merge overlapping segments
        if covered_segments:
            covered_segments.sort()
            merged_segments = [covered_segments[0]]
            
            for current in covered_segments[1:]:
                prev = merged_segments[-1]
                
                # If current segment overlaps with previous one, merge them
                if current[0] <= prev[1]:
                    merged_segments[-1] = (prev[0], max(prev[1], current[1]))
                else:
                    merged_segments.append(current)
            
            covered_segments = merged_segments
        
        # Calculate total covered length
        covered_length = sum(segment[1] - segment[0] for segment in covered_segments)
        
        # Calculate coverage percentage
        coverage_percentage = (covered_length / edge_length) * 100
        
        # Identify gaps
        gaps = []
        
        if covered_segments:
            # Add gap at start if needed
            if covered_segments[0][0] > 0:
                gaps.append({
                    "start_position": 0,
                    "end_position": covered_segments[0][0],
                    "length": covered_segments[0][0]
                })
            
            # Add gaps between segments
            for i in range(len(covered_segments) - 1):
                gap_start = covered_segments[i][1]
                gap_end = covered_segments[i + 1][0]
                
                gaps.append({
                    "start_position": gap_start,
                    "end_position": gap_end,
                    "length": gap_end - gap_start
                })
            
            # Add gap at end if needed
            if covered_segments[-1][1] < edge_length:
                gaps.append({
                    "start_position": covered_segments[-1][1],
                    "end_position": edge_length,
                    "length": edge_length - covered_segments[-1][1]
                })
        else:
            # No coverage at all
            gaps.append({
                "start_position": 0,
                "end_position": edge_length,
                "length": edge_length
            })
        
        # Check if edge is sufficiently covered
        is_covered = coverage_percentage >= self.coverage_threshold
        
        # Return coverage analysis results
        return {
            "edge_id": edge.get("id", "unknown"),
            "start_node_id": edge.get("start_node_id", "unknown"),
            "end_node_id": edge.get("end_node_id", "unknown"),
            "edge_length": edge_length,
            "covered_length": covered_length,
            "coverage_percentage": coverage_percentage,
            "is_covered": is_covered,
            "gaps": gaps,
            "street_view_points": len(edge_points)
        }
    
    def verify_coverage(self, road_network_data: Dict[str, Any], 
                      coverage_data: List[Dict[str, Any]],
                      parallel: bool = True,
                      max_workers: int = 4) -> Dict[str, Any]:
        """
        Verify Street View coverage for an entire road network.
        
        Args:
            road_network_data: Dictionary with road network nodes and edges
            coverage_data: List of Street View points with coordinates and metadata
            parallel: Whether to process edges in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary with coverage verification results
        """
        # Extract nodes and edges
        nodes = road_network_data.get("nodes", {})
        edges = road_network_data.get("edges", [])
        
        logger.info(f"Verifying coverage for {len(edges)} edges with {len(coverage_data)} Street View points")
        
        # Process edges in parallel if requested
        edge_results = []
        
        if parallel and len(edges) > 10:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for edge in edges:
                    start_node_id = edge.get("start_node_id")
                    end_node_id = edge.get("end_node_id")
                    
                    # Skip if missing node references
                    if not start_node_id or not end_node_id:
                        continue
                    
                    # Get node data
                    start_node = nodes.get(start_node_id)
                    end_node = nodes.get(end_node_id)
                    
                    # Skip if nodes don't exist
                    if not start_node or not end_node:
                        continue
                    
                    # Submit task
                    future = executor.submit(
                        self.analyze_edge_coverage,
                        edge,
                        start_node,
                        end_node,
                        coverage_data
                    )
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    try:
                        result = future.result()
                        edge_results.append(result)
                    except Exception as e:
                        logger.error(f"Error analyzing edge coverage: {str(e)}")
        else:
            # Process edges sequentially
            for edge in edges:
                start_node_id = edge.get("start_node_id")
                end_node_id = edge.get("end_node_id")
                
                # Skip if missing node references
                if not start_node_id or not end_node_id:
                    continue
                
                # Get node data
                start_node = nodes.get(start_node_id)
                end_node = nodes.get(end_node_id)
                
                # Skip if nodes don't exist
                if not start_node or not end_node:
                    continue
                
                try:
                    result = self.analyze_edge_coverage(edge, start_node, end_node, coverage_data)
                    edge_results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing edge coverage: {str(e)}")
        
        # Calculate overall statistics
        total_edges = len(edge_results)
        covered_edges = sum(1 for r in edge_results if r["is_covered"])
        coverage_percentage = (covered_edges / total_edges * 100) if total_edges > 0 else 0
        
        total_length = sum(r["edge_length"] for r in edge_results)
        total_covered_length = sum(r["covered_length"] for r in edge_results)
        length_coverage_percentage = (total_covered_length / total_length * 100) if total_length > 0 else 0
        
        # Find edges with gaps
        edges_with_gaps = [r for r in edge_results if r["gaps"]]
        
        # Return overall results
        return {
            "total_edges": total_edges,
            "covered_edges": covered_edges,
            "coverage_percentage": coverage_percentage,
            "total_length": total_length,
            "covered_length": total_covered_length,
            "length_coverage_percentage": length_coverage_percentage,
            "edges_with_gaps": len(edges_with_gaps),
            "edge_results": edge_results,
            "street_view_points": len(coverage_data),
            "timestamp": time.time()
        }
    
    def generate_coverage_report(self, verification_results: Dict[str, Any],
                               output_file: str = None) -> str:
        """
        Generate a human-readable coverage report.
        
        Args:
            verification_results: Results from verify_coverage
            output_file: Path to save the report, or None for default
            
        Returns:
            Path to the generated report file
        """
        # Set default output file if not provided
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"coverage_report_{timestamp}.txt"
        
        # Create report content
        report_lines = [
            "======================================================",
            "             STREET VIEW COVERAGE REPORT              ",
            "======================================================",
            "",
            f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "OVERALL STATISTICS:",
            f"- Total edges analyzed: {verification_results['total_edges']}",
            f"- Covered edges: {verification_results['covered_edges']} ({verification_results['coverage_percentage']:.2f}%)",
            f"- Total road length: {verification_results['total_length']:.2f}m",
            f"- Covered road length: {verification_results['covered_length']:.2f}m ({verification_results['length_coverage_percentage']:.2f}%)",
            f"- Edges with coverage gaps: {verification_results['edges_with_gaps']}",
            f"- Total Street View points: {verification_results['street_view_points']}",
            "",
            "COVERAGE THRESHOLD SETTINGS:",
            f"- Minimum coverage required: {self.coverage_threshold:.2f}%",
            f"- Maximum allowed gap: {self.max_gap_distance:.2f}m",
            ""
        ]
        
        # Add details of poorly covered edges
        poor_coverage_edges = [
            r for r in verification_results["edge_results"] 
            if r["coverage_percentage"] < self.coverage_threshold
        ]
        
        if poor_coverage_edges:
            report_lines.extend([
                "EDGES WITH INSUFFICIENT COVERAGE:",
                "-------------------------------------",
                ""
            ])
            
            # Sort by coverage percentage (ascending)
            poor_coverage_edges.sort(key=lambda e: e["coverage_percentage"])
            
            for edge in poor_coverage_edges[:20]:  # Limit to top 20 worst edges
                report_lines.extend([
                    f"Edge ID: {edge['edge_id']}",
                    f"- From node: {edge['start_node_id']} to node: {edge['end_node_id']}",
                    f"- Length: {edge['edge_length']:.2f}m",
                    f"- Coverage: {edge['coverage_percentage']:.2f}%",
                    f"- Street View points: {edge['street_view_points']}",
                    f"- Gaps: {len(edge['gaps'])}",
                    ""
                ])
            
            if len(poor_coverage_edges) > 20:
                report_lines.append(f"... and {len(poor_coverage_edges) - 20} more edges with poor coverage.")
                report_lines.append("")
        
        # Add recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-------------------------------------",
            ""
        ])
        
        # Generate recommendations based on results
        if verification_results["coverage_percentage"] < 50:
            report_lines.append("- Coverage is severely insufficient. Consider increasing the Street View collection density.")
        elif verification_results["coverage_percentage"] < 80:
            report_lines.append("- Coverage needs improvement. Focus on collecting additional points for roads with large gaps.")
        else:
            report_lines.append("- Overall coverage is good. Consider targeting specific edges with gaps if needed.")
        
        if verification_results["edges_with_gaps"] / verification_results["total_edges"] > 0.3:
            report_lines.append("- Many edges have coverage gaps. Review the sampling interval for Street View collection.")
        
        report_lines.extend([
            "",
            "======================================================",
            "                      END OF REPORT                   ",
            "======================================================"
        ])
        
        # Write report to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Coverage report generated at: {output_file}")
        
        return output_file
    
    def visualize_coverage(self, road_network_data: Dict[str, Any],
                         coverage_data: List[Dict[str, Any]],
                         verification_results: Dict[str, Any],
                         output_path: str = None) -> Dict[str, Any]:
        """
        Generate a visualization of the coverage verification results.
        
        Args:
            road_network_data: Dictionary with road network nodes and edges
            coverage_data: List of Street View points with coordinates and metadata
            verification_results: Results from verify_coverage
            output_path: Path to save the visualization, or None for default
            
        Returns:
            Dictionary with visualization results
        """
        if not VISUALIZATION_AVAILABLE:
            return {
                "success": False,
                "error": "Visualization module not available"
            }
        
        # Set default output path if not provided
        if output_path is None:
            timestamp = int(time.time())
            output_path = f"coverage_verification_{timestamp}.png"
        
        # Prepare coverage data with status based on verification results
        enhanced_coverage_data = []
        
        # Copy original coverage data
        for point in coverage_data:
            enhanced_coverage_data.append(point.copy())
        
        # Add coverage information for edges
        edge_results = verification_results.get("edge_results", [])
        
        # Add custom visualization for edges with poor coverage
        edges_to_add = []
        
        for edge_result in edge_results:
            # Skip if edge is sufficiently covered
            if edge_result["is_covered"]:
                continue
            
            # Get node IDs
            start_node_id = edge_result.get("start_node_id")
            end_node_id = edge_result.get("end_node_id")
            
            # Skip if missing node IDs
            if not start_node_id or not end_node_id:
                continue
            
            # Get nodes
            nodes = road_network_data.get("nodes", {})
            start_node = nodes.get(start_node_id)
            end_node = nodes.get(end_node_id)
            
            # Skip if nodes don't exist
            if not start_node or not end_node:
                continue
            
            # Get coordinates
            start_x, start_y = start_node.get("x", 0), start_node.get("y", 0)
            end_x, end_y = end_node.get("x", 0), end_node.get("y", 0)
            
            # Calculate edge center for annotation
            center_x = (start_x + end_x) / 2
            center_y = (start_y + end_y) / 2
            
            # Add center point with gap information
            coverage_percentage = edge_result.get("coverage_percentage", 0)
            
            # Add a marker for the edge center
            enhanced_coverage_data.append({
                "x": center_x,
                "y": center_y,
                "status": "gap",
                "coverage": coverage_percentage,
                "edge_id": edge_result.get("edge_id", "unknown")
            })
        
        # Generate visualization
        return visualize_street_view_coverage(
            road_network_data,
            enhanced_coverage_data,
            output_path
        )


# Convenience functions for direct usage

def verify_coverage(road_network_data: Dict[str, Any], 
                  coverage_data: List[Dict[str, Any]],
                  coverage_threshold: float = 50.0,
                  max_gap_distance: float = 30.0,
                  parallel: bool = True) -> Dict[str, Any]:
    """
    Verify Street View coverage for a road network.
    
    Args:
        road_network_data: Dictionary with road network nodes and edges
        coverage_data: List of Street View points with coordinates and metadata
        coverage_threshold: Minimum percentage coverage required (0-100)
        max_gap_distance: Maximum allowed distance (meters) between Street View points
        parallel: Whether to process edges in parallel
        
    Returns:
        Dictionary with coverage verification results
    """
    verifier = CoverageVerifier(coverage_threshold, max_gap_distance)
    return verifier.verify_coverage(road_network_data, coverage_data, parallel)

def generate_coverage_report(verification_results: Dict[str, Any],
                           coverage_threshold: float = 50.0,
                           max_gap_distance: float = 30.0,
                           output_file: str = None) -> str:
    """
    Generate a human-readable coverage report.
    
    Args:
        verification_results: Results from verify_coverage
        coverage_threshold: Minimum percentage coverage required (0-100)
        max_gap_distance: Maximum allowed distance (meters) between Street View points
        output_file: Path to save the report, or None for default
        
    Returns:
        Path to the generated report file
    """
    verifier = CoverageVerifier(coverage_threshold, max_gap_distance)
    return verifier.generate_coverage_report(verification_results, output_file)

def visualize_coverage_verification(road_network_data: Dict[str, Any],
                                  coverage_data: List[Dict[str, Any]],
                                  verification_results: Dict[str, Any],
                                  output_path: str = None) -> Dict[str, Any]:
    """
    Generate a visualization of the coverage verification results.
    
    Args:
        road_network_data: Dictionary with road network nodes and edges
        coverage_data: List of Street View points with coordinates and metadata
        verification_results: Results from verify_coverage
        output_path: Path to save the visualization, or None for default
        
    Returns:
        Dictionary with visualization results
    """
    verifier = CoverageVerifier()
    return verifier.visualize_coverage(road_network_data, coverage_data, verification_results, output_path)


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Arcanum Street View Coverage Verification")
    parser.add_argument("--road-network", required=True, help="Path to road network JSON file")
    parser.add_argument("--coverage", required=True, help="Path to Street View coverage JSON file")
    parser.add_argument("--threshold", type=float, default=50.0, help="Minimum coverage percentage required (0-100)")
    parser.add_argument("--max-gap", type=float, default=30.0, help="Maximum allowed distance (meters) between Street View points")
    parser.add_argument("--report", help="Path to save the coverage report, or None for default")
    parser.add_argument("--visualize", action="store_true", help="Generate coverage visualization")
    parser.add_argument("--output", help="Path to save the visualization, or None for default")
    parser.add_argument("--sequential", action="store_true", help="Process edges sequentially (not in parallel)")
    args = parser.parse_args()
    
    # Load road network data
    try:
        with open(args.road_network, 'r') as f:
            road_network_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading road network data: {str(e)}")
        sys.exit(1)
    
    # Load coverage data
    try:
        with open(args.coverage, 'r') as f:
            coverage_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading coverage data: {str(e)}")
        sys.exit(1)
    
    # Create verifier
    verifier = CoverageVerifier(
        coverage_threshold=args.threshold,
        max_gap_distance=args.max_gap
    )
    
    # Verify coverage
    logger.info("Verifying Street View coverage...")
    verification_results = verifier.verify_coverage(
        road_network_data,
        coverage_data,
        not args.sequential  # Use parallel processing unless sequential flag is set
    )
    
    # Generate report
    if args.report:
        report_path = verifier.generate_coverage_report(verification_results, args.report)
        logger.info(f"Coverage report generated at: {report_path}")
    else:
        # Always generate a report with default path
        report_path = verifier.generate_coverage_report(verification_results)
        logger.info(f"Coverage report generated at: {report_path}")
    
    # Generate visualization if requested
    if args.visualize:
        logger.info("Generating coverage visualization...")
        visualization_result = verifier.visualize_coverage(
            road_network_data,
            coverage_data,
            verification_results,
            args.output
        )
        
        if visualization_result.get("success"):
            logger.info(f"Coverage visualization generated at: {visualization_result.get('visualization_path')}")
        else:
            logger.error(f"Error generating visualization: {visualization_result.get('error')}")
    
    # Print summary
    print("\nCOVERAGE VERIFICATION SUMMARY:")
    print(f"Total edges: {verification_results['total_edges']}")
    print(f"Covered edges: {verification_results['covered_edges']} ({verification_results['coverage_percentage']:.2f}%)")
    print(f"Total road length: {verification_results['total_length']:.2f}m")
    print(f"Covered road length: {verification_results['covered_length']:.2f}m ({verification_results['length_coverage_percentage']:.2f}%)")
    print(f"Edges with gaps: {verification_results['edges_with_gaps']}")
    print(f"Report generated at: {report_path}")