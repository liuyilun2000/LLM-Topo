#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01a_graph_generation.py - Generate graph representation using polygon-based repulsive force method

This script generates graph structures using a physics-based approach:
- Creates a 2n-polygon and samples points uniformly
- Applies repulsive forces to distribute points evenly
- Performs topology gluing based on gluing rules (use capital letters for reversed edges, e.g., "abAB" for torus)
- Saves adjacency matrix, nodes, coordinates, and distance matrix

Output files:
  - A_{dataset_name}_labeled.csv : Adjacency matrix with node labels
  - A_{dataset_name}.npy : Adjacency matrix
  - nodes_{dataset_name}.csv : Node information
  - coords_{dataset_name}.csv : Coordinates (x, y, z)
  - coords_{dataset_name}.npy : Coordinates as numpy array
  - distance_matrix_{dataset_name}.npy : Shortest path distance matrix
  - graph_info_{dataset_name}.json : Graph metadata
"""

import argparse
import re
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, KDTree
from shapely.geometry import Polygon as ShapelyPolygon, Point, Polygon
import os
import json
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from math import cos, sin, pi, sqrt


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class EvolutionConfig:
    """Configuration for point evolution."""
    iters: int = 200
    step_size: float = 0.05
    repulsive_strength: float = 2.0
    interaction_range: Optional[float] = None
    adaptive_step: bool = True
    max_force: float = 0.5
    k_neighbors: int = 25
    save_plots: bool = True
    plot_interval: int = 10


@dataclass
class EvolutionMetrics:
    """Metrics collected during evolution."""
    deltas: List[float]
    avg_distances: List[float]
    distance_vars: List[float]


# =============================================================================
# Gluing Helper Class
# =============================================================================

class GluingHelper:
    """
    Helper class for handling topology gluing operations.
    Creates mirrored copies of polygon and points through glued edges.
    """
    
    def __init__(self, polygon: np.ndarray, gluing_rules: List[Tuple[int, int, int]]):
        """
        Initialize gluing helper.
        
        Args:
            polygon: Original polygon vertices (2n edges)
            gluing_rules: List of (edge1_idx, edge2_idx, orientation) tuples
                         orientation: -1 for standard/reverse, 1 for twist
        """
        self.polygon = polygon
        self.gluing_rules = gluing_rules
        self.num_edges = len(polygon)
        self.domain = self._get_shapely_domain(polygon)
    
    @staticmethod
    def _get_shapely_domain(polygon: np.ndarray) -> ShapelyPolygon:
        """Get Shapely polygon domain."""
        domain = ShapelyPolygon(polygon)
        return domain if domain.is_valid else domain.buffer(0.0)
    
    def _mirror_point_through_edge(self, point: np.ndarray, edge1_idx: int, 
                                    edge2_idx: int, orientation: int) -> np.ndarray:
        """
        Mirror a point through a glued edge pair, correctly handling orientation.
        
        Args:
            point: Point to mirror
            edge1_idx: Source edge index
            edge2_idx: Target edge index (glued to edge1)
            orientation: 1 for twist (same direction), -1 for standard reverse
        
        Returns:
            Mirrored point position
        """
        # Get edge endpoints
        p0_e1 = self.polygon[edge1_idx]
        p1_e1 = self.polygon[(edge1_idx + 1) % self.num_edges]
        p0_e2 = self.polygon[edge2_idx]
        p1_e2 = self.polygon[(edge2_idx + 1) % self.num_edges]
        
        edge_vec_e1 = p1_e1 - p0_e1
        edge_vec_e2 = p1_e2 - p0_e2
        edge_len_e1 = np.linalg.norm(edge_vec_e1)
        edge_len_e2 = np.linalg.norm(edge_vec_e2)
        
        if edge_len_e1 < 1e-10 or edge_len_e2 < 1e-10:
            return point.copy()
        
        # Project point onto edge1 to get parameter t (0 to 1 along edge)
        to_point = point - p0_e1
        t = np.clip(np.dot(to_point, edge_vec_e1) / (edge_len_e1 ** 2), 0.0, 1.0)
        
        # Closest point on edge1
        closest_on_e1 = p0_e1 + t * edge_vec_e1
        
        # Offset from edge1 (perpendicular distance)
        offset = point - closest_on_e1
        
        # Map parameter t to edge2 based on orientation
        if orientation == 1:  # Twist: same direction
            t_mapped = t
        else:  # Standard reverse: opposite direction
            t_mapped = 1.0 - t
        
        # Corresponding position on edge2
        corresponding_on_e2 = p0_e2 + t_mapped * edge_vec_e2
        
        # Handle offset based on orientation
        if orientation == -1:
            # Get edge1 normal (perpendicular to edge1, pointing "inside" polygon)
            edge_normal_e1 = np.array([-edge_vec_e1[1], edge_vec_e1[0]]) / edge_len_e1
            
            # For reverse gluing, reflect the offset across edge1's normal
            offset_normal_component = np.dot(offset, edge_normal_e1)
            offset = offset - 2 * offset_normal_component * edge_normal_e1
        
        # Apply offset to corresponding position on edge2
        mirrored_point = corresponding_on_e2 + offset
        
        return mirrored_point
    
    def create_extended_space(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create extended space with mirrored copies of polygon and points.
        
        For each edge, mirror the entire graph and points through the corresponding glued edge
        in the opposite direction, creating a complete tiled representation with copies
        across all glued edges.
        
        Args:
            points: Original point set (n, 2)
        
        Returns:
            extended_polygon: Extended polygon vertices (original + mirrored copies)
            extended_points: Extended point set (original + all mirrored points)
            point_mapping: Mapping from extended point index to original point index
        """
        n_points = len(points)
        
        # Start with original polygon and points
        all_polygons = [self.polygon.copy()]
        all_points = [points.copy()]
        point_mapping = [np.arange(n_points)]  # Original points map to themselves
        
        # For each gluing rule, create mirrored copies in both directions
        for e1, e2, orient in self.gluing_rules:
            # Mirror polygon through e1->e2
            mirrored_polygon_e1_to_e2 = []
            for vertex in self.polygon:
                # Find which edge this vertex belongs to and mirror accordingly
                # For simplicity, mirror each vertex through the edge pair
                mirrored_vertex = self._mirror_point_through_edge(vertex, e1, e2, orient)
                mirrored_polygon_e1_to_e2.append(mirrored_vertex)
            all_polygons.append(np.array(mirrored_polygon_e1_to_e2))
            
            # Mirror points through e1->e2
            mirrored_points_e1_to_e2 = []
            for point in points:
                mirrored_point = self._mirror_point_through_edge(point, e1, e2, orient)
                mirrored_points_e1_to_e2.append(mirrored_point)
            all_points.append(np.array(mirrored_points_e1_to_e2))
            point_mapping.append(np.arange(n_points))
            
            # Mirror polygon through e2->e1
            mirrored_polygon_e2_to_e1 = []
            for vertex in self.polygon:
                mirrored_vertex = self._mirror_point_through_edge(vertex, e2, e1, orient)
                mirrored_polygon_e2_to_e1.append(mirrored_vertex)
            all_polygons.append(np.array(mirrored_polygon_e2_to_e1))
            
            # Mirror points through e2->e1
            mirrored_points_e2_to_e1 = []
            for point in points:
                mirrored_point = self._mirror_point_through_edge(point, e2, e1, orient)
                mirrored_points_e2_to_e1.append(mirrored_point)
            all_points.append(np.array(mirrored_points_e2_to_e1))
            point_mapping.append(np.arange(n_points))
        
        # Combine all polygons (for visualization, we mainly use the first one)
        # Extended points and mapping are what we need for force computation
        extended_points = np.vstack(all_points)
        point_mapping = np.concatenate(point_mapping)
        
        # Return first polygon (original) and extended points
        return all_polygons[0], extended_points, point_mapping
    
    def wrap_point(self, point: np.ndarray) -> np.ndarray:
        """
        Wrap a point through gluing rules if it's outside the polygon.
        Like Snake game boundaries - point wraps to corresponding position on glued edge.
        """
        pt = Point(point)

        # If point is inside, return as-is
        if self.domain.contains(pt) or self.domain.touches(pt):
            return point

        # Find which edge the point is closest to
        edge_info = self._find_edge_and_parameter(point)
        if edge_info is None:
            # Fallback: project to boundary
            nearest = self.domain.boundary.interpolate(self.domain.boundary.project(pt))
            return np.array([nearest.x, nearest.y])

        edge_idx, t = edge_info

        # Find if this edge is glued to another edge
        for e1, e2, orient in self.gluing_rules:
            if e1 == edge_idx:
                p0_e2 = self.polygon[e2]
                p1_e2 = self.polygon[(e2 + 1) % self.num_edges]
                if orient == 1:  # Twist: same direction
                    wrapped_point = p0_e2 + t * (p1_e2 - p0_e2)
                else:  # Standard reverse: opposite direction
                    wrapped_point = p1_e2 + (1 - t) * (p0_e2 - p1_e2)
                return wrapped_point
            elif e2 == edge_idx:
                p0_e1 = self.polygon[e1]
                p1_e1 = self.polygon[(e1 + 1) % self.num_edges]
                if orient == 1:  # Twist: same direction
                    wrapped_point = p0_e1 + t * (p1_e1 - p0_e1)
                else:  # Standard reverse: opposite direction
                    wrapped_point = p1_e1 + (1 - t) * (p0_e1 - p1_e1)
                return wrapped_point

        # No gluing rule found - project to boundary
        nearest = self.domain.boundary.interpolate(self.domain.boundary.project(pt))
        return np.array([nearest.x, nearest.y])

    def _find_edge_and_parameter(self, point: np.ndarray) -> Optional[Tuple[int, float]]:
        """Find which edge a point is closest to and its parameter along that edge."""
        pt = Point(point)
        if self.domain.contains(pt) or self.domain.touches(pt):
            return None

        min_dist = float('inf')
        closest_edge = None
        closest_t = 0.0

        for i in range(self.num_edges):
            p0 = self.polygon[i]
            p1 = self.polygon[(i + 1) % self.num_edges]
            edge_vec = p1 - p0
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-10:
                continue

            to_point = point - p0
            t = np.dot(to_point, edge_vec) / (edge_len ** 2)
            t = np.clip(t, 0.0, 1.0)

            closest_on_edge = p0 + t * edge_vec
            dist = np.linalg.norm(point - closest_on_edge)

            if dist < min_dist:
                min_dist = dist
                closest_edge = i
                closest_t = t

        return (closest_edge, closest_t) if closest_edge is not None else None


# =============================================================================
# Polygon Environment Class
# =============================================================================

class PolygonEnvironment:
    """
    Main environment class for polygon-based graph generation.
    Contains polygon, points, configuration, and all operations.
    """
    
    def __init__(self, n: int, N_total: int, seed: int = 42, 
                 topology_rule: Optional[str] = None,
                 config: Optional[EvolutionConfig] = None,
                 radius: float = 1.0):
        """
        Initialize polygon environment with points and configuration.
        
        Args:
            n: Parameter for 2n-polygon
            N_total: Total number of points to sample
            seed: Random seed
            topology_rule: Topology rule string (e.g., "abAB" for torus)
            config: Evolution configuration
            radius: Polygon radius
        """
        # Store parameters
        self.n = n
        self.N_total = N_total
        self.seed = seed
        self.radius = radius
        self.config = config or EvolutionConfig()
        
        # Initialize polygon
        self.polygon = self._create_regular_2n_polygon(n, radius)
        self.domain = self._get_shapely_domain(self.polygon)
        
        # Parse topology rules if provided
        self.gluing_rules = None
        self.gluing_helper = None
        if topology_rule:
            num_edges = 2 * n
            self.gluing_rules = self._parse_topology_string(topology_rule, num_edges)
            self.gluing_helper = GluingHelper(self.polygon, self.gluing_rules)
            print(f"[INFO] Initialized topology: {topology_rule} with {len(self.gluing_rules)} gluing rules")
        
        # Initialize random generator and sample points
        self.rng = np.random.default_rng(seed)
        self.points = self._sample_interior_points(N_total)
        
        # Evolution state
        self.current_iter = 0
        self.metrics = EvolutionMetrics([], [], [])
        
        print(f"[INFO] Initialized PolygonEnvironment: n={n}, N_total={N_total}, seed={seed}")
    
    @staticmethod
    def _create_regular_2n_polygon(n: int, radius: float = 1.0) -> np.ndarray:
        """Construct regular 2n-polygon, vertices in counterclockwise order."""
        m = 2 * n
        vertices = []
        for k in range(m):
            theta = 2 * pi * k / m
            vertices.append([radius * cos(theta), radius * sin(theta)])
        return np.array(vertices, dtype=float)
    
    @staticmethod
    def _get_shapely_domain(polygon: np.ndarray) -> ShapelyPolygon:
        """Get Shapely polygon domain."""
        domain = ShapelyPolygon(polygon)
        return domain if domain.is_valid else domain.buffer(0.0)
    
    @staticmethod
    def _parse_topology_string(rule_str: str, num_edges: int) -> List[Tuple[int, int, int]]:
        """
        Parse topology string using capital letter notation.
        
        Input format:
        - Lowercase letters (a, b, c, ...) = normal edges
        - Uppercase letters (A, B, C, ...) = reversed edges (A = a^-1, B = b^-1, etc.)
        
        Example: "abAB" means edges a, b, a^-1, b^-1
        
        Returns:
            List of (edge1_idx, edge2_idx, orientation) tuples
            orientation: -1 for standard/reverse, 1 for twist
        """
        tokens = re.findall(r"([a-zA-Z])", rule_str)
        if len(tokens) != num_edges:
            raise ValueError(
                f"[Error] Topology rule defines {len(tokens)} edges, "
                f"but the polygon has {num_edges} edges.\nInput: {rule_str}"
            )
        
        gluing_rules = []
        seen = {}
        for i, char in enumerate(tokens):
            is_inverse = char.isupper()
            char_lower = char.lower()
            
            if char_lower in seen:
                prev_i, prev_is_inv = seen[char_lower]
                orientation = -1 if (is_inverse != prev_is_inv) else 1
                gluing_rules.append((prev_i, i, orientation))
                del seen[char_lower]
            else:
                seen[char_lower] = (i, is_inverse)
        
        return gluing_rules
    
    def _triangulate_fan(self, vertices: np.ndarray) -> List[Tuple[int, int, int]]:
        """Fan triangulation of convex polygon using vertex 0 as center."""
        m = len(vertices)
        tris = []
        for i in range(1, m - 1):
            tris.append((0, i, i + 1))
        return tris
    
    def _compute_area(self, vertices: np.ndarray) -> float:
        """Compute polygon area using shoelace formula."""
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    
    def _sample_interior_points(self, N_interior: int) -> np.ndarray:
        """
        Sample N_interior points uniformly inside polygon using triangulation.
        This function is called during initialization to distribute points.
        """
        if N_interior <= 0:
            return np.zeros((0, 2), dtype=float)
        
        tris = self._triangulate_fan(self.polygon)
        tri_areas = []
        for (i, j, k) in tris:
            A = self._compute_area(self.polygon[[i, j, k]])
            tri_areas.append(A)
        
        tri_areas = np.array(tri_areas, dtype=float)
        probs = tri_areas / tri_areas.sum()
        samples = []
        
        for _ in range(N_interior):
            t_idx = self.rng.choice(len(tris), p=probs)
            i, j, k = tris[t_idx]
            a, b, c = self.polygon[i], self.polygon[j], self.polygon[k]
            
            # Uniform sampling in triangle
            u = self.rng.random()
            v = self.rng.random()
            su = sqrt(u)
            w1 = 1 - su
            w2 = su * (1 - v)
            w3 = su * v
            p = w1 * a + w2 * b + w3 * c
            samples.append(p)
        
        return np.array(samples, dtype=float)
    
    def _compute_forces(self) -> np.ndarray:
        """
        Compute repulsive forces on current points.
        Uses GluingHelper to create extended space if gluing rules exist.
        """
        n = len(self.points)
        if n <= 1:
            return np.zeros((n, 2))
        
        # Determine interaction range
        interaction_range = self.config.interaction_range
        if interaction_range is None:
            poly_size = np.ptp(self.points, axis=0).max() if len(self.points) > 0 else 10.0
            interaction_range = poly_size * 10.0
            
        forces = np.zeros((n, 2))
        
        if self.gluing_helper:
            # Use GluingHelper to create extended space
            _, extended_points, point_mapping = self.gluing_helper.create_extended_space(self.points)
            tree = KDTree(extended_points)
            
            # For each original point, find neighbors in extended space
            for i in range(n):
                # Original point i is at index i in extended_points (first n points are originals)
                neighbors = tree.query_ball_point(extended_points[i], interaction_range)
                
                processed_pairs = set()
                for neighbor_idx in neighbors:
                    j_original = point_mapping[neighbor_idx]
                    # Skip if same original point
                    if j_original == i:
                        continue
                    
                    pair = (min(i, j_original), max(i, j_original))
                    if pair in processed_pairs:
                        continue
                    processed_pairs.add(pair)
                    
                    # Compute force
                    vec = extended_points[i] - extended_points[neighbor_idx]
                    dist = max(np.linalg.norm(vec), 1e-6)
                    force_mag = self.config.repulsive_strength / (dist ** 2)
                    vec_normalized = vec / dist
                    forces[i] += force_mag * vec_normalized
        else:
            # No gluing - standard computation
            tree = KDTree(self.points)
            for i in range(n):
                neighbors = tree.query_ball_point(self.points[i], interaction_range)
                for j in neighbors:
                    if i == j:
                        continue
                    vec = self.points[i] - self.points[j]
                    dist = max(np.linalg.norm(vec), 1e-6)
                    force_mag = self.config.repulsive_strength / (dist ** 2)
                    vec_normalized = vec / dist
                    forces[i] += force_mag * vec_normalized
        
        return forces
    
    def evolve_one_step(self) -> None:
        """
        Run one evolution step.
        Internally uses GluingHelper to create extended space and performs evolution on it.
        """
        # Compute forces
        forces = self._compute_forces()

        # Limit force magnitude
        force_mags = np.linalg.norm(forces, axis=1)
        mask = force_mags > self.config.max_force
        if np.any(mask):
            forces[mask] *= (self.config.max_force / force_mags[mask, np.newaxis])

        # Determine step size
        if self.config.adaptive_step:
            current_step = self.config.step_size * (1.0 - self.current_iter / self.config.iters) ** 2
        else:
            current_step = self.config.step_size

        # Update positions
        old_points = self.points.copy()
        self.points = self.points + current_step * forces

        # Wrap points through gluing if they move outside polygon
        if self.gluing_helper:
            for i in range(len(self.points)):
                self.points[i] = self.gluing_helper.wrap_point(self.points[i])
        else:
            # No gluing - project to domain
            for i in range(len(self.points)):
                pt = Point(self.points[i])
                if not (self.domain.contains(pt) or self.domain.touches(pt)):
                    nearest = self.domain.boundary.interpolate(self.domain.boundary.project(pt))
                    self.points[i] = np.array([nearest.x, nearest.y])

        # Update metrics
        delta = float(np.linalg.norm(self.points - old_points, axis=1).mean())
        self.metrics.deltas.append(delta)

        if len(self.points) > 1:
            nn_dists = KDTree(self.points).query(self.points, k=2)[0][:, 1]
            avg_dist = float(nn_dists.mean())
            dist_var = float(nn_dists.var())
        else:
            avg_dist = 0.0
            dist_var = 0.0

        self.metrics.avg_distances.append(avg_dist)
        self.metrics.distance_vars.append(dist_var)

        self.current_iter += 1
    
    def run_evolution(self, output_dir: str = ".") -> None:
        """
        Run full evolution for specified iterations.
        """
        plot_dir = os.path.join(output_dir, "evolution_plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        print(f"[Evolution] Starting evolution: {self.config.iters} iterations")
        if self.gluing_helper:
            print("[Evolution] Using gluing-aware force computation")
        
        for k in range(self.config.iters):
            self.evolve_one_step()
            
            # Save plots at intervals
            if self.config.save_plots and (k == 0 or k == self.config.iters - 1 or 
                                          (k + 1) % self.config.plot_interval == 0):
                plot_path = os.path.join(plot_dir, f"iter_{k:03d}.png")
                self.visualize(plot_path, f"Iteration {k+1}/{self.config.iters}")
                print(f"[Evolution] Saved plot: {plot_path}")
    
        # Print final statistics
        if len(self.points) > 1:
            nn_dists = KDTree(self.points).query(self.points, k=2)[0][:, 1]
            uniformity = nn_dists.std() / nn_dists.mean() if nn_dists.mean() > 0 else float('inf')
            print(f"[FINAL] Uniformity (CV): {uniformity:.6f}")
            print(f"[FINAL] NN distance: mean={nn_dists.mean():.4f}, std={nn_dists.std():.4f}")
    
    def visualize(self, save_path: str, title: str = "Voronoi diagram") -> None:
        """
        Visualization function.
        Plot Voronoi diagram with optional gluing boundary visualization.
        """
        vor = Voronoi(self.points)
        regions, vtx = self._voronoi_finite_polygons_2d(vor)
        fig, ax = plt.subplots(figsize=(8, 8))
        poly_closed = np.vstack([self.polygon, self.polygon[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], "k-", lw=2, label="polygon boundary")
        
        # Highlight glued edges if gluing rules provided
        if self.gluing_rules:
            num_edges = len(self.polygon)
            for e1, e2, orient in self.gluing_rules:
                p0_e1 = self.polygon[e1]
                p1_e1 = self.polygon[(e1 + 1) % num_edges]
                ax.plot([p0_e1[0], p1_e1[0]], [p0_e1[1], p1_e1[1]], 
                        "r-", lw=2, alpha=0.6, label="glued edges" if e1 == self.gluing_rules[0][0] else "")
                
                p0_e2 = self.polygon[e2]
                p1_e2 = self.polygon[(e2 + 1) % num_edges]
                ax.plot([p0_e2[0], p1_e2[0]], [p0_e2[1], p1_e2[1]], 
                        "r-", lw=2, alpha=0.6)
        
        # Plot Voronoi cells
        for region in regions:
            if len(region) == 0:
                continue
            coords = self._sort_vertices(vtx[region])
            if coords is None or len(coords) < 3:
                continue
            in_domain = np.array([
                self.domain.contains(Point(pt)) if np.isfinite(pt).all() else False 
                for pt in coords
            ])
            if not np.any(in_domain):
                continue
            if np.all(in_domain):
                if len(coords) >= 3 and not np.allclose(coords[0], coords[-1], atol=1e-10):
                    coords = np.vstack([coords, coords[0]])
                ax.plot(coords[:, 0], coords[:, 1], "b-", lw=0.5, alpha=0.4)
            else:
                if len(coords) < 3:
                    continue
                if not np.allclose(coords[0], coords[-1], atol=1e-10):
                    coords = np.vstack([coords, coords[0]])
                if len(coords) < 4:
                    continue
                try:
                    cell = ShapelyPolygon(coords)
                    if cell.is_valid:
                        self._plot_clipped_cell(ax, cell, self.domain)
                except (ValueError, Exception):
                    continue
        
        ax.scatter(self.points[:, 0], self.points[:, 1], s=15, c='blue', alpha=0.8, label="points", zorder=5)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def _voronoi_finite_polygons_2d(self, vor: Voronoi, radius: Optional[float] = None):
        """Clip infinite Voronoi regions to finite polygons."""
        if vor.points.shape[1] != 2:
            raise ValueError("Only supports 2D Voronoi.")
        new_regions = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        if radius is None:
            radius = np.ptp(vor.points, axis=0).max() * 2.0
        all_ridges: dict = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))
        for p_idx, region_idx in enumerate(vor.point_region):
            verts = vor.regions[region_idx]
            if len(verts) == 0:
                new_regions.append([])
                continue
            if all(v >= 0 for v in verts):
                new_regions.append(verts)
                continue
            ridges = all_ridges.get(p_idx, [])
            new_region = [v for v in verts if v >= 0]
            for q_idx, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue
                t = vor.points[q_idx] - vor.points[p_idx]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = vor.points[[p_idx, q_idx]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius
                new_vertices.append(far_point.tolist())
                new_idx = len(new_vertices) - 1
                new_region.append(new_idx)
            vs = np.asarray(new_vertices)[new_region]
            angles = np.arctan2(vs[:, 1] - center[1], vs[:, 0] - center[0])
            new_region = [new_region[i] for i in np.argsort(angles)]
            new_regions.append(new_region)
        return new_regions, np.asarray(new_vertices)
    
    @staticmethod
    def _sort_vertices(coords: np.ndarray) -> Optional[np.ndarray]:
        """Sort vertices by angle around center."""
        if len(coords) < 2:
            return None
        coords = np.unique(coords[np.isfinite(coords).all(axis=1)], axis=0)
        if len(coords) < 2:
            return None
        center = coords.mean(axis=0)
        angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
        return coords[np.argsort(angles)]
    
    @staticmethod
    def _plot_clipped_cell(ax, cell_poly: ShapelyPolygon, domain: ShapelyPolygon):
        """Plot clipped Voronoi cell."""
        clipped = cell_poly.intersection(domain)
        if clipped.is_empty:
            return
        geoms = clipped.geoms if hasattr(clipped, 'geoms') else [clipped]
        for geom in geoms:
            if geom.geom_type == 'Polygon':
                coords = np.array(geom.exterior.coords)
                ax.plot(coords[:, 0], coords[:, 1], "b-", lw=0.5, alpha=0.6)
    
    def build_voronoi_graph(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Generate Voronoi polygon edge connection rules.
        Uses GluingHelper to return mirrored copies of polygon and point states
        to handle boundary connections. Finally outputs and stores the desired graph.

        Returns:
            nodes: Node coordinates
            edges: List of edge tuples
        """
        num_points = len(self.points)
        print(f"[Graph] Building graph with {num_points} points using gluing-aware Voronoi...")

        if self.gluing_helper:
            # Use GluingHelper to create extended space
            _, extended_points, point_mapping = self.gluing_helper.create_extended_space(self.points)
            print(f"[Graph] Created extended point set: {len(extended_points)} points "
                  f"(original + {len(extended_points) - num_points} mirrored copies)")
        else:
            # No gluing - use original points
            extended_points = self.points
            point_mapping = np.arange(num_points)

        # Compute Voronoi diagram on the extended space
        vor = Voronoi(extended_points)
        raw_edges = vor.ridge_points

        # Extract edges and map back to original point indices
        final_edges_set = set()

        for (ext_i, ext_j) in raw_edges:
            if ext_i == ext_j:
                continue

            # Map extended indices back to original point indices
            orig_i = point_mapping[ext_i]
            orig_j = point_mapping[ext_j]

            # Only add edge if it connects different original points
            if orig_i != orig_j:
                # Ensure consistent ordering (smaller index first)
                edge = (min(orig_i, orig_j), max(orig_i, orig_j))
                final_edges_set.add(edge)

        print(f"[Graph] Generated {len(final_edges_set)} edges from Voronoi neighbors in extended space.")

        return self.points, list(final_edges_set)
    
    @staticmethod
    def reorder_nodes_bfs(nodes: np.ndarray, edges: List[Tuple[int, int]], 
                          start_node: int = 0) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
        """
        Reorder nodes using BFS traversal starting from a specified node.
        
        Args:
            nodes: Array of node coordinates (N, 2)
            edges: List of edge tuples (u, v)
            start_node: Starting node index for BFS (default: 0)
        
        Returns:
            reordered_nodes: Nodes reordered by BFS traversal
            reordered_edges: Edges remapped to new node IDs
            old_to_new: Mapping from old node ID to new node ID
        """
        num_nodes = len(nodes)
        
        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # BFS traversal
        visited = [False] * num_nodes
        queue = [start_node]
        visited[start_node] = True
        bfs_order = [start_node]
        
        while queue:
            current = queue.pop(0)
            neighbors = sorted(adj_list[current])
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    bfs_order.append(neighbor)
        
        # Handle disconnected components
        for i in range(num_nodes):
            if not visited[i]:
                bfs_order.append(i)
        
        # Create mapping from old ID to new ID
        old_to_new = np.zeros(num_nodes, dtype=int)
        for new_id, old_id in enumerate(bfs_order):
            old_to_new[old_id] = new_id
        
        # Reorder nodes
        reordered_nodes = nodes[bfs_order]
        
        # Remap edges
        reordered_edges = []
        for u, v in edges:
            new_u = old_to_new[u]
            new_v = old_to_new[v]
            if new_u != new_v:
                if new_u > new_v:
                    new_u, new_v = new_v, new_u
                reordered_edges.append((new_u, new_v))
        
        reordered_edges = list(set(reordered_edges))
        
        return reordered_nodes, reordered_edges, old_to_new


# =============================================================================
# Dataset Export
# =============================================================================

class DatasetExporter:
    """Handles exporting graph datasets to files."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_dataset(self, nodes: np.ndarray, edges: List[Tuple[int, int]], 
                    dataset_name: str, metadata: Dict) -> None:
        """
        Save complete dataset files.
        """
        print(f"\n[Dataset] Generating files for '{dataset_name}'...")
        num_nodes = len(nodes)
        
        # Build adjacency matrix
        rows = [e[0] for e in edges] + [e[1] for e in edges]
        cols = [e[1] for e in edges] + [e[0] for e in edges]
        data = [1] * len(rows)
        adj_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        adj_dense = adj_sparse.toarray()
        
        # Save adjacency matrix
        np.save(f"{self.output_dir}/A_{dataset_name}.npy", adj_dense)
        df_adj = pd.DataFrame(adj_dense, index=range(num_nodes), columns=range(num_nodes))
        df_adj.to_csv(f"{self.output_dir}/A_{dataset_name}_labeled.csv")
        
        # Save coordinates
        np.save(f"{self.output_dir}/coords_{dataset_name}.npy", nodes)
        nodes_3d = np.column_stack([nodes, np.zeros(num_nodes)])
        df_coords = pd.DataFrame(nodes_3d, columns=['x', 'y', 'z'])
        df_coords['node_id'] = range(num_nodes)
        df_coords.set_index('node_id', inplace=True)
        df_coords.to_csv(f"{self.output_dir}/coords_{dataset_name}.csv")
        
        # Save node information
        degrees = np.array(adj_sparse.sum(axis=1)).flatten()
        df_nodes = pd.DataFrame({
            'node_id': range(num_nodes),
            'degree': degrees,
            'type': 'manifold_point'
        })
        df_nodes.set_index('node_id', inplace=True)
        df_nodes.to_csv(f"{self.output_dir}/nodes_{dataset_name}.csv")
        
        # Compute and save distance matrix
        print("[Dataset] Calculating shortest paths (this may take a moment)...")
        dist_matrix = shortest_path(csgraph=adj_sparse, directed=False, unweighted=True)
        
        # Handle infinite distances
        max_finite_dist = np.max(dist_matrix[np.isfinite(dist_matrix)])
        if np.isinf(max_finite_dist) or max_finite_dist == 0:
            max_finite_dist = num_nodes * 2
        else:
            max_finite_dist = max_finite_dist + 1
        
        dist_matrix = np.where(np.isfinite(dist_matrix), dist_matrix, max_finite_dist)
        np.fill_diagonal(dist_matrix, 0.0)
        
        np.save(f"{self.output_dir}/distance_matrix_{dataset_name}.npy", dist_matrix)
        
        # Save metadata
        info = {
            "dataset_name": dataset_name,
            "num_nodes": int(num_nodes),
            "num_edges": len(edges),
            "avg_degree": float(np.mean(degrees)),
            "euler_characteristic_est": int(num_nodes - len(edges) + len(edges)/3),
            **metadata
        }
        with open(f"{self.output_dir}/graph_info_{dataset_name}.json", "w") as f:
            json.dump(info, f, indent=4)
        
        print(f"[Dataset] Done! All files saved to './{self.output_dir}/'")
        print(f"  - A_{dataset_name}_labeled.csv")
        print(f"  - A_{dataset_name}.npy")
        print(f"  - nodes_{dataset_name}.csv")
        print(f"  - coords_{dataset_name}.csv")
        print(f"  - coords_{dataset_name}.npy")
        print(f"  - distance_matrix_{dataset_name}.npy")
        print(f"  - graph_info_{dataset_name}.json")


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Repulsive force point distribution on 2n-polygon")
    p.add_argument("--iters", type=int, default=200, help="Iterations")
    p.add_argument("--plot_interval", type=int, default=10, help="Plot interval")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--N_total", type=int, default=None, help="Total number of points (if not provided, will be estimated)")
    p.add_argument("--density_factor", type=float, default=1.0, help="Density factor for point estimation")
    p.add_argument("--step_size", type=float, default=0.1, help="Step size")
    p.add_argument("--repulsive_strength", type=float, default=2.0, help="Repulsive force strength")
    p.add_argument("--interaction_range", type=float, default=None, help="Interaction range")
    p.add_argument("--k_neighbors", type=int, default=25, help="K-neighbors for force computation")
    p.add_argument("--no_adaptive_step", action="store_true", help="Disable adaptive step")
    p.add_argument("--topology", type=str, default=None, 
                    help="Gluing rule string: use capital letters for reversed edges (A=a^-1, B=b^-1, etc.). Example: 'abAB' for torus")
    p.add_argument("--prefix", type=str, default=None,
                    help="Topology prefix for dataset naming (e.g., 'torus', 'klein', 'sphere'). If not provided, will be auto-detected from topology rule.")
    p.add_argument("--output_dir", type=str, default="output", help="Output directory")
    return p.parse_args()


def estimate_num_points(n: int, density_factor: float = 1.0) -> int:
    """Estimate number of points based on polygon area and desired density."""
    vertices = PolygonEnvironment._create_regular_2n_polygon(n, radius=1.0)
    poly = Polygon(vertices)
    area = poly.area
    
    rho = density_factor * 10.0  # points per unit area
    n_total = int(round(area * rho))
    n_total = max(n_total, 50)  # Minimum points
    
    print(f"[INFO] Estimated {n_total} points for polygon area {area:.4f}")
    return n_total


def detect_topology_prefix(topology_rule: str) -> str:
    """Auto-detect topology prefix based on topology rule."""
    if topology_rule in ["abAB", "abABcdCD", "abABcdCDefEF"]:
        return "torus"
    elif topology_rule in ["abAb", "abaB"]:
        return "klein"
    elif topology_rule in ["abBA", "aAbB"]:
        return "sphere"
    elif topology_rule == "abab":
        return "projective"
    else:
        return topology_rule.lower()[:8]


def main():
    """Main execution function."""
    args = parse_args()
    
    # Parse topology rules early if provided and compute n from topology rule
    n = None
    if args.topology:
        topology_edges = len(re.findall(r"([a-zA-Z])", args.topology))
        if topology_edges % 2 != 0:
            raise ValueError(
                f"[Error] Topology rule '{args.topology}' has {topology_edges} edges, "
                f"which is not even. A 2n-polygon must have an even number of edges.\n"
                f"  Please check your topology rule."
            )
        n = topology_edges // 2
        print(f"[INFO] Computed n={n} from topology rule '{args.topology}' ({topology_edges} edges)")
    else:
        n = 2
        print(f"[INFO] No topology rule provided, using default n={n}")
    
    # Estimate number of points if not provided
    if args.N_total is None:
        args.N_total = estimate_num_points(n, args.density_factor)
    
    # Create evolution configuration
    config = EvolutionConfig(
        iters=args.iters,
        step_size=args.step_size,
        repulsive_strength=args.repulsive_strength,
        interaction_range=args.interaction_range,
        adaptive_step=not args.no_adaptive_step,
        k_neighbors=args.k_neighbors,
        plot_interval=args.plot_interval
    )
    
    # Create polygon environment (initializes polygon, points, and gluing if provided)
    env = PolygonEnvironment(
        n=n,
        N_total=args.N_total,
        seed=args.seed,
        topology_rule=args.topology,
        config=config
    )
    
    print(f"[INFO] N_total={args.N_total}, iters={args.iters}")
    if env.gluing_helper:
        print(f"[INFO] Using gluing-aware force computation with {len(env.gluing_rules)} gluing rules")
        print("[INFO] Points can wrap through gluing boundaries (Snake game style)")
    
    # Run evolution
    env.run_evolution(output_dir=args.output_dir)
    print(f"[Final] Total points after evolution: {len(env.points)}")
    
    # Build graph considering gluing
    if args.topology:
        print(f"\n[Topology] Building graph with Gluing Rule: {args.topology}")
        try:
            final_nodes, final_edges = env.build_voronoi_graph()
            
            print("[Topology] Completed!")
            print(f"  - Nodes: {len(final_nodes)}")
            print(f"  - Edges: {len(final_edges)}")
            
            # Reorder nodes using BFS
            print("\n[Reordering] Applying BFS traversal to assign node IDs...")
            final_nodes, final_edges, _ = PolygonEnvironment.reorder_nodes_bfs(final_nodes, final_edges, start_node=0)
            print(f"  - Reordered {len(final_nodes)} nodes using BFS from node 0")
            print(f"  - Remapped {len(final_edges)} edges")
            
            # Construct dataset name with prefix
            if args.prefix:
                prefix = args.prefix
            else:
                prefix = detect_topology_prefix(args.topology)
            
            dataset_name = f"{prefix}_{args.topology}_N{args.N_total}_iter{args.iters}"
            
            # Prepare metadata
            metadata = {
                "topology_rule": args.topology,
                "n_polygon": n,
                "n_total": args.N_total,
                "density_factor": args.density_factor,
                "iters": args.iters
            }
            
            # Export dataset
            exporter = DatasetExporter(args.output_dir)
            exporter.save_dataset(final_nodes, final_edges, dataset_name, metadata)
        except Exception as e:
            print(f"[Topology Error] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
