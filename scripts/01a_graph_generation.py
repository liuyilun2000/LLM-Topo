#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01a_graph_generation.py - Generate graph representation of topological manifolds

This script creates graph structures from fundamental polygons with edge-gluing rules,
representing topological surfaces like torus, Klein bottle, double torus, etc.

Approach:
1. Start with a regular 2n-polygon (fundamental polygon)
2. Sample points uniformly inside (no fixed boundary points)
3. Use repulsive force simulation with periodic boundary conditions
4. Build Voronoi graph in tiled space to capture cross-boundary edges

Key features:
- Affine transformations handle edge gluing with correct orientation
- Multi-layer tiling ensures accurate neighbor detection across boundaries
- Snake-game wrapping for point evolution with proper edge identification
- Degree-uniform graphs (no artificially high-degree corner vertices)

Gluing rule notation:
- Lowercase letters (a, b, c, ...) = edges in standard direction
- Uppercase letters (A, B, C, ...) = reversed edges (A = a^-1)
- Examples: "abAB" (torus), "abABcdCD" (double torus), "abAb" (Klein bottle)

Output files:
  - A_{name}.npy, A_{name}_labeled.csv : Adjacency matrix
  - nodes_{name}.csv : Node information with degrees
  - coords_{name}.csv, coords_{name}.npy : 2D/3D coordinates
  - distance_matrix_{name}.npy : Shortest path distances
  - graph_info_{name}.json : Metadata
"""

import argparse
import re
import os
import json
from typing import Tuple, List, Optional, Dict, Set
from dataclasses import dataclass, field
from collections import deque
from math import cos, sin, pi, sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, KDTree
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from shapely.geometry import Polygon as ShapelyPolygon, Point


# =============================================================================
# Visualization: Polygon edge palette and arrow styling
# =============================================================================
# Official matplotlib tab20 palette - 20 distinct colors (supports up to 16 gluing pairs)
def _get_cmap(name: str = 'tab20'):
    """Get colormap (compatible with matplotlib 3.7+)."""
    try:
        return plt.colormaps[name]
    except (AttributeError, TypeError):
        return plt.cm.get_cmap(name)


def _get_edge_palette(n: int = 16) -> List:
    """Get n distinct colors from tab20 colormap (official matplotlib qualitative)."""
    cmap = _get_cmap('tab20')
    return [cmap(i / 19) for i in range(min(n, 20))]


def _get_node_palette(n: int) -> np.ndarray:
    """Get index-based colors for n nodes (continuous viridis, for BFS-ordered coloring)."""
    cmap = _get_cmap('viridis')
    if n <= 1:
        return np.array([cmap(0.5)] * n) if n else np.zeros((0, 4))
    return np.array([cmap(i / (n - 1)) for i in range(n)])

ARROW_SCALE = 0.12      # Arrow shaft length (fraction of plot scale)
ARROW_HEAD_SCALE = 18   # Arrow head size (mutation_scale)
EDGE_LABEL_OFFSET = 0.05   # Label offset from edge, outer side (fraction of plot scale)

# Marker sizes: smaller when tiled view (many points)
MARKER_MAIN = 32       # Main points, fundamental domain only
MARKER_MAIN_TILED = 20  # Main points when tiling shown
MARKER_TILED_COPY = 10  # Duplicated/tiled copies


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvolutionConfig:
    """Parameters for the repulsive force evolution simulation."""
    iters: int = 200              # Number of evolution iterations
    step_size: float = 0.05       # Base step size for point movement
    repulsive_strength: float = 2.0  # Strength of repulsive forces
    interaction_range: Optional[float] = None  # Auto-computed if None
    adaptive_step: bool = True    # Decay step size over iterations
    max_force: float = 0.5        # Clamp force magnitude
    save_plots: bool = True       # Save evolution visualizations
    plot_interval: int = 10       # Iterations between saved plots
    tiling_layers: int = 3        # Layers of tiled copies for neighbor detection


@dataclass
class EvolutionMetrics:
    """Metrics tracked during evolution."""
    deltas: List[float] = field(default_factory=list)
    avg_distances: List[float] = field(default_factory=list)
    distance_vars: List[float] = field(default_factory=list)


# =============================================================================
# Affine Transformations for Edge Gluing
# =============================================================================

class AffineTransform:
    """
    Represents an affine transformation: T(p) = matrix @ p + translation

    Used to map points across glued edges in the fundamental polygon.
    These are the deck transformations of the universal cover.
    """

    def __init__(self, matrix: np.ndarray, translation: np.ndarray):
        """
        Args:
            matrix: 2x2 rotation/reflection matrix
            translation: 2D translation vector
        """
        self.matrix = np.asarray(matrix, dtype=float)
        self.translation = np.asarray(translation, dtype=float)

        # Precompute inverse for efficiency
        self._inv_matrix = np.linalg.inv(self.matrix)
        self._inv_translation = -self._inv_matrix @ self.translation

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply transformation to point(s). Handles both (2,) and (N, 2) shapes."""
        if points.ndim == 1:
            return self.matrix @ points + self.translation
        return (self.matrix @ points.T).T + self.translation

    def apply_inverse(self, points: np.ndarray) -> np.ndarray:
        """Apply inverse transformation to point(s)."""
        if points.ndim == 1:
            return self._inv_matrix @ points + self._inv_translation
        return (self._inv_matrix @ points.T).T + self._inv_translation

    def compose(self, other: 'AffineTransform') -> 'AffineTransform':
        """Return composition: self(other(p))"""
        new_matrix = self.matrix @ other.matrix
        new_translation = self.matrix @ other.translation + self.translation
        return AffineTransform(new_matrix, new_translation)

    def inverse(self) -> 'AffineTransform':
        """Return the inverse transformation."""
        return AffineTransform(self._inv_matrix, self._inv_translation)

    def signature(self, precision: int = 4) -> Tuple:
        """Return a hashable signature for uniqueness checking."""
        m = self.matrix.flatten()
        t = self.translation
        return tuple(round(x, precision) for x in [m[0], m[1], m[2], m[3], t[0], t[1]])

    @staticmethod
    def identity() -> 'AffineTransform':
        """Return the identity transformation."""
        return AffineTransform(np.eye(2), np.zeros(2))


def compute_gluing_transform(polygon: np.ndarray,
                              source_edge: int,
                              target_edge: int,
                              reverse_orientation: bool) -> AffineTransform:
    """
    Compute the affine transformation that places a TILED COPY of the polygon
    adjacent to the original polygon, glued along source_edge.

    When you "exit" through source_edge of the original polygon, you "enter"
    through target_edge of the adjacent copy. This function computes the
    transformation that maps the original polygon to that adjacent copy.

    The key insight: we want to place a copy of the polygon such that its
    target_edge coincides with the source_edge of the original, but on the
    OUTSIDE of the original polygon.

    Args:
        polygon: Vertices of fundamental polygon (N, 2), counterclockwise order
        source_edge: Index of edge being exited (on original polygon)
        target_edge: Index of edge that will align with source_edge (on the copy)
        reverse_orientation: True if edges glue in opposite directions (aA),
                            False if same direction (aa - twist)

    Returns:
        AffineTransform that maps points from original to the adjacent tiled copy
    """
    num_vertices = len(polygon)

    # Source edge: the edge we're crossing on the original polygon
    src_p0 = polygon[source_edge]
    src_p1 = polygon[(source_edge + 1) % num_vertices]
    src_mid = (src_p0 + src_p1) / 2
    src_vec = src_p1 - src_p0
    src_len = np.linalg.norm(src_vec)

    if src_len < 1e-10:
        return AffineTransform.identity()

    src_dir = src_vec / src_len

    # Target edge: this edge of the COPY will align with source_edge
    tgt_p0 = polygon[target_edge]
    tgt_p1 = polygon[(target_edge + 1) % num_vertices]
    tgt_mid = (tgt_p0 + tgt_p1) / 2
    tgt_vec = tgt_p1 - tgt_p0
    tgt_len = np.linalg.norm(tgt_vec)

    if tgt_len < 1e-10:
        return AffineTransform.identity()

    tgt_dir = tgt_vec / tgt_len

    # Step 1: Determine the rotation needed
    # After gluing, we want:
    # - For reverse_orientation (aA): tgt_dir should align with -src_dir
    #   (walking forward on target = walking backward on source)
    # - For same_orientation (aa): tgt_dir should align with src_dir

    if reverse_orientation:
        # Target edge direction should be opposite to source
        desired_tgt_dir = -src_dir
    else:
        # Target edge direction should match source
        desired_tgt_dir = src_dir

    # Rotation to align tgt_dir with desired_tgt_dir
    cos_theta = np.dot(tgt_dir, desired_tgt_dir)
    sin_theta = tgt_dir[0] * desired_tgt_dir[1] - tgt_dir[1] * desired_tgt_dir[0]
    rotation = np.array([[cos_theta, -sin_theta],
                         [sin_theta, cos_theta]])

    rotated_tgt_mid = rotation @ tgt_mid

    # Step 2: Translation - place copy so its target edge COINCIDES with source edge
    #
    # For proper tiling, adjacent copies must SHARE the glued edge (no gap).
    # The copy's target edge should lie exactly on the original's source edge.
    # The rotation above already ensures the copy's interior is on the OUTSIDE
    # (opposite side of the edge from the original) - no additional shift needed.
    #
    # T(tgt_mid) = src_mid  =>  translation = src_mid - rotation @ tgt_mid

    translation = src_mid - rotated_tgt_mid

    return AffineTransform(rotation, translation)


# =============================================================================
# Topology Gluing Manager
# =============================================================================

class TopologyGluing:
    """
    Manages edge gluing rules and coordinate transformations for a topological surface.

    The fundamental polygon has edges that are pairwise identified (glued) according
    to the topology rule string. This class:
    1. Parses the gluing rules
    2. Computes affine transformations for each edge crossing
    3. Generates multi-layer tiled space for neighbor detection
    4. Provides wrapping function for evolution
    """

    def __init__(self, polygon: np.ndarray,
                 gluing_rules: List[Tuple[int, int, int]],
                 tiling_layers: int = 3):
        """
        Args:
            polygon: Vertices of fundamental polygon (2n, 2)
            gluing_rules: List of (edge1_idx, edge2_idx, orientation) tuples
                         orientation: -1 = reverse (aA), +1 = twist (aa)
            tiling_layers: Number of layers for tiled space generation
        """
        self.polygon = polygon
        self.gluing_rules = gluing_rules
        self.num_edges = len(polygon)
        self.tiling_layers = tiling_layers

        # Create Shapely domain for containment checks
        self.domain = ShapelyPolygon(polygon)
        if not self.domain.is_valid:
            self.domain = self.domain.buffer(0.0)

        # Compute transformation for each edge
        self.edge_transforms: Dict[int, AffineTransform] = {}
        self._compute_edge_transforms()

        # Generate tiled copies
        self.tiling_transforms: List[AffineTransform] = []
        self._generate_tiling()

    def _compute_edge_transforms(self):
        """Compute bidirectional transforms for each glued edge pair."""
        for edge1, edge2, orientation in self.gluing_rules:
            reverse = (orientation == -1)

            # Transform for crossing edge1 (going to edge2's side)
            self.edge_transforms[edge1] = compute_gluing_transform(
                self.polygon, edge1, edge2, reverse)

            # Transform for crossing edge2 (going to edge1's side)
            self.edge_transforms[edge2] = compute_gluing_transform(
                self.polygon, edge2, edge1, reverse)

    def _generate_tiling(self):
        """Generate transformations for multi-layer tiled space using BFS."""
        identity = AffineTransform.identity()
        seen = {identity.signature()}

        current_layer = [identity]
        all_transforms = [identity]

        for _ in range(self.tiling_layers):
            next_layer = []
            for T_base in current_layer:
                for T_edge in self.edge_transforms.values():
                    T_composed = T_edge.compose(T_base)
                    sig = T_composed.signature()
                    if sig not in seen:
                        seen.add(sig)
                        next_layer.append(T_composed)
                        all_transforms.append(T_composed)

            current_layer = next_layer
            if not current_layer:
                break

        self.tiling_transforms = all_transforms
        print(f"[TopologyGluing] Generated {len(all_transforms)} tiling copies "
              f"({self.tiling_layers} layers)")

    def create_tiled_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create tiled space with all copies of points.

        Args:
            points: Original points (N, 2)

        Returns:
            tiled_points: All copies (M*N, 2) where M = number of tiling transforms
            original_indices: Maps each tiled point to its original index (M*N,)
        """
        n_points = len(points)
        n_copies = len(self.tiling_transforms)

        tiled_points = np.zeros((n_copies * n_points, 2))
        original_indices = np.zeros(n_copies * n_points, dtype=int)

        for i, T in enumerate(self.tiling_transforms):
            start = i * n_points
            end = (i + 1) * n_points
            tiled_points[start:end] = T.apply(points)
            original_indices[start:end] = np.arange(n_points)

        return tiled_points, original_indices

    def wrap_to_domain(self, point: np.ndarray, max_iters: int = 10) -> np.ndarray:
        """
        Wrap a point back into the fundamental domain.

        Implements "snake game" wrapping: when a point exits through one edge,
        it enters through the glued edge with the appropriate transformation.

        Args:
            point: Point coordinates (2,)
            max_iters: Maximum wrapping iterations (prevents infinite loops)

        Returns:
            Wrapped point inside the fundamental domain
        """
        current = point.copy()
        centroid = np.array([self.domain.centroid.x, self.domain.centroid.y])

        for _ in range(max_iters):
            pt = Point(current)

            # Check if inside domain
            if self.domain.contains(pt):
                return current

            # If very close to boundary, nudge toward centroid
            if self.domain.boundary.distance(pt) < 1e-8:
                direction = centroid - current
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    current = current + 1e-6 * direction / norm
                return current

            # Find which edge we're outside of (most positive outward distance)
            best_edge = None
            best_outward_dist = -float('inf')

            for i in range(self.num_edges):
                p0 = self.polygon[i]
                p1 = self.polygon[(i + 1) % self.num_edges]
                edge_vec = p1 - p0
                edge_len = np.linalg.norm(edge_vec)

                if edge_len < 1e-10:
                    continue

                # Outward normal (for ccw polygon)
                outward_normal = np.array([edge_vec[1], -edge_vec[0]]) / edge_len
                outward_dist = np.dot(current - p0, outward_normal)

                if outward_dist > best_outward_dist:
                    best_outward_dist = outward_dist
                    best_edge = i

            if best_edge is None or best_outward_dist <= 0:
                return current

            # Apply inverse transform: point is in the tiled copy outside best_edge;
            # map it back to the fundamental domain via the glued edge.
            # edge_transforms[best_edge] maps fundamental -> copy, so we need inverse.
            if best_edge in self.edge_transforms:
                current = self.edge_transforms[best_edge].apply_inverse(current)
            else:
                # No transform - project to boundary
                nearest = self.domain.boundary.interpolate(
                    self.domain.boundary.project(pt))
                return np.array([nearest.x, nearest.y])

        # Max iterations - project to boundary
        pt = Point(current)
        nearest = self.domain.boundary.interpolate(self.domain.boundary.project(pt))
        return np.array([nearest.x, nearest.y])


# =============================================================================
# Polygon Environment and Evolution
# =============================================================================

class FundamentalPolygon:
    """
    Represents a regular 2n-polygon as fundamental domain for a surface.

    Handles:
    - Polygon geometry
    - Point sampling
    - Repulsive force evolution
    - Graph construction
    """

    def __init__(self, n: int, num_points: int,
                 topology_rule: Optional[str] = None,
                 config: Optional[EvolutionConfig] = None,
                 seed: int = 42, radius: float = 1.0):
        """
        Args:
            n: Creates a 2n-polygon
            num_points: Number of points to sample
            topology_rule: Gluing rule string (e.g., "abAB" for torus)
            config: Evolution parameters
            seed: Random seed
            radius: Polygon circumradius
        """
        self.n = n
        self.num_points = num_points
        self.topology_rule = topology_rule
        self.config = config or EvolutionConfig()
        self.radius = radius

        # Create polygon
        self.polygon = self._create_polygon(n, radius)
        self.domain = ShapelyPolygon(self.polygon)
        if not self.domain.is_valid:
            self.domain = self.domain.buffer(0.0)

        # Parse topology and create gluing manager
        self.gluing_rules = None
        self.topology = None
        if topology_rule:
            self.gluing_rules = self._parse_topology(topology_rule)
            self.topology = TopologyGluing(
                self.polygon,
                self.gluing_rules,
                self.config.tiling_layers
            )
            print(f"[FundamentalPolygon] Topology '{topology_rule}' with "
                  f"{len(self.gluing_rules)} gluing rules")

        # Initialize points
        self.rng = np.random.default_rng(seed)
        self.points = self._sample_points(num_points)

        # Evolution state
        self.current_iter = 0
        self.metrics = EvolutionMetrics()

        print(f"[FundamentalPolygon] Created {2*n}-gon with {num_points} points")

    @staticmethod
    def _create_polygon(n: int, radius: float) -> np.ndarray:
        """Create regular 2n-polygon with vertices in counterclockwise order."""
        num_vertices = 2 * n
        angles = [2 * pi * k / num_vertices for k in range(num_vertices)]
        return np.array([[radius * cos(a), radius * sin(a)] for a in angles])

    @staticmethod
    def _get_edge_directions(rule_str: str) -> List[bool]:
        """
        Get arrow direction for each edge from gluing rule.
        Returns list of bool: True = reversed (uppercase, arrow p1->p0), False = forward (p0->p1).
        E.g. abABcdCD -> [False,False,True,True,False,False,True,True]
        """
        tokens = re.findall(r"([a-zA-Z])", rule_str)
        return [char.isupper() for char in tokens]

    @staticmethod
    def _get_edge_labels(rule_str: str) -> List[str]:
        """
        Get labels for each edge: lowercase -> "a", uppercase -> "a⁻¹" (Unicode superscript).
        Plain text so fontweight='bold' works (LaTeX ignores it).
        """
        tokens = re.findall(r"([a-zA-Z])", rule_str)
        labels = []
        for char in tokens:
            letter = char.lower()
            labels.append(f"{letter}⁻¹" if char.isupper() else letter)
        return labels

    @staticmethod
    def _parse_topology(rule_str: str) -> List[Tuple[int, int, int]]:
        """
        Parse topology string into gluing rules.

        Format: Letters indicate edge labels. Lowercase = forward, Uppercase = reversed.
        Matching letters are glued together.

        Returns:
            List of (edge1_idx, edge2_idx, orientation) where
            orientation = -1 (reverse gluing aA) or +1 (twist gluing aa)
        """
        tokens = re.findall(r"([a-zA-Z])", rule_str)
        rules = []
        edge_labels = {}

        for idx, char in enumerate(tokens):
            is_upper = char.isupper()
            label = char.lower()

            if label in edge_labels:
                prev_idx, prev_upper = edge_labels[label]
                # orientation = -1 if one is upper and one is lower (reverse)
                # orientation = +1 if both same case (twist)
                orientation = -1 if (is_upper != prev_upper) else 1
                rules.append((prev_idx, idx, orientation))
                del edge_labels[label]
            else:
                edge_labels[label] = (idx, is_upper)

        return rules

    def _sample_points(self, num_points: int) -> np.ndarray:
        """Sample points uniformly inside polygon using fan triangulation."""
        if num_points <= 0:
            return np.zeros((0, 2))

        # Fan triangulation from vertex 0
        triangles = [(0, i, i + 1) for i in range(1, len(self.polygon) - 1)]

        # Compute triangle areas
        areas = []
        for v0, v1, v2 in triangles:
            a, b, c = self.polygon[v0], self.polygon[v1], self.polygon[v2]
            area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            areas.append(area)

        areas = np.array(areas)
        probs = areas / areas.sum()

        # Sample points
        samples = []
        for _ in range(num_points):
            tri_idx = self.rng.choice(len(triangles), p=probs)
            v0, v1, v2 = triangles[tri_idx]
            a, b, c = self.polygon[v0], self.polygon[v1], self.polygon[v2]

            # Uniform sampling in triangle using sqrt trick
            u, v = self.rng.random(), self.rng.random()
            su = sqrt(u)
            w1, w2, w3 = 1 - su, su * (1 - v), su * v
            samples.append(w1 * a + w2 * b + w3 * c)

        return np.array(samples)

    def _compute_forces(self) -> np.ndarray:
        """
        Compute repulsive forces on all original points.

        For periodic boundaries (topological surfaces), uses tiled copies
        to correctly compute forces from neighbors that appear through
        glued edges. Only original points receive forces and are moved.
        """
        n = len(self.points)
        if n <= 1:
            return np.zeros((n, 2))

        # Determine interaction range (auto-compute if not set)
        interaction_range = self.config.interaction_range
        if interaction_range is None:
            tree = KDTree(self.points)
            nn_dists = tree.query(self.points, k=min(2, n))[0]
            if nn_dists.ndim > 1:
                nn_dists = nn_dists[:, -1]
            interaction_range = nn_dists.mean() * 5.0

        forces = np.zeros((n, 2))
        strength = self.config.repulsive_strength

        if self.topology:
            # Create tiled space: original points + copies through each transformation
            # orig_idx maps each tiled point back to its original point index
            tiled_pts, orig_idx = self.topology.create_tiled_points(self.points)
            tree = KDTree(tiled_pts)

            # For each original point, compute force - DEDUPLICATE by physical neighbor.
            # Each physical point (B) may appear as B, B', B'' etc. in tiled space.
            # Only the CLOSEST copy of each neighbor contributes force (B and B' count as 1).
            for i in range(n):
                neighbor_indices = tree.query_ball_point(self.points[i], interaction_range)

                # Group by orig_idx: keep only closest copy per physical neighbor
                best_by_orig: Dict[int, Tuple[int, float, np.ndarray]] = {}
                for j in neighbor_indices:
                    j_orig = orig_idx[j]
                    if j_orig == i:
                        continue  # Skip self

                    vec = self.points[i] - tiled_pts[j]
                    dist_sq = vec[0]**2 + vec[1]**2
                    if dist_sq < 1e-16:
                        vec = self.rng.random(2) - 0.5
                        dist_sq = vec[0]**2 + vec[1]**2
                        if dist_sq < 1e-16:
                            continue

                    dist = sqrt(dist_sq)
                    # Keep only the closest copy of each physical neighbor
                    if j_orig not in best_by_orig or dist < best_by_orig[j_orig][1]:
                        best_by_orig[j_orig] = (j, dist, vec)

                # Add force once per physical neighbor (from closest copy)
                for (_, dist, vec) in best_by_orig.values():
                    dist_sq = dist * dist
                    force_mag = strength / dist_sq
                    forces[i] += (force_mag / dist) * vec
        else:
            # Standard (non-periodic) force computation
            tree = KDTree(self.points)
            for i in range(n):
                neighbor_indices = tree.query_ball_point(self.points[i], interaction_range)
                for j in neighbor_indices:
                    if i == j:
                        continue
                    vec = self.points[i] - self.points[j]
                    dist_sq = vec[0]**2 + vec[1]**2
                    if dist_sq < 1e-16:
                        continue
                    dist = sqrt(dist_sq)
                    force_mag = strength / dist_sq
                    forces[i] += (force_mag / dist) * vec

        return forces

    def evolve_step(self):
        """Execute one evolution step."""
        forces = self._compute_forces()

        # Clamp force magnitudes
        force_mags = np.linalg.norm(forces, axis=1)
        mask = force_mags > self.config.max_force
        if np.any(mask):
            forces[mask] *= (self.config.max_force / force_mags[mask, np.newaxis])

        # Adaptive step size
        if self.config.adaptive_step:
            progress = self.current_iter / self.config.iters
            step = self.config.step_size * (1.0 - progress) ** 2
        else:
            step = self.config.step_size

        # Update positions
        old_points = self.points.copy()
        self.points = self.points + step * forces

        # Wrap points to domain
        if self.topology:
            for i in range(len(self.points)):
                self.points[i] = self.topology.wrap_to_domain(self.points[i])
        else:
            for i in range(len(self.points)):
                pt = Point(self.points[i])
                if not self.domain.contains(pt):
                    nearest = self.domain.boundary.interpolate(
                        self.domain.boundary.project(pt))
                    self.points[i] = np.array([nearest.x, nearest.y])

        # Record metrics
        delta = np.linalg.norm(self.points - old_points, axis=1).mean()
        self.metrics.deltas.append(delta)

        if len(self.points) > 1:
            nn_dists = KDTree(self.points).query(self.points, k=2)[0][:, 1]
            self.metrics.avg_distances.append(nn_dists.mean())
            self.metrics.distance_vars.append(nn_dists.var())

        self.current_iter += 1

    def run_evolution(self, output_dir: str = "."):
        """Run full evolution simulation."""
        plot_dir = os.path.join(output_dir, "evolution_plots")
        os.makedirs(plot_dir, exist_ok=True)

        n_copies = len(self.topology.tiling_transforms) if self.topology else 1
        print(f"[Evolution] Starting {self.config.iters} iterations "
              f"({n_copies} tiling copies)")

        for k in range(self.config.iters):
            self.evolve_step()

            # Save plots at intervals
            should_plot = (self.config.save_plots and
                          (k == 0 or k == self.config.iters - 1 or
                           (k + 1) % self.config.plot_interval == 0))
            if should_plot:
                self._save_plot(os.path.join(plot_dir, f"iter_{k:03d}.pdf"),
                               f"Iteration {k+1}/{self.config.iters}")
                print(f"[Evolution] Iter {k+1}: delta={self.metrics.deltas[-1]:.6f}, "
                      f"avg_nn_dist={self.metrics.avg_distances[-1]:.4f}")

        # Final statistics
        if len(self.points) > 1:
            nn_dists = KDTree(self.points).query(self.points, k=2)[0][:, 1]
            cv = nn_dists.std() / nn_dists.mean() if nn_dists.mean() > 0 else float('inf')
            print(f"[Evolution] Final: CV={cv:.4f}, mean_nn={nn_dists.mean():.4f}")

    def _save_plot(self, path: str, title: str, show_tiling: bool = True):
        """
        Save visualization of current point distribution.

        Args:
            path: Output file path
            title: Plot title
            show_tiling: If True and topology exists, show tiled copies
        """
        fig, ax = plt.subplots(figsize=(12, 12))

        # Draw tiled copies if topology exists and requested
        if show_tiling and self.topology:
            tiled_pts, _ = self.topology.create_tiled_points(self.points)

            # Draw tiled polygon copies (faded)
            for T in self.topology.tiling_transforms[1:]:  # Skip identity
                tiled_poly = T.apply(self.polygon)
                tiled_closed = np.vstack([tiled_poly, tiled_poly[0]])
                ax.plot(tiled_closed[:, 0], tiled_closed[:, 1],
                       color='gray', lw=0.5, alpha=0.3)

            # Draw tiled points (faded, small markers)
            ax.scatter(tiled_pts[len(self.points):, 0],
                      tiled_pts[len(self.points):, 1],
                      s=MARKER_TILED_COPY, c='lightgray', alpha=0.4, zorder=2)

        # Draw main polygon (bold)
        poly_closed = np.vstack([self.polygon, self.polygon[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'k-', lw=2, zorder=4)

        # Color edges by gluing pairs with arrows and letter labels (always shown)
        if self.gluing_rules and self.topology_rule:
            edge_is_reversed = self._get_edge_directions(self.topology_rule)
            edge_labels = self._get_edge_labels(self.topology_rule)
            colors = _get_edge_palette(len(self.gluing_rules))
            arrow_len = ARROW_SCALE * max(self.polygon.max() - self.polygon.min(), 1.0)
            label_offset = EDGE_LABEL_OFFSET * max(self.polygon.max() - self.polygon.min(), 1.0)
            arrow_props = dict(arrowstyle='->', color='k', lw=2.5,
                              mutation_scale=ARROW_HEAD_SCALE)
            centroid = self.polygon.mean(axis=0)
            for idx, (e1, e2, orient) in enumerate(self.gluing_rules):
                color = colors[idx]

                # Draw edges with arrows and labels per gluing rule
                for edge_idx, is_first in [(e1, True), (e2, False)]:
                    p0 = self.polygon[edge_idx]
                    p1 = self.polygon[(edge_idx + 1) % len(self.polygon)]
                    style = '-' if is_first else '--'
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                           color=color, lw=3.5, alpha=0.9, linestyle=style, zorder=4)

                    mid = (p0 + p1) / 2
                    edge_vec = p1 - p0
                    norm = np.linalg.norm(edge_vec)

                    # Arrow: always drawn
                    if norm > 1e-10:
                        direction = (p1 - p0) if not edge_is_reversed[edge_idx] else (p0 - p1)
                        direction = direction / np.linalg.norm(direction) * arrow_len
                        ax.annotate('', xy=mid + direction, xytext=mid - direction,
                                   arrowprops={**arrow_props, 'color': color},
                                   zorder=5)

                    # Letter label: offset outward (away from centroid), closer to edge, bold
                    if norm > 1e-10 and edge_idx < len(edge_labels):
                        to_center = centroid - mid
                        outward = to_center - np.dot(to_center, edge_vec / norm) * (edge_vec / norm)
                        out_norm = np.linalg.norm(outward)
                        if out_norm > 1e-10:
                            outward = -outward / out_norm * label_offset  # negate: outward from polygon
                        else:
                            outward = np.array([edge_vec[1], -edge_vec[0]]) / norm * label_offset
                        label_xy = mid + outward
                        ax.text(label_xy[0], label_xy[1], edge_labels[edge_idx],
                               fontsize=11, ha='center', va='center', color=color,
                               fontweight='bold', zorder=5)

        # Draw main points (smaller when tiling to reduce clutter)
        marker_main = MARKER_MAIN_TILED if (show_tiling and self.topology) else 25
        ax.scatter(self.points[:, 0], self.points[:, 1], s=marker_main, c='blue',
                  alpha=0.8, zorder=6, edgecolors='darkblue', linewidths=0.5)

        # Set axis limits with padding
        all_pts = self.points
        if show_tiling and self.topology:
            tiled_pts, _ = self.topology.create_tiled_points(self.points)
            all_pts = tiled_pts

        margin = 0.1 * max(all_pts.max() - all_pts.min(), 1.0)
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(f"{title}\n({len(self.points)} points, "
                    f"{len(self.topology.tiling_transforms) if self.topology else 1} tiles)")

        plt.tight_layout()
        plt.savefig(path, format='pdf', dpi=150)
        plt.close()

    def build_graph(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Build graph using Voronoi neighbors in tiled space.

        The key insight: we need to find the TRUE neighbors of each original point.
        In tiled space, each original point has copies in each tile. We build the
        Voronoi diagram on ALL tiled points, then for each edge in the Voronoi:
        - We only count edges where at least one endpoint is in copy 0 (original)
        - This finds all neighbors of original points, including cross-boundary ones
        - The neighbor might be in copy 0 (same tile) or another copy (cross-boundary)

        This properly handles the periodic boundaries while avoiding degree explosion.

        Returns:
            nodes: Point coordinates (N, 2)
            edges: List of (u, v) tuples with u < v
        """
        print(f"[Graph] Building graph from {len(self.points)} points...")

        n_points = len(self.points)

        if self.topology:
            tiled_pts, orig_idx = self.topology.create_tiled_points(self.points)
            n_copies = len(self.topology.tiling_transforms)
            print(f"[Graph] Using {len(tiled_pts)} tiled points "
                  f"({n_copies} copies)")

            vor = Voronoi(tiled_pts)

            # Build adjacency from Voronoi ridge points
            # We need to find all neighbors of points in copy 0 (the original)
            edges = set()

            # First, build a list of Voronoi neighbors for each tiled point
            voronoi_neighbors = [[] for _ in range(len(tiled_pts))]
            for ext_i, ext_j in vor.ridge_points:
                voronoi_neighbors[ext_i].append(ext_j)
                voronoi_neighbors[ext_j].append(ext_i)

            # For each original point (copy 0), find its Voronoi neighbors
            for i in range(n_points):
                orig_i = i  # Points 0 to n_points-1 are the original copy

                for ext_j in voronoi_neighbors[i]:
                    # Get the original index of this neighbor
                    orig_j = orig_idx[ext_j]

                    # Don't create self-loops
                    if orig_i != orig_j:
                        edges.add((min(orig_i, orig_j), max(orig_i, orig_j)))
        else:
            vor = Voronoi(self.points)
            edges = set()
            for i, j in vor.ridge_points:
                if i != j:
                    edges.add((min(i, j), max(i, j)))

        print(f"[Graph] Generated {len(edges)} edges")
        return self.points.copy(), list(edges)

    def save_graph_visualization(self, edges: List[Tuple[int, int]], path: str):
        """
        Save visualization of the final graph showing edges and boundary crossings.
        Nodes are colored by BFS index (same palette for both panels; tiled copies at 50% alpha).

        Args:
            edges: List of edge tuples (indices refer to self.points after BFS reorder)
            path: Output file path
        """
        n = len(self.points)
        node_colors = _get_node_palette(n)

        fig, axes = plt.subplots(1, 2, figsize=(30, 15))

        for ax_idx, (ax, show_tiling) in enumerate(zip(axes, [False, True])):
            # Draw tiled space if requested
            if show_tiling and self.topology:
                tiled_pts, orig_idx = self.topology.create_tiled_points(self.points)

                # Draw tiled polygon copies
                for T in self.topology.tiling_transforms[1:]:
                    tiled_poly = T.apply(self.polygon)
                    tiled_closed = np.vstack([tiled_poly, tiled_poly[0]])
                    ax.plot(tiled_closed[:, 0], tiled_closed[:, 1],
                           color='gray', lw=0.5, alpha=0.3)

                # Draw tiled copies: index-based palette at 20% transparency
                tiled_copy_colors = np.array([
                    (*node_colors[orig_idx[i]][:3], 0.2)
                    for i in range(n, len(tiled_pts))
                ])
                ax.scatter(tiled_pts[n:, 0], tiled_pts[n:, 1],
                          s=MARKER_TILED_COPY, c=tiled_copy_colors, zorder=2)

                # Draw edges including cross-boundary ones
                # For each edge (u,v), find the correct tiled copy to connect to:
                # the copy of v closest to u (Voronoi neighbor in tiled space)
                identity_T = self.topology.tiling_transforms[0]
                for u, v in edges:
                    p_u, p_v = self.points[u], self.points[v]
                    # Find best copy of v relative to u (min distance = Voronoi neighbor)
                    best_dist = float('inf')
                    best_p_v = p_v
                    best_T = None
                    for T in self.topology.tiling_transforms:
                        p_v_tiled = T.apply(p_v)
                        d = np.linalg.norm(p_v_tiled - p_u)
                        if d < best_dist:
                            best_dist = d
                            best_p_v = p_v_tiled
                            best_T = T
                    # Red when edge crosses boundary (best copy is not identity)
                    crosses_boundary = best_T is not identity_T
                    if crosses_boundary:
                        ax.plot([p_u[0], best_p_v[0]], [p_u[1], best_p_v[1]],
                               'r-', lw=0.8, alpha=0.6, zorder=3)
                    else:
                        ax.plot([p_u[0], best_p_v[0]], [p_u[1], best_p_v[1]],
                               color='black', lw=0.4, alpha=0.5, zorder=3)
            else:
                # Draw edges in fundamental domain: red for cross-boundary, black for intra-domain
                if self.topology:
                    identity_T = self.topology.tiling_transforms[0]
                    for u, v in edges:
                        p_u, p_v = self.points[u], self.points[v]
                        best_T = None
                        best_dist = float('inf')
                        for T in self.topology.tiling_transforms:
                            p_v_tiled = T.apply(p_v)
                            d = np.linalg.norm(p_v_tiled - p_u)
                            if d < best_dist:
                                best_dist = d
                                best_T = T
                        crosses_boundary = best_T is not identity_T
                        if crosses_boundary:
                            ax.plot([p_u[0], p_v[0]], [p_u[1], p_v[1]],
                                   'r-', lw=0.8, alpha=0.6, zorder=3)
                        else:
                            ax.plot([p_u[0], p_v[0]], [p_u[1], p_v[1]],
                                   color='black', lw=0.5, alpha=0.5, zorder=3)
                else:
                    for u, v in edges:
                        p1, p2 = self.points[u], self.points[v]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                               color='black', lw=0.5, alpha=0.5, zorder=3)

            # Draw main polygon (thinner on right/tiled view)
            poly_closed = np.vstack([self.polygon, self.polygon[0]])
            poly_lw = 0.8 if show_tiling else 2
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'k-', lw=poly_lw, zorder=4)

            # Color edges by gluing with arrows and letter labels (always shown)
            if self.gluing_rules and self.topology_rule:
                edge_is_reversed = self._get_edge_directions(self.topology_rule)
                edge_labels = self._get_edge_labels(self.topology_rule)
                colors = _get_edge_palette(len(self.gluing_rules))
                arrow_len = ARROW_SCALE * max(self.polygon.max() - self.polygon.min(), 1.0)
                label_offset = EDGE_LABEL_OFFSET * max(self.polygon.max() - self.polygon.min(), 1.0)
                edge_lw = 1.0 if show_tiling else 3.5
                arrow_lw = 1.2 if show_tiling else 2.5
                arrow_scale = 10 if show_tiling else ARROW_HEAD_SCALE
                arrow_props = dict(arrowstyle='->', lw=arrow_lw, mutation_scale=arrow_scale)
                centroid = self.polygon.mean(axis=0)
                for idx, (e1, e2, _) in enumerate(self.gluing_rules):
                    color = colors[idx]
                    for edge in [e1, e2]:
                        p0 = self.polygon[edge]
                        p1 = self.polygon[(edge + 1) % len(self.polygon)]
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                               color=color, lw=edge_lw, alpha=0.9, zorder=5)
                        mid = (p0 + p1) / 2
                        edge_vec = p1 - p0
                        norm = np.linalg.norm(edge_vec)
                        # Arrow: always drawn
                        if norm > 1e-10:
                            direction = (p1 - p0) if not edge_is_reversed[edge] else (p0 - p1)
                            direction = direction / np.linalg.norm(direction) * arrow_len
                            ax.annotate('', xy=mid + direction, xytext=mid - direction,
                                       arrowprops={**arrow_props, 'color': color},
                                       zorder=6)
                        # Letter label: offset outward (away from polygon), closer to edge, bold
                        if norm > 1e-10 and edge < len(edge_labels):
                            to_center = centroid - mid
                            outward = to_center - np.dot(to_center, edge_vec / norm) * (edge_vec / norm)
                            out_norm = np.linalg.norm(outward)
                            if out_norm > 1e-10:
                                outward = -outward / out_norm * label_offset  # outward from polygon
                            else:
                                outward = np.array([edge_vec[1], -edge_vec[0]]) / norm * label_offset
                            label_xy = mid + outward
                            ax.text(label_xy[0], label_xy[1], edge_labels[edge],
                                   fontsize=11, ha='center', va='center', color=color,
                                   fontweight='bold', zorder=6)

            # Draw main points: index-based palette (smaller when tiled)
            marker_main = MARKER_MAIN_TILED if show_tiling else MARKER_MAIN
            ax.scatter(self.points[:, 0], self.points[:, 1], s=marker_main, c=node_colors,
                      alpha=0.9, zorder=6, edgecolors='black', linewidths=0.2)

            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.3)
            title = "Graph (fundamental domain)" if ax_idx == 0 else "Graph (tiled view)"
            ax.set_title(f"{title}\n{len(self.points)} nodes, {len(edges)} edges")

        plt.tight_layout()
        plt.savefig(path, format='pdf', dpi=200)
        plt.close()
        print(f"[Graph] Saved visualization to '{path}'")

    @staticmethod
    def reorder_by_bfs(nodes: np.ndarray, edges: List[Tuple[int, int]],
                       start: int = 0) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
        """
        Reorder nodes using BFS traversal for consistent ordering.

        Returns:
            reordered_nodes, reordered_edges, old_to_new_mapping
        """
        n = len(nodes)

        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS
        visited = [False] * n
        order = []
        queue = deque([start])
        visited[start] = True
        order.append(start)

        while queue:
            curr = queue.popleft()
            for neighbor in sorted(adj[curr]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    order.append(neighbor)

        # Add any disconnected nodes
        for i in range(n):
            if not visited[i]:
                order.append(i)

        # Create mapping and reorder
        old_to_new = np.zeros(n, dtype=int)
        for new_id, old_id in enumerate(order):
            old_to_new[old_id] = new_id

        new_nodes = nodes[order]
        new_edges = []
        for u, v in edges:
            new_u, new_v = old_to_new[u], old_to_new[v]
            if new_u > new_v:
                new_u, new_v = new_v, new_u
            new_edges.append((new_u, new_v))

        return new_nodes, list(set(new_edges)), old_to_new


# =============================================================================
# Dataset Export
# =============================================================================

class GraphDatasetExporter:
    """Exports graph data to standard file formats."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, nodes: np.ndarray, edges: List[Tuple[int, int]],
             name: str, metadata: Dict):
        """
        Save complete graph dataset.

        Creates:
        - A_{name}.npy, A_{name}_labeled.csv : Adjacency matrix
        - nodes_{name}.csv : Node info
        - coords_{name}.csv, coords_{name}.npy : Coordinates
        - distance_matrix_{name}.npy : Shortest paths
        - graph_info_{name}.json : Metadata
        """
        print(f"\n[Export] Saving dataset '{name}'...")

        n = len(nodes)

        # Build adjacency matrix
        rows = [e[0] for e in edges] + [e[1] for e in edges]
        cols = [e[1] for e in edges] + [e[0] for e in edges]
        data = [1] * len(rows)
        adj_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        adj_dense = adj_sparse.toarray()

        # Save adjacency
        np.save(f"{self.output_dir}/A_{name}.npy", adj_dense)
        pd.DataFrame(adj_dense).to_csv(f"{self.output_dir}/A_{name}_labeled.csv")

        # Save coordinates
        np.save(f"{self.output_dir}/coords_{name}.npy", nodes)
        coords_3d = np.column_stack([nodes, np.zeros(n)])
        pd.DataFrame(coords_3d, columns=['x', 'y', 'z']).to_csv(
            f"{self.output_dir}/coords_{name}.csv", index_label='node_id')

        # Save node info
        degrees = np.array(adj_sparse.sum(axis=1)).flatten()
        pd.DataFrame({
            'node_id': range(n),
            'degree': degrees,
            'type': 'manifold_point'
        }).set_index('node_id').to_csv(f"{self.output_dir}/nodes_{name}.csv")

        # Compute and save distances
        print("[Export] Computing shortest paths...")
        dist_matrix = shortest_path(adj_sparse, directed=False, unweighted=True)
        max_finite = np.max(dist_matrix[np.isfinite(dist_matrix)])
        if np.isinf(max_finite) or max_finite == 0:
            max_finite = n * 2
        dist_matrix = np.where(np.isfinite(dist_matrix), dist_matrix, max_finite + 1)
        np.fill_diagonal(dist_matrix, 0)
        np.save(f"{self.output_dir}/distance_matrix_{name}.npy", dist_matrix)

        # Save metadata
        info = {
            "dataset_name": name,
            "num_nodes": int(n),
            "num_edges": len(edges),
            "avg_degree": float(degrees.mean()),
            "std_degree": float(degrees.std()),
            "min_degree": int(degrees.min()),
            "max_degree": int(degrees.max()),
            **metadata
        }
        with open(f"{self.output_dir}/graph_info_{name}.json", 'w') as f:
            json.dump(info, f, indent=4)

        print(f"[Export] Degree stats: mean={degrees.mean():.2f}, "
              f"range=[{degrees.min()}, {degrees.max()}]")
        print(f"[Export] Files saved to '{self.output_dir}/'")


# =============================================================================
# Command Line Interface
# =============================================================================

def detect_topology_name(rule: str) -> str:
    """Detect common topology names from gluing rules."""
    patterns = {
        "torus": ["abAB", "abABcdCD", "abABcdCDefEF"],
        "klein": ["abAb", "abaB"],
        "sphere": ["abBA", "aAbB"],
        "projective": ["abab"]
    }
    for name, rules in patterns.items():
        if rule in rules:
            return name
    return rule.lower()[:8]


def estimate_point_count(n: int, density: float = 1.0) -> int:
    """Estimate number of points based on polygon area."""
    vertices = FundamentalPolygon._create_polygon(n, 1.0)
    area = ShapelyPolygon(vertices).area
    count = max(50, int(area * density * 10))
    print(f"[Config] Estimated {count} points for area {area:.4f}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Generate graph from topological surface")
    parser.add_argument("--topology", type=str, default=None,
                       help="Gluing rule (e.g., 'abAB' for torus)")
    parser.add_argument("--prefix", type=str, default=None,
                       help="Dataset name prefix")
    parser.add_argument("--N_total", type=int, default=None,
                       help="Number of points")
    parser.add_argument("--density_factor", type=float, default=1.0,
                       help="Point density factor")
    parser.add_argument("--iters", type=int, default=200,
                       help="Evolution iterations")
    parser.add_argument("--tiling_layers", type=int, default=3,
                       help="Tiling layers for neighbor detection")
    parser.add_argument("--step_size", type=float, default=0.05,
                       help="Evolution step size")
    parser.add_argument("--plot_interval", type=int, default=10,
                       help="Plot saving interval")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory")
    args = parser.parse_args()

    # Determine polygon size from topology
    if args.topology:
        num_edges = len(re.findall(r"[a-zA-Z]", args.topology))
        if num_edges % 2 != 0:
            raise ValueError(f"Topology '{args.topology}' has odd number of edges")
        n = num_edges // 2
        print(f"[Config] Topology '{args.topology}' -> {num_edges}-gon (n={n})")
    else:
        n = 2
        print("[Config] No topology, using square (n=2)")

    # Determine point count
    if args.N_total is None:
        args.N_total = estimate_point_count(n, args.density_factor)

    # Create configuration
    config = EvolutionConfig(
        iters=args.iters,
        step_size=args.step_size,
        tiling_layers=args.tiling_layers,
        plot_interval=args.plot_interval
    )

    # Create and evolve polygon
    poly = FundamentalPolygon(
        n=n,
        num_points=args.N_total,
        topology_rule=args.topology,
        config=config,
        seed=args.seed
    )

    poly.run_evolution(output_dir=args.output_dir)

    # Build and export graph
    if args.topology:
        nodes, edges = poly.build_graph()

        # BFS reorder immediately after graph construction (before visualization and export)
        nodes, edges, _ = FundamentalPolygon.reorder_by_bfs(nodes, edges)
        poly.points = nodes  # Update for visualization (nodes now in BFS order)

        # Save graph visualization with BFS-index coloring
        graph_viz_path = os.path.join(args.output_dir, "graph_visualization.pdf")
        poly.save_graph_visualization(edges, graph_viz_path)

        # Print degree statistics
        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        degrees = [len(adj.get(i, [])) for i in range(len(nodes))]
        print(f"[Graph] Degree distribution: mean={np.mean(degrees):.2f}, "
              f"range=[{min(degrees)}, {max(degrees)}]")

        # Determine dataset name
        prefix = args.prefix or detect_topology_name(args.topology)
        dataset_name = f"{prefix}_{args.topology}_N{args.N_total}_iter{args.iters}"

        # Export
        metadata = {
            "topology_rule": args.topology,
            "n_polygon": n,
            "tiling_layers": args.tiling_layers,
            "seed": args.seed
        }
        exporter = GraphDatasetExporter(args.output_dir)
        exporter.save(nodes, edges, dataset_name, metadata)


if __name__ == "__main__":
    main()
