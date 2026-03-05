#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01a_graph_generation.py - Generate graph representation of topological manifolds

This script creates graph structures from fundamental polygons with edge-gluing rules,
representing topological surfaces like torus, Klein bottle, double torus, etc.

Approach:
1. Start with a regular 2n-polygon (fundamental polygon)
2. Sample points uniformly inside (no fixed boundary points)
3. Lloyd's relaxation: move each point toward the centroid of its Voronoi cell
   (optionally with decaying noise). Stop when mean move < tol or max_iters.
4. Final graph = Voronoi neighbours at the converged positions

Key features:
- Affine transformations handle edge gluing with correct orientation
- Multi-layer tiling ensures accurate neighbor detection across boundaries
- Snake-game wrapping for point evolution with proper edge identification
- Degree-uniform graphs: graph is the primary object throughout evolution

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
    """Parameters for Lloyd relaxation, noise, and saving/visualization."""

    max_iters: int = 2000         # Safety cap; stop when delta < convergence_tol or this
    convergence_tol: float = 1e-6     # Stop when mean move < this
    lloyd_step_size: float = 1.0       # Step toward centroid (1.0 = full step)
    tiling_layers: int = 3        # Layers of tiled copies for neighbour detection
    target_degree: int = 6       # Desired Voronoi degree (metadata / display)

    save_plots: bool = True       # Save evolution visualizations
    plot_interval: int = 10       # Iterations between saved plots
    log_interval: int = 1         # Iterations between progress logs (delta, degrees)

    noise_strength: float = 0.02   # Per-step perturbation (0 to disable); decays over iters
    noise_decay_power: float = 2.0  # Noise scaled by (1 - progress)^this


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
        self.graph_edges: Set[Tuple[int, int]] = set()  # Updated each step by _compute_voronoi_graph()

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

    def _compute_voronoi_graph(
        self,
        tiled_pts: Optional[np.ndarray],
        orig_idx: Optional[np.ndarray],
    ) -> Tuple[Set[Tuple[int, int]], List[int]]:
        """Compute the current Voronoi graph from pre-built tiled space.

        Uses the tiled_pts/orig_idx computed once per step in evolve_step so
        the tiling is not duplicated between graph construction and force computation.

        Returns:
            edges:  set of (u, v) pairs with u < v
            degree: list of per-node Voronoi degree
        """
        n = len(self.points)
        edges: Set[Tuple[int, int]] = set()

        if self.topology and tiled_pts is not None and len(tiled_pts) >= 4:
            vor = Voronoi(tiled_pts)

            # Build per-tiled-point neighbour list from all Voronoi ridges.
            vor_nbrs: List[List[int]] = [[] for _ in range(len(tiled_pts))]
            for ext_i, ext_j in vor.ridge_points:
                vor_nbrs[ext_i].append(ext_j)
                vor_nbrs[ext_j].append(ext_i)

            # For each ORIGINAL-COPY node i (indices 0..n-1) collect its
            # Voronoi neighbours in the tiled space.  The same physical node j
            # may appear via several tiled copies (e.g. tiled[j], tiled[j+n],
            # tiled[j+2n]).  Keep only the CLOSEST copy per physical neighbour
            # so that each physical pair is counted exactly once — mirroring
            # the best_by_orig logic used for Lloyd step.
            for i in range(n):
                closest: Dict[int, float] = {}   # j_orig → min squared dist
                for ext_j in vor_nbrs[i]:
                    j_orig = int(orig_idx[ext_j])
                    if j_orig == i:
                        continue
                    vec = self.points[i] - tiled_pts[ext_j]
                    d2 = float(vec[0] * vec[0] + vec[1] * vec[1])
                    if j_orig not in closest or d2 < closest[j_orig]:
                        closest[j_orig] = d2
                for j_orig in closest:
                    edges.add((min(i, j_orig), max(i, j_orig)))
        elif n >= 4:
            vor = Voronoi(self.points)
            for ext_i, ext_j in vor.ridge_points:
                if ext_i != ext_j:
                    edges.add((min(ext_i, ext_j), max(ext_i, ext_j)))

        degree = [0] * n
        for i, j in edges:
            degree[i] += 1
            degree[j] += 1

        return edges, degree

    def _compute_lloyd_step(
        self,
        tiled_pts: np.ndarray,
        orig_idx: np.ndarray,
    ) -> np.ndarray:
        """Lloyd's relaxation: move each point toward the centroid of its Voronoi cell.

        Computes the Voronoi diagram on tiled_pts; for each original point i,
        takes the region of the i-th site, clips it to a bounding box if
        unbounded, and sets the target position to the polygon centroid.
        Returns step_vector = target_positions - self.points (to be scaled by
        step_size in evolve_step).
        """
        n = len(self.points)
        if n <= 1:
            return np.zeros((n, 2))

        vor = Voronoi(tiled_pts)
        target_positions = np.zeros((n, 2))

        # Bounding box for clipping unbounded Voronoi regions
        xmin, ymin = tiled_pts.min(axis=0)
        xmax, ymax = tiled_pts.max(axis=0)
        margin = max(xmax - xmin, ymax - ymin) * 0.5
        box = ShapelyPolygon([
            (xmin - margin, ymin - margin),
            (xmax + margin, ymin - margin),
            (xmax + margin, ymax + margin),
            (xmin - margin, ymax + margin),
        ])

        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            # Finite vertices in order (preserve order, drop -1)
            ordered_finite = [vor.vertices[j] for j in region if j >= 0]
            if len(ordered_finite) < 3:
                target_positions[i] = self.points[i].copy()
                continue
            poly = ShapelyPolygon(ordered_finite)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < 1e-20:
                target_positions[i] = self.points[i].copy()
                continue
            clipped = poly.intersection(box)
            if clipped.is_empty or clipped.area < 1e-20:
                target_positions[i] = np.array(ordered_finite).mean(axis=0)
            else:
                cen = clipped.centroid
                target_positions[i] = np.array([cen.x, cen.y])

        # Wrap targets to fundamental domain
        if self.topology:
            for i in range(n):
                target_positions[i] = self.topology.wrap_to_domain(target_positions[i])
        else:
            for i in range(n):
                pt = Point(target_positions[i])
                if not self.domain.contains(pt):
                    nearest = self.domain.boundary.interpolate(
                        self.domain.boundary.project(pt))
                    target_positions[i] = np.array([nearest.x, nearest.y])

        return target_positions - self.points

    def evolve_step(self):
        """Execute one Lloyd step: Voronoi graph, move toward cell centroids, optional noise, wrap."""
        if self.topology:
            tiled_pts, orig_idx = self.topology.create_tiled_points(self.points)
        else:
            tiled_pts = self.points
            orig_idx = np.arange(len(self.points))

        self.graph_edges, _ = self._compute_voronoi_graph(tiled_pts, orig_idx)
        step_vector = self._compute_lloyd_step(tiled_pts, orig_idx)
        step = self.config.lloyd_step_size

        # Update positions
        old_points = self.points.copy()
        self.points = self.points + step * step_vector

        # Decaying noise: small perturbation per step (helps avoid local minima)
        if self.config.noise_strength > 0:
            progress = self.current_iter / max(1, self.config.max_iters)
            noise_scale = self.config.noise_strength * (
                (1.0 - progress) ** self.config.noise_decay_power
            )
            n_pts = len(self.points)
            self.points += noise_scale * (self.rng.random((n_pts, 2)) - 0.5) * 2.0

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

    def run_evolution(self, output_dir: str = ".") -> int:
        """Run layout evolution. Returns actual number of iterations performed.

        Lloyd: runs until mean delta < convergence_tol or max_iters (no fixed count).
        F-R: runs exactly max_iters steps.
        """
        plot_dir = os.path.join(output_dir, "evolution_plots")
        os.makedirs(plot_dir, exist_ok=True)

        n_copies = len(self.topology.tiling_transforms) if self.topology else 1
        print(f"[Evolution] Lloyd: stop when delta < {self.config.convergence_tol} "
              f"(max {self.config.max_iters} steps), {n_copies} tiling copies")
        print(f"  step_size={self.config.lloyd_step_size}, noise={self.config.noise_strength}")
        best_points: Optional[np.ndarray] = None
        best_graph_edges: Optional[Set[Tuple[int, int]]] = None
        best_count_at_target = -1
        best_count_outside_pm1 = float("inf")
        best_iter = -1

        # Detailed per-step log for later visualization (delta, degrees, etc.)
        evolution_steps: List[Dict] = []

        k = 0
        for k in range(self.config.max_iters):
            self.evolve_step()
            delta = self.metrics.deltas[-1]

            # Track best step (fewest outside ±1, then most at target)
            if self.graph_edges:
                deg = [0] * len(self.points)
                for u, v in self.graph_edges:
                    deg[u] += 1
                    deg[v] += 1
                target = self.config.target_degree
                count_at_target = sum(1 for d in deg if d == target)
                count_outside_pm1 = sum(
                    1 for d in deg if abs(d - target) > 1
                )
                # Prefer fewer outliers, then more at target
                is_better = (
                    count_outside_pm1 < best_count_outside_pm1
                    or (
                        count_outside_pm1 == best_count_outside_pm1
                        and count_at_target >= best_count_at_target
                    )
                )
                if is_better:
                    best_count_at_target = count_at_target
                    best_count_outside_pm1 = count_outside_pm1
                    best_points = self.points.copy()
                    best_graph_edges = set(self.graph_edges)
                    best_iter = k + 1

            # Record step for evolution_metadata.json (delta, avg_nn_dist, degree_counts)
            avg_nn = float(self.metrics.avg_distances[-1]) if self.metrics.avg_distances else 0.0
            deg_count_dict: Dict[int, int] = {}
            if self.graph_edges:
                deg = [0] * len(self.points)
                for u, v in self.graph_edges:
                    deg[u] += 1
                    deg[v] += 1
                for d in deg:
                    deg_count_dict[d] = deg_count_dict.get(d, 0) + 1
            # JSON-friendly: degree keys as strings, values as ints
            degree_counts = {str(dk): dv for dk, dv in sorted(deg_count_dict.items())}
            mean_deg = (
                sum(d * c for d, c in deg_count_dict.items()) / sum(deg_count_dict.values())
                if deg_count_dict else 0.0
            )
            evolution_steps.append({
                "iter": k + 1,
                "delta": float(delta),
                "avg_nn_dist": avg_nn,
                "degree_counts": degree_counts,
                "mean_degree": float(mean_deg),
            })

            if delta < self.config.convergence_tol:
                print(f"[Evolution] Lloyd converged at iter {k+1}: delta={delta:.2e}")
                if self.config.save_plots:
                    self._save_plot(
                        os.path.join(plot_dir, f"iter_{k:03d}.pdf"),
                        f"Iteration {k+1} (converged)",
                        edges=self.graph_edges,
                    )
                k += 1  # actual_iters
                break

            # Log progress at log_interval (delta, degree stats)
            should_log = (
                k == 0 or k == self.config.max_iters - 1 or
                (k + 1) % self.config.log_interval == 0
            )
            if should_log:
                deg_info = ""
                if self.graph_edges:
                    deg = [0] * len(self.points)
                    for u, v in self.graph_edges:
                        deg[u] += 1
                        deg[v] += 1
                    deg_count: Dict[int, int] = {}
                    for d in deg:
                        deg_count[d] = deg_count.get(d, 0) + 1
                    deg_str = ", ".join(
                        f"{dk}: {dv}" for dk, dv in sorted(deg_count.items())
                    )
                    deg_info = f"  mean={np.mean(deg):.1f}  [{deg_str}]"
                print(f"[Evolution] Iter {k+1}: delta={delta:.6f}, "
                      f"avg_nn_dist={self.metrics.avg_distances[-1]:.4f}{deg_info}")

            # Save plots at plot_interval
            should_plot = (self.config.save_plots and
                          (k == 0 or k == self.config.max_iters - 1 or
                           (k + 1) % self.config.plot_interval == 0))
            if should_plot:
                self._save_plot(os.path.join(plot_dir, f"iter_{k:03d}.pdf"),
                               f"Iteration {k+1}/{self.config.max_iters}",
                               edges=self.graph_edges)
        else:
            k += 1  # loop completed without break

        if best_points is not None and best_graph_edges is not None:
            self.points = best_points
            self.graph_edges = best_graph_edges
            n_at_target = best_count_at_target
            n_outside = best_count_outside_pm1
            total = len(self.points)
            print(f"[Evolution] Restored best step (iter {best_iter}): "
                  f"{n_at_target}/{total} at target degree {self.config.target_degree}, "
                  f"{n_outside} outside ±1")
            if self.config.save_plots:
                self._save_plot(
                    os.path.join(plot_dir, "best_step.pdf"),
                    f"Best step (iter {best_iter}): {n_at_target}/{total} at target, {n_outside} outside ±1",
                    edges=self.graph_edges,
                )
                print(f"[Evolution] Best-step visualization saved to {plot_dir}/best_step.pdf")

        # Final statistics
        if len(self.points) > 1:
            nn_dists = KDTree(self.points).query(self.points, k=2)[0][:, 1]
            cv = nn_dists.std() / nn_dists.mean() if nn_dists.mean() > 0 else float('inf')
            print(f"[Evolution] Final: {k} iters, CV={cv:.4f}, mean_nn={nn_dists.mean():.4f}")

        # Save detailed evolution metadata for later visualization
        meta = {
            "config": {
                "target_degree": self.config.target_degree,
                "max_iters": self.config.max_iters,
                "log_interval": self.config.log_interval,
                "plot_interval": self.config.plot_interval,
                "noise_strength": self.config.noise_strength,
                "noise_decay_power": self.config.noise_decay_power,
                "n_points": len(self.points),
            },
            "steps": evolution_steps,
            "actual_iters": k,
        }
        if best_iter >= 0:
            meta["lloyd_best_iter"] = best_iter
            meta["lloyd_best_count_at_target"] = best_count_at_target
            meta["lloyd_best_count_outside_pm1"] = best_count_outside_pm1
        meta_path = os.path.join(output_dir, "evolution_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Evolution] Metadata saved to {meta_path}")

        return k

    def _save_plot(self, path: str, title: str, show_tiling: bool = True,
                  edges: Optional[Set] = None):
        """Save evolution snapshot.

        Left panel (3/4 width): spatial view — polygon, tiled copies, graph edges,
            nodes coloured by degree (green=target, gold=±1, tomato=other).
        Right panel (1/4 width): degree distribution bar chart.

        Args:
            path:         Output file path (.pdf)
            title:        Plot title string
            show_tiling:  If True and topology exists, render tiled copies
            edges:        Current graph edge set; if None nodes are drawn plain blue
        """
        has_edges = edges is not None and len(edges) > 0
        target_k = self.config.target_degree

        # Compute per-node degree from edge set
        degree_count: Dict[int, int] = {i: 0 for i in range(len(self.points))}
        if has_edges:
            for u, v in edges:
                degree_count[u] += 1
                degree_count[v] += 1

        fig, (ax, ax_hist) = plt.subplots(
            1, 2, figsize=(18, 9),
            gridspec_kw={'width_ratios': [3, 1]}
        )

        # ------------------------------------------------------------------
        # Left panel: spatial view
        # ------------------------------------------------------------------

        # Tiled polygon copies and tiled points (faded); cache for axis limits reuse
        tiled_pts_cache = None
        if show_tiling and self.topology:
            tiled_pts_cache, _ = self.topology.create_tiled_points(self.points)
            for T in self.topology.tiling_transforms[1:]:
                tiled_poly = T.apply(self.polygon)
                tiled_closed = np.vstack([tiled_poly, tiled_poly[0]])
                ax.plot(tiled_closed[:, 0], tiled_closed[:, 1],
                       color='gray', lw=0.5, alpha=0.3)
            ax.scatter(tiled_pts_cache[len(self.points):, 0],
                      tiled_pts_cache[len(self.points):, 1],
                      s=MARKER_TILED_COPY, c='lightgray', alpha=0.4, zorder=2)

        # Main polygon boundary
        poly_closed = np.vstack([self.polygon, self.polygon[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'k-', lw=2, zorder=4)

        # Gluing arrows and labels
        if self.gluing_rules and self.topology_rule:
            edge_is_reversed = self._get_edge_directions(self.topology_rule)
            edge_labels_list = self._get_edge_labels(self.topology_rule)
            colors = _get_edge_palette(len(self.gluing_rules))
            arrow_len = ARROW_SCALE * max(self.polygon.max() - self.polygon.min(), 1.0)
            label_offset = EDGE_LABEL_OFFSET * max(self.polygon.max() - self.polygon.min(), 1.0)
            arrow_props = dict(arrowstyle='->', color='k', lw=2.5,
                              mutation_scale=ARROW_HEAD_SCALE)
            centroid = self.polygon.mean(axis=0)
            for idx, (e1, e2, orient) in enumerate(self.gluing_rules):
                color = colors[idx]
                for edge_idx, is_first in [(e1, True), (e2, False)]:
                    p0 = self.polygon[edge_idx]
                    p1 = self.polygon[(edge_idx + 1) % len(self.polygon)]
                    style = '-' if is_first else '--'
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                           color=color, lw=3.5, alpha=0.9, linestyle=style, zorder=4)
                    mid = (p0 + p1) / 2
                    edge_vec = p1 - p0
                    norm = np.linalg.norm(edge_vec)
                    if norm > 1e-10:
                        direction = (p1 - p0) if not edge_is_reversed[edge_idx] else (p0 - p1)
                        direction = direction / np.linalg.norm(direction) * arrow_len
                        ax.annotate('', xy=mid + direction, xytext=mid - direction,
                                   arrowprops={**arrow_props, 'color': color}, zorder=5)
                    if norm > 1e-10 and edge_idx < len(edge_labels_list):
                        to_center = centroid - mid
                        outward = to_center - np.dot(to_center, edge_vec / norm) * (edge_vec / norm)
                        out_norm = np.linalg.norm(outward)
                        if out_norm > 1e-10:
                            outward = -outward / out_norm * label_offset
                        else:
                            outward = np.array([edge_vec[1], -edge_vec[0]]) / norm * label_offset
                        ax.text((mid + outward)[0], (mid + outward)[1],
                               edge_labels_list[edge_idx],
                               fontsize=11, ha='center', va='center', color=color,
                               fontweight='bold', zorder=5)

        # Graph edges
        if has_edges:
            identity_T = self.topology.tiling_transforms[0] if self.topology else None
            for u, v in edges:
                p_u = self.points[u]
                p_v = self.points[v]
                if self.topology:
                    # Find tiled copy of v closest to u
                    best_dist = float('inf')
                    best_p_v = p_v
                    best_T = None
                    for T in self.topology.tiling_transforms:
                        p_v_t = T.apply(p_v)
                        d = np.linalg.norm(p_v_t - p_u)
                        if d < best_dist:
                            best_dist = d
                            best_p_v = p_v_t
                            best_T = T

                    crosses = best_T is not identity_T
                    color = 'firebrick' if crosses else '#444444'

                    if crosses:
                        # Cross-boundary edge: draw BOTH stubs so both endpoints
                        # are shown exiting toward their respective glued boundary.
                        #   Stub 1: p_u  →  best_T(p_v)   [u exits toward v's tile]
                        #   Stub 2: p_v  →  best_T⁻¹(p_u) [v exits toward u's tile]
                        p_u_from_v = best_T.inverse().apply(p_u)
                        ax.plot([p_u[0], best_p_v[0]], [p_u[1], best_p_v[1]],
                               color=color, lw=0.9, alpha=0.55, zorder=3)
                        ax.plot([p_v[0], p_u_from_v[0]], [p_v[1], p_u_from_v[1]],
                               color=color, lw=0.9, alpha=0.55, zorder=3)
                    else:
                        # Intra-domain: straight line between the two points
                        ax.plot([p_u[0], best_p_v[0]], [p_u[1], best_p_v[1]],
                               color=color, lw=0.9, alpha=0.55, zorder=3)
                else:
                    ax.plot([p_u[0], p_v[0]], [p_u[1], p_v[1]],
                           '#444444', lw=0.9, alpha=0.55, zorder=3)

        # Nodes coloured by degree
        marker_main = MARKER_MAIN_TILED if (show_tiling and self.topology) else MARKER_MAIN
        if has_edges:
            node_colors_deg = []
            for i in range(len(self.points)):
                d = degree_count.get(i, 0)
                if d == target_k:
                    node_colors_deg.append('limegreen')
                elif abs(d - target_k) == 1:
                    node_colors_deg.append('gold')
                else:
                    node_colors_deg.append('tomato')
            ax.scatter(self.points[:, 0], self.points[:, 1], s=marker_main,
                      c=node_colors_deg, alpha=0.9, zorder=6,
                      edgecolors='black', linewidths=0.4)
        else:
            ax.scatter(self.points[:, 0], self.points[:, 1], s=marker_main,
                      c='steelblue', alpha=0.85, zorder=6,
                      edgecolors='darkblue', linewidths=0.5)

        # Axis limits — reuse cached tiled_pts if available
        all_pts = self.points if tiled_pts_cache is None else tiled_pts_cache
        margin = 0.1 * max(all_pts.max() - all_pts.min(), 1.0)
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)

        n_edges = len(edges) if has_edges else 0
        ax.set_title(f"{title}  ({len(self.points)} nodes, {n_edges} edges)",
                    fontsize=10)

        # ------------------------------------------------------------------
        # Right panel: degree histogram
        # ------------------------------------------------------------------
        if has_edges:
            dvals = list(degree_count.values())
            all_degs = sorted(set(dvals))
            counts = [dvals.count(k) for k in all_degs]
            bar_colors = []
            for k in all_degs:
                if k == target_k:
                    bar_colors.append('limegreen')
                elif abs(k - target_k) == 1:
                    bar_colors.append('gold')
                else:
                    bar_colors.append('tomato')
            ax_hist.bar(all_degs, counts, color=bar_colors,
                       edgecolor='black', linewidth=0.6)
            ax_hist.axvline(target_k, color='royalblue', linestyle='--',
                           lw=1.5, label=f'target={target_k}')
            ax_hist.set_xlabel('Degree', fontsize=11)
            ax_hist.set_ylabel('Count', fontsize=11)
            ax_hist.set_title('Degree distribution', fontsize=11)
            ax_hist.legend(fontsize=9)
            ax_hist.set_xticks(all_degs)
            ax_hist.grid(True, axis='y', linestyle='--', alpha=0.4)
        else:
            ax_hist.axis('off')

        plt.tight_layout()
        plt.savefig(path, format='pdf', dpi=150)
        plt.close()

    def build_graph(self) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Return the Voronoi graph at the current point positions.

        After run_evolution(), self.graph_edges holds the Voronoi graph from
        the final iteration. Falls back to computing it fresh if not yet set
        (e.g. when evolution was skipped).

        Returns:
            nodes: Point coordinates (N, 2)
            edges: List of (u, v) tuples with u < v
        """
        if self.graph_edges:
            print(f"[Graph] Returning evolved graph: "
                  f"{len(self.graph_edges)} edges")
            return self.points.copy(), list(self.graph_edges)

        # Fallback: compute Voronoi graph now
        print(f"[Graph] Computing Voronoi graph from {len(self.points)} points...")
        if self.topology:
            tiled_pts, orig_idx = self.topology.create_tiled_points(self.points)
        else:
            tiled_pts, orig_idx = None, None
        edges, _ = self._compute_voronoi_graph(tiled_pts, orig_idx)
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

                # Draw edges including cross-boundary ones.
                # Cross-boundary edges are drawn as TWO stubs — one from each endpoint
                # exiting toward its respective glued boundary — so both sides are visible.
                identity_T = self.topology.tiling_transforms[0]
                for u, v in edges:
                    p_u, p_v = self.points[u], self.points[v]
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
                    crosses_boundary = best_T is not identity_T
                    if crosses_boundary:
                        # Stub 1: p_u → best_T(p_v)        [u exits toward v's tile]
                        # Stub 2: p_v → best_T⁻¹(p_u)     [v exits toward u's tile]
                        p_u_from_v = best_T.inverse().apply(p_u)
                        ax.plot([p_u[0], best_p_v[0]], [p_u[1], best_p_v[1]],
                               'r-', lw=0.8, alpha=0.6, zorder=3)
                        ax.plot([p_v[0], p_u_from_v[0]], [p_v[1], p_u_from_v[1]],
                               'r-', lw=0.8, alpha=0.6, zorder=3)
                    else:
                        ax.plot([p_u[0], best_p_v[0]], [p_u[1], best_p_v[1]],
                               color='black', lw=0.4, alpha=0.5, zorder=3)
            else:
                # Draw edges in fundamental domain.
                # Cross-boundary edges: TWO stubs, each endpoint exits toward its boundary.
                # Intra-domain edges: straight line between the two points.
                if self.topology:
                    identity_T = self.topology.tiling_transforms[0]
                    for u, v in edges:
                        p_u, p_v = self.points[u], self.points[v]
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
                        crosses_boundary = best_T is not identity_T
                        if crosses_boundary:
                            p_u_from_v = best_T.inverse().apply(p_u)
                            ax.plot([p_u[0], best_p_v[0]], [p_u[1], best_p_v[1]],
                                   'r-', lw=0.8, alpha=0.6, zorder=3)
                            ax.plot([p_v[0], p_u_from_v[0]], [p_v[1], p_u_from_v[1]],
                                   'r-', lw=0.8, alpha=0.6, zorder=3)
                        else:
                            ax.plot([p_u[0], best_p_v[0]], [p_u[1], best_p_v[1]],
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

        deg_count: Dict[int, int] = {}
        for d in degrees.flat:
            deg_count[int(d)] = deg_count.get(int(d), 0) + 1
        deg_str = ", ".join(f"deg {k}: {v}" for k, v in sorted(deg_count.items()))
        print(f"[Export] Degree stats: mean={degrees.mean():.2f}, "
              f"range=[{degrees.min()}, {degrees.max()}]  [{deg_str}]")
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
    parser.add_argument("--max_iters", type=int, default=2000,
                       help="Lloyd: safety cap; F-R: exact iteration count")
    parser.add_argument("--convergence_tol", type=float, default=1e-6,
                       help="Lloyd only: stop when mean delta < this")
    parser.add_argument("--lloyd_step_size", type=float, default=1.0,
                       help="Step toward centroid (1.0 = full)")
    parser.add_argument("--tiling_layers", type=int, default=3,
                       help="Tiling layers for neighbor detection")
    parser.add_argument("--target_degree", type=int, default=6,
                       help="Target Voronoi degree (metadata)")
    parser.add_argument("--noise_strength", type=float, default=0.02,
                       help="Per-step position perturbation (0 to disable; decays over iters)")
    parser.add_argument("--noise_decay_power", type=float, default=2.0,
                       help="Noise scaled by (1 - progress)^this")
    parser.add_argument("--plot_interval", type=int, default=10,
                       help="Iterations between saved plots")
    parser.add_argument("--log_interval", type=int, default=1,
                       help="Iterations between progress logs (delta, degrees)")
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

    config = EvolutionConfig(
        max_iters=args.max_iters,
        convergence_tol=args.convergence_tol,
        lloyd_step_size=args.lloyd_step_size,
        tiling_layers=args.tiling_layers,
        target_degree=args.target_degree,
        plot_interval=args.plot_interval,
        log_interval=args.log_interval,
        noise_strength=args.noise_strength,
        noise_decay_power=args.noise_decay_power,
    )

    # Create and evolve polygon
    poly = FundamentalPolygon(
        n=n,
        num_points=args.N_total,
        topology_rule=args.topology,
        config=config,
        seed=args.seed
    )

    actual_iters = poly.run_evolution(output_dir=args.output_dir)

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
        deg_count: Dict[int, int] = {}
        for d in degrees:
            deg_count[d] = deg_count.get(d, 0) + 1
        deg_str = ", ".join(f"deg {k}: {v}" for k, v in sorted(deg_count.items()))
        print(f"[Graph] Degree distribution: mean={np.mean(degrees):.2f}, "
              f"range=[{min(degrees)}, {max(degrees)}]  [{deg_str}]")

        # Determine dataset name
        prefix = args.prefix or detect_topology_name(args.topology)
        dataset_name = f"{prefix}_{args.topology}_N{args.N_total}_iter{args.max_iters}"

        # Export
        metadata = {
            "topology_rule": args.topology,
            "n_polygon": n,
            "tiling_layers": args.tiling_layers,
            "max_iters": args.max_iters,
            "actual_iters": actual_iters,
            "seed": args.seed,
            "target_degree": args.target_degree,
        }
        exporter = GraphDatasetExporter(args.output_dir)
        exporter.save(nodes, edges, dataset_name, metadata)


if __name__ == "__main__":
    main()
