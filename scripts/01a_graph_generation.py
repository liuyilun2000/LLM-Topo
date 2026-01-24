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
import math
import re
from typing import Tuple, List, Optional, Set
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, KDTree, Delaunay
from shapely.geometry import Polygon as ShapelyPolygon, Point
from shapely.geometry import Point as SPoint, Polygon
import os
import json
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from math import cos, sin, pi, sqrt


# =============================================================================
#  Basic Geometry Functions
# =============================================================================

def regular_2n_polygon(n: int, radius: float = 1.0) -> np.ndarray:
    """Construct regular 2n-polygon, vertices in counterclockwise order."""
    m = 2 * n
    vertices = []
    for k in range(m):
        theta = 2 * pi * k / m
        vertices.append([radius * cos(theta), radius * sin(theta)])
    return np.array(vertices, dtype=float)


def polygon_area(vertices: np.ndarray) -> float:
    """Compute polygon area using shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def triangulate_fan(vertices: np.ndarray) -> List[Tuple[int, int, int]]:
    """Fan triangulation of convex polygon using vertex 0 as center."""
    m = len(vertices)
    tris = []
    for i in range(1, m - 1):
        tris.append((0, i, i + 1))
    return tris


def sample_interior_points(vertices: np.ndarray, N_interior: int, rng: np.random.Generator) -> np.ndarray:
    """Sample N_interior points uniformly inside polygon using triangulation."""
    if N_interior <= 0:
        return np.zeros((0, 2), dtype=float)
    tris = triangulate_fan(vertices)
    tri_areas = []
    for (i, j, k) in tris:
        A = polygon_area(vertices[[i, j, k]])
        tri_areas.append(A)
    tri_areas = np.array(tri_areas, dtype=float)
    probs = tri_areas / tri_areas.sum()
    samples = []
    for _ in range(N_interior):
        t_idx = rng.choice(len(tris), p=probs)
        i, j, k = tris[t_idx]
        a, b, c = vertices[i], vertices[j], vertices[k]
        u = rng.random()
        v = rng.random()
        su = sqrt(u)
        w1 = 1 - su
        w2 = su * (1 - v)
        w3 = su * v
        p = w1 * a + w2 * b + w3 * c
        samples.append(p)
    return np.array(samples, dtype=float)


def sample_uniform_boundary(vertices: np.ndarray, K_edge: int):
    """
    Sample points uniformly along 2n-polygon boundary.
    
    Returns:
        boundary_points: (m*K_edge, 2) array of unique boundary points
        edge_point_ids: List[List[int]] - point IDs for each edge
        point_edge_ids: (m*K_edge,) array - edge assignment for each point
    """
    m = len(vertices)
    num_boundary_points = m * K_edge
    all_points = np.zeros((num_boundary_points, 2), dtype=float)
    for i in range(m):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % m]
        for j in range(K_edge):
            u = i * K_edge + j
            t = j / K_edge
            q = (1 - t) * p0 + t * p1
            all_points[u] = q
    edge_point_ids: List[List[int]] = []
    for i in range(m):
        ids = [(i * K_edge + j) % num_boundary_points for j in range(K_edge + 1)]
        edge_point_ids.append(ids)
    point_edge_ids = np.zeros(num_boundary_points, dtype=int)
    for i in range(m):
        for j in range(K_edge):
            u = i * K_edge + j
            point_edge_ids[u] = i
    return all_points, edge_point_ids, point_edge_ids


def voronoi_finite_polygons_2d(vor: Voronoi, radius: Optional[float] = None):
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


# =============================================================================
#  Node Reordering Functions
# =============================================================================

def reorder_nodes_bfs(nodes: np.ndarray, edges: List[Tuple[int, int]], start_node: int = 0) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """
    Reorder nodes using BFS traversal starting from a specified node.
    
    Args:
        nodes: Array of node coordinates (N, 2)
        edges: List of edge tuples (u, v)
        start_node: Starting node index for BFS (default: 0, first boundary point)
    
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
        # Sort neighbors for deterministic ordering
        neighbors = sorted(adj_list[current])
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                bfs_order.append(neighbor)
    
    # Handle disconnected components (if any)
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
        if new_u != new_v:  # Skip self-loops
            # Ensure u < v for consistency
            if new_u > new_v:
                new_u, new_v = new_v, new_u
            reordered_edges.append((new_u, new_v))
    
    # Remove duplicate edges
    reordered_edges = list(set(reordered_edges))
    
    return reordered_nodes, reordered_edges, old_to_new


# =============================================================================
#  Dataset Saving Functions
# =============================================================================

def save_dataset(nodes: np.ndarray, edges: List[Tuple[int, int]], args, dataset_name: str = "custom_topology", output_dir: str = "output"):
    """
    Save complete dataset files:
      - A_labeled.csv, A.npy (adjacency matrix)
      - nodes.csv (node information)
      - coords.csv, coords.npy (coordinates)
      - distance_matrix.npy (shortest paths)
      - graph_info.json (metadata)
    """
    print(f"\n[Dataset] Generating files for '{dataset_name}'...")
    num_nodes = len(nodes)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build adjacency matrix
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    data = [1] * len(rows)
    adj_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adj_dense = adj_sparse.toarray()
    
    # Save adjacency matrix
    np.save(f"{output_dir}/A_{dataset_name}.npy", adj_dense)
    df_adj = pd.DataFrame(adj_dense, index=range(num_nodes), columns=range(num_nodes))
    df_adj.to_csv(f"{output_dir}/A_{dataset_name}_labeled.csv")
    
    # Save coordinates
    np.save(f"{output_dir}/coords_{dataset_name}.npy", nodes)
    nodes_3d = np.column_stack([nodes, np.zeros(num_nodes)])
    df_coords = pd.DataFrame(nodes_3d, columns=['x', 'y', 'z'])
    df_coords['node_id'] = range(num_nodes)
    df_coords.set_index('node_id', inplace=True)
    df_coords.to_csv(f"{output_dir}/coords_{dataset_name}.csv")
    
    # Save node information
    degrees = np.array(adj_sparse.sum(axis=1)).flatten()
    df_nodes = pd.DataFrame({
        'node_id': range(num_nodes),
        'degree': degrees,
        'type': 'manifold_point'
    })
    df_nodes.set_index('node_id', inplace=True)
    df_nodes.to_csv(f"{output_dir}/nodes_{dataset_name}.csv")
    
    # Compute and save distance matrix
    print("[Dataset] Calculating shortest paths (this may take a moment)...")
    dist_matrix = shortest_path(csgraph=adj_sparse, directed=False, unweighted=True)
    np.save(f"{output_dir}/distance_matrix_{dataset_name}.npy", dist_matrix)
    
    # Save metadata
    info = {
        "dataset_name": dataset_name,
        "num_nodes": int(num_nodes),
        "num_edges": len(edges),
        "topology_rule": args.topology,
        "n_polygon": args.n,
        "k_edge": args.K_edge,
        "avg_degree": float(np.mean(degrees)),
        "euler_characteristic_est": int(num_nodes - len(edges) + len(edges)/3)
    }
    with open(f"{output_dir}/graph_info_{dataset_name}.json", "w") as f:
        json.dump(info, f, indent=4)
    
    print(f"[Dataset] Done! All files saved to './{output_dir}/'")
    print(f"  - A_{dataset_name}_labeled.csv")
    print(f"  - A_{dataset_name}.npy")
    print(f"  - nodes_{dataset_name}.csv")
    print(f"  - coords_{dataset_name}.csv")
    print(f"  - coords_{dataset_name}.npy")
    print(f"  - distance_matrix_{dataset_name}.npy")
    print(f"  - graph_info_{dataset_name}.json")


# =============================================================================
#  Point Distribution and Evolution
# =============================================================================

def filter_redundant_interior(layer0_points, interior_points, min_dist_threshold):
    """Remove interior points that are too close to boundary."""
    if len(interior_points) == 0:
        return interior_points
    tree = KDTree(layer0_points)
    dists, _ = tree.query(interior_points)
    keep_mask = dists > min_dist_threshold
    kept_interior = interior_points[keep_mask]
    removed_count = len(interior_points) - len(kept_interior)
    if removed_count > 0:
        print(f"[Filter] Removed {removed_count} interior points that were too close to boundary.")
    return kept_interior


def _get_domain(polygon: np.ndarray) -> ShapelyPolygon:
    """Get Shapely polygon domain."""
    domain = ShapelyPolygon(polygon)
    return domain if domain.is_valid else domain.buffer(0.0)


def _create_boundary_layer(polygon_shapely: ShapelyPolygon, num_points: int) -> Optional[np.ndarray]:
    """Create boundary layer points."""
    if hasattr(polygon_shapely, 'geoms'):
        polygon_shapely = max(polygon_shapely.geoms, key=lambda p: p.area)
    if not hasattr(polygon_shapely, 'exterior'):
        return None
    boundary = polygon_shapely.exterior
    total_length = boundary.length
    points = []
    for i in range(num_points):
        distance = (i / num_points) * total_length
        pt = boundary.interpolate(distance)
        points.append([pt.x, pt.y])
    return np.array(points)


def create_multi_layer_boundaries(polygon: np.ndarray, K_edge: int, num_layers: int, layer_interval: float) -> np.ndarray:
    """Create multiple boundary layers."""
    if num_layers <= 0:
        return np.zeros((0, 2))
    domain = _get_domain(polygon)
    layers = []
    points, _, _ = sample_uniform_boundary(polygon, K_edge)
    num_points_per_layer = len(points)
    layers.append(points)
    for layer in range(1, num_layers):
        offset_poly = domain.buffer(layer * layer_interval)
        layer_points = _create_boundary_layer(offset_poly, num_points_per_layer)
        if layer_points is not None:
            layers.append(layer_points)
    return np.vstack(layers) if layers else np.zeros((0, 2))


def initialize_points(polygon: np.ndarray, K_edge: int, N_interior: int, rng: np.random.Generator,
                      num_boundary_layers: int = 1, boundary_layer_interval: float = 0.1) -> Tuple[np.ndarray, int]:
    """Initialize points: boundary layers + interior points."""
    boundary_points = create_multi_layer_boundaries(polygon, K_edge, num_boundary_layers, boundary_layer_interval)
    interior_points = sample_interior_points(polygon, N_interior, rng=rng)
    return np.vstack([boundary_points, interior_points]), len(boundary_points)


def project_to_domain(point: np.ndarray, domain: ShapelyPolygon) -> np.ndarray:
    """Project point to domain boundary if outside."""
    pt = Point(point)
    if domain.contains(pt) or domain.touches(pt):
        return point
    nearest = domain.boundary.interpolate(domain.boundary.project(pt))
    return np.array([nearest.x, nearest.y])


def _compute_global_avg_distance(points: np.ndarray, k_neighbors: int) -> float:
    """Compute global average distance to k nearest neighbors."""
    if len(points) <= 1:
        return 1.0
    tree = KDTree(points)
    distances = []
    for i in range(len(points)):
        dists, _ = tree.query(points[i], k=min(k_neighbors + 1, len(points)))
        if len(dists) > 1:
            distances.extend(dists[1:])
    return np.mean(distances) if distances else 1.0


def compute_forces(points: np.ndarray, N_boundary: int, repulsive_strength: float = 0.5,
                   interaction_range: float = None, k_neighbors: int = 5) -> np.ndarray:
    """Compute repulsive forces for interior points."""
    n = len(points)
    if n <= 1:
        return np.zeros((n, 2))
    global_avg = _compute_global_avg_distance(points, k_neighbors)
    interaction_range = interaction_range or global_avg * 2.0
    tree = KDTree(points)
    forces = np.zeros((n, 2))
    interior_indices = np.arange(N_boundary, n)
    for i in interior_indices:
        neighbors = tree.query_ball_point(points[i], interaction_range)
        for j in neighbors:
            if i == j:
                continue
            vec = points[i] - points[j]
            dist = max(np.linalg.norm(vec), 1e-6)
            diff = dist - global_avg
            if diff < 0:
                force_mag = repulsive_strength * (-diff) / (dist ** 2)
                forces[i] += force_mag * vec / dist
    return forces


def force_step(points: np.ndarray, polygon: np.ndarray, N_boundary: int, step_size: float = 0.05,
               repulsive_strength: float = 0.5, interaction_range: float = None,
               max_force: float = 0.5, k_neighbors: int = 5) -> np.ndarray:
    """Apply one step of force-based evolution."""
    domain = _get_domain(polygon)
    forces = compute_forces(points, N_boundary, repulsive_strength, interaction_range, k_neighbors)
    interior_indices = np.arange(N_boundary, len(points))
    if len(interior_indices) > 0:
        force_mags = np.linalg.norm(forces[interior_indices], axis=1)
        mask = force_mags > max_force
        if np.any(mask):
            forces[interior_indices[mask]] *= (max_force / force_mags[mask, np.newaxis])
    new_points = points.copy()
    new_points[interior_indices] += step_size * forces[interior_indices]
    for i in interior_indices:
        new_points[i] = project_to_domain(new_points[i], domain)
    return new_points


def _compute_metrics(points: np.ndarray) -> Tuple[float, float]:
    """Compute mean and variance of nearest neighbor distances."""
    if len(points) <= 1:
        return 0.0, 0.0
    nn_dists = KDTree(points).query(points, k=2)[0][:, 1]
    return float(nn_dists.mean()), float(nn_dists.var())


def run_evolution(points: np.ndarray, polygon: np.ndarray, N_boundary: int, iters: int,
                 step_size: float = 0.05, repulsive_strength: float = 0.5,
                 interaction_range: float = None, adaptive_step: bool = True,
                 k_neighbors: int = 5, save_plots: bool = False, plot_interval: int = 1,
                 prefix: str = "iter") -> Tuple[np.ndarray, List[float], List[float], List[float]]:
    """Run repulsive force evolution for specified iterations."""
    x = points.copy()
    deltas, avg_distances, distance_vars = [], [], []
    interior_indices = np.arange(N_boundary, len(x))
    for k in range(iters):
        old_x = x.copy()
        current_step = step_size * (1.0 - k / iters) ** 2 if adaptive_step else step_size
        x = force_step(x, polygon, N_boundary, current_step, repulsive_strength,
                      interaction_range, k_neighbors=k_neighbors)
        if len(interior_indices) > 0:
            deltas.append(float(np.linalg.norm(x[interior_indices] - old_x[interior_indices], axis=1).mean()))
        else:
            deltas.append(0.0)
        avg_dist, dist_var = _compute_metrics(x)
        avg_distances.append(avg_dist)
        distance_vars.append(dist_var)
        if save_plots and (k == 0 or k == iters - 1 or (k + 1) % plot_interval == 0):
            plot_voronoi(x, polygon, f"{prefix}_{k:03d}.png", f"Iteration {k+1}/{iters}")
    return x, deltas, avg_distances, distance_vars


def compute_areas(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Compute Voronoi cell areas."""
    domain = _get_domain(polygon)
    vor = Voronoi(points)
    regions, vtx = voronoi_finite_polygons_2d(vor)
    areas = []
    for region in regions:
        if len(region) < 3:
            areas.append(0.0)
            continue
        coords = np.unique(vtx[region][np.isfinite(vtx[region]).all(axis=1)], axis=0)
        if len(coords) < 3:
            areas.append(0.0)
            continue
        cell = ShapelyPolygon(coords)
        if not cell.is_valid:
            cell = cell.buffer(0.0)
        clipped = cell.intersection(domain)
        areas.append(float(clipped.area) if not clipped.is_empty else 0.0)
    return np.array(areas)


def compute_nn_distances(points: np.ndarray) -> np.ndarray:
    """Compute nearest neighbor distances."""
    if len(points) <= 1:
        return np.zeros(0)
    return KDTree(points).query(points, k=2)[0][:, 1]


def evaluate_uniformity(points: np.ndarray) -> float:
    """Evaluate point distribution uniformity (coefficient of variation)."""
    nn_dists = compute_nn_distances(points)
    if len(nn_dists) == 0 or nn_dists.mean() == 0:
        return float('inf')
    return nn_dists.std() / nn_dists.mean()


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


def plot_voronoi(points: np.ndarray, polygon: np.ndarray, save_path: str, title: str = "Voronoi diagram") -> None:
    """Plot Voronoi diagram."""
    vor = Voronoi(points)
    regions, vtx = voronoi_finite_polygons_2d(vor)
    domain = _get_domain(polygon)
    fig, ax = plt.subplots(figsize=(6, 6))
    poly_closed = np.vstack([polygon, polygon[0]])
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], "k-", lw=1.5, label="domain")
    for region in regions:
        if len(region) == 0:
            continue
        coords = _sort_vertices(vtx[region])
        if coords is None or len(coords) < 2:
            continue
        in_domain = np.array([domain.contains(Point(pt)) if np.isfinite(pt).all() else False for pt in coords])
        if not np.any(in_domain):
            continue
        if np.all(in_domain):
            if len(coords) >= 3 and not np.allclose(coords[0], coords[-1], atol=1e-10):
                coords = np.vstack([coords, coords[0]])
            ax.plot(coords[:, 0], coords[:, 1], "b-", lw=0.5, alpha=0.6)
        else:
            cell = ShapelyPolygon(coords)
            if cell.is_valid:
                _plot_clipped_cell(ax, cell, domain)
    ax.scatter(points[:, 0], points[:, 1], s=12, c='blue', alpha=0.8, label="points")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
#  Topology Gluing Module
# =============================================================================

def parse_topology_string(rule_str: str, num_edges: int) -> List[Tuple[int, int, int]]:
    """
    Parse topology string using capital letter notation.
    
    Input format (user should input this directly):
    - Lowercase letters (a, b, c, ...) = normal edges
    - Uppercase letters (A, B, C, ...) = reversed edges
      (In mathematical notation: A = a^-1, B = b^-1, etc.)
    
    Example: "abAB" means edges a, b, a^-1, b^-1 (where A=a^-1, B=b^-1)
    
    Returns:
        List of (edge1_idx, edge2_idx, orientation) tuples
        orientation: -1 for standard/reverse, 1 for twist
    """
    # Extract all characters (both lowercase and uppercase)
    tokens = re.findall(r"([a-zA-Z])", rule_str)
    if len(tokens) != num_edges:
        raise ValueError(
            f"[Error] Topology rule defines {len(tokens)} edges, "
            f"but the polygon has {num_edges} edges.\nInput: {rule_str}"
        )
    gluing_rules = []
    seen = {}
    for i, char in enumerate(tokens):
        # Lowercase = normal edge, Uppercase = reversed edge
        is_inverse = char.isupper()
        char_lower = char.lower()
        
        if char_lower in seen:
            prev_i, prev_is_inv = seen[char_lower]
            # Different signs (a, A) -> -1 (Standard Reverse); same signs -> 1 (Twist)
            orientation = -1 if (is_inverse != prev_is_inv) else 1
            gluing_rules.append((prev_i, i, orientation))
            del seen[char_lower]
        else:
            seen[char_lower] = (i, is_inverse)
    return gluing_rules


def glue_boundary_edges(points: np.ndarray, polygon: np.ndarray, 
                        gluing_rules: List[Tuple[int, int, int]],
                        K_edge: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Glue boundary edges based on topology rules using Voronoi-based adjacency.
    
    Args:
        points: Point coordinates (N x 2)
        polygon: Polygon vertices
        gluing_rules: List of (edge1, edge2, orientation) tuples
        K_edge: Number of points per edge
        
    Returns:
        final_coords: Glued node coordinates
        final_edges: List of (u, v) edge tuples
    """
    num_poly_edges = len(polygon)
    num_points = len(points)
    N_boundary = num_poly_edges * K_edge
    
    if num_points < N_boundary:
        raise ValueError(f"Points count ({num_points}) is less than required boundary points ({N_boundary})!")
    
    # Build base topology using Voronoi
    vor = Voronoi(points)
    raw_edges = vor.ridge_points
    
    # Initialize union-find
    parent = list(range(num_points))
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i
    
    # Index arithmetic for boundary points
    def get_boundary_index(edge_idx, step):
        return (edge_idx * K_edge + step) % N_boundary
    
    print(f"[Gluing] Applying rules with K_edge={K_edge}, Boundary_points={N_boundary} (Method: Voronoi)...")
    
    # Apply gluing rules
    for e1, e2, orient in gluing_rules:
        for step in range(K_edge + 1):
            u = get_boundary_index(e1, step)
            if orient == 1:  # Twist (a -> a)
                v = get_boundary_index(e2, step)
            else:  # Standard (a -> a^-1)
                v = get_boundary_index(e2, K_edge - step)
            union(u, v)
    
    # Rebuild nodes
    old_to_new = np.zeros(num_points, dtype=int)
    groups = {}
    for i in range(num_points):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    sorted_roots = sorted(groups.keys())
    final_coords = []
    for new_id, root in enumerate(sorted_roots):
        indices = groups[root]
        old_to_new[indices] = new_id
        coords_group = points[indices]
        mean_pt = np.mean(coords_group, axis=0)
        final_coords.append(mean_pt)
    
    final_coords = np.array(final_coords)
    
    # Rebuild edges
    final_edges_set = set()
    mapped_edges = old_to_new[raw_edges]
    for (n1, n2) in mapped_edges:
        if n1 == n2:
            continue
        if n1 > n2:
            n1, n2 = n2, n1
        final_edges_set.add((n1, n2))
    
    print(f"[Gluing] Collapsed {num_points} -> {len(final_coords)} nodes.")
    print(f"[Gluing] Generated {len(final_edges_set)} edges (Voronoi).")
    
    return final_coords, list(final_edges_set)


# =============================================================================
#  Main Execution
# =============================================================================

def run_diagnostics(n: int, iters: int, seed: int, K_edge: int, N_interior: int, plot_interval: int = 1,
                   step_size: float = 0.05, repulsive_strength: float = 0.5,
                   interaction_range: float = None, adaptive_step: bool = True,
                   k_neighbors: int = 5, num_boundary_layers: int = 1, boundary_layer_interval: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Run point distribution evolution with diagnostics."""
    rng = np.random.default_rng(seed)
    polygon = regular_2n_polygon(n)
    all_points, N_boundary = initialize_points(polygon, K_edge, N_interior, rng, num_boundary_layers, boundary_layer_interval)
    
    print(f"[INFO] K_edge={K_edge}, N_boundary={N_boundary}, N_interior={N_interior}, iters={iters}")
    
    x_final, deltas, avg_distances, distance_vars = run_evolution(
        all_points, polygon, N_boundary, iters, step_size, repulsive_strength,
        interaction_range, adaptive_step, k_neighbors, save_plots=False, plot_interval=plot_interval)
    
    areas = compute_areas(x_final, polygon)
    nn_dists = compute_nn_distances(x_final)
    
    print(f"[FINAL] Uniformity (CV): {evaluate_uniformity(x_final):.6f}")
    print(f"[FINAL] NN distance: mean={nn_dists.mean():.4f}, std={nn_dists.std():.4f}")
    
    return x_final, polygon


def estimate_num_interior_points(N: int, segments_per_edge: int) -> int:
    """Estimate number of interior points based on boundary density."""
    vertices = regular_2n_polygon(n=N, radius=1.0)
    poly = Polygon(vertices)
    num_edges = len(vertices)
    perimeter = poly.length
    area = poly.area
    N_boundary = num_edges * segments_per_edge
    d = perimeter / N_boundary
    rho = 1.0 / ((math.sqrt(3)/2) * (d ** 2))
    n_allpoints = int(round(area * rho))
    n_allpoints = max(n_allpoints, N_boundary + 1)
    
    if segments_per_edge <= 13:
        n_interior = n_allpoints - int(round((1 + (num_edges-6)/num_edges * (math.log(n_allpoints/N_boundary/(segments_per_edge/10)) - (1+num_edges/6)/abs(num_edges-6))) * N_boundary))
    elif segments_per_edge <= 25:
        n_interior = n_allpoints - int(round((1 + abs(num_edges-6)/num_edges * (math.log(n_allpoints/N_boundary/(segments_per_edge/10)) - (1+num_edges/6)/abs(num_edges-6))) * N_boundary))
    else:
        n_interior = n_allpoints - int(round((1 + abs(num_edges-6)/2 * (math.log(n_allpoints/N_boundary/(segments_per_edge/10)) - (1+num_edges/6)/abs(num_edges-6))) * N_boundary))
    
    n_allpoints = N_boundary + n_interior
    n_interior = max(n_interior, 1)
    
    print(f"[INFO_of_vertex] D_boundary={d:.4f}, N_boundary={N_boundary}, N_interior={n_interior}, Total_points={n_allpoints}")
    return n_interior, d, max(round(math.sqrt(n_interior) / 2), 8)


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Repulsive force point distribution on 2n-polygon")
    p.add_argument("--n", type=int, default=4, help="n in 2n-polygon")
    p.add_argument("--iters", type=int, default=200, help="Iterations")
    p.add_argument("--plot_interval", type=int, default=10, help="Plot interval")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--K_edge", type=int, default=13, help="Points per edge")
    p.add_argument("--N_interior", type=int, default=None, help="Interior points")
    p.add_argument("--step_size", type=float, default=0.1, help="Step size")
    p.add_argument("--repulsive_strength", type=float, default=0.5, help="Repulsive force strength")
    p.add_argument("--interaction_range", type=float, default=None, help="Interaction range")
    p.add_argument("--k_neighbors", type=int, default=25, help="K-neighbors for force computation")
    p.add_argument("--num_boundary_layers", type=int, default=2, help="Number of boundary layers")
    p.add_argument("--boundary_layer_interval", type=float, default=0.05, help="Distance between boundary layers")
    p.add_argument("--no_adaptive_step", action="store_true", help="Disable adaptive step")
    p.add_argument("--topology", type=str, default=None, 
                    help="Gluing rule string: use capital letters for reversed edges (A=a^-1, B=b^-1, etc.). Example: 'abAB' for torus")
    p.add_argument("--prefix", type=str, default=None,
                    help="Topology prefix for dataset naming (e.g., 'torus', 'klein', 'sphere'). If not provided, will be auto-detected from topology rule.")
    p.add_argument("--output_dir", type=str, default="output", help="Output directory")
    return p.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    if args.N_interior is None:
        args.N_interior, args.boundary_layer_interval, args.k_neighbors = estimate_num_interior_points(
            N=args.n, segments_per_edge=args.K_edge
        )
    
    # Run point distribution evolution
    x_final, polygon = run_diagnostics(
        args.n, args.iters, args.seed, args.K_edge, args.N_interior,
        args.plot_interval, args.step_size, args.repulsive_strength,
        args.interaction_range, not args.no_adaptive_step, args.k_neighbors,
        args.num_boundary_layers, args.boundary_layer_interval
    )
    
    num_poly_edges = 2 * args.n
    num_layer0 = num_poly_edges * args.K_edge
    layer0_points = x_final[0:num_layer0]
    interior_points = x_final[-args.N_interior:]
    
    # Filter redundant interior points
    threshold = args.boundary_layer_interval * 0.25
    clean_interior = filter_redundant_interior(layer0_points, interior_points, threshold)
    x_final = np.vstack([layer0_points, clean_interior])
    print(f"[Filter] After filtering, total points: {len(x_final)}")
    
    # Apply topology gluing if specified
    if args.topology:
        print(f"\n[Topology] Processing Gluing Rule: {args.topology}")
        try:
            # Count edges in topology rule
            topology_edges = len(re.findall(r"([a-zA-Z])", args.topology))
            num_poly_edges = 2 * args.n
            
            # Validate that polygon has correct number of edges
            if topology_edges != num_poly_edges:
                required_n = topology_edges // 2
                if topology_edges % 2 != 0:
                    raise ValueError(
                        f"[Error] Topology rule '{args.topology}' has {topology_edges} edges, "
                        f"which is not even. A 2n-polygon must have an even number of edges.\n"
                        f"  Please check your topology rule."
                    )
                raise ValueError(
                    f"[Error] Topology rule '{args.topology}' requires {topology_edges} edges, "
                    f"but polygon with n={args.n} has {num_poly_edges} edges (2n={2*args.n}).\n"
                    f"  Solution: Set n={required_n} to get {topology_edges} edges (2n={2*required_n}).\n"
                    f"  Example: For double torus (abABcdCD), use n=4."
                )
            
            gluing_rules = parse_topology_string(args.topology, num_poly_edges)
            nodes_raw, edges_raw = glue_boundary_edges(x_final, polygon, gluing_rules, args.K_edge)
            final_nodes, final_edges = nodes_raw, edges_raw
            
            print(f"[Topology] Completed!")
            print(f"  - Original Nodes: {len(x_final)}")
            print(f"  - Glued Nodes: {len(final_nodes)}")
            print(f"  - Edges: {len(final_edges)}")
            
            # Reorder nodes using BFS starting from first boundary point (node 0)
            print(f"\n[Reordering] Applying BFS traversal to assign node IDs...")
            final_nodes, final_edges, old_to_new = reorder_nodes_bfs(final_nodes, final_edges, start_node=0)
            print(f"  - Reordered {len(final_nodes)} nodes using BFS from node 0")
            print(f"  - Remapped {len(final_edges)} edges")
            
            # Construct dataset name with prefix
            # Topology rule should already be in capital letter form (e.g., abAB instead of aba^-1b^-1)
            if args.prefix:
                prefix = args.prefix
            else:
                # Auto-detect prefix based on topology rule (matching shell script logic)
                topology_rule = args.topology
                if topology_rule in ["abAB", "abABcdCD", "abABcdCDefEF"]:
                    prefix = "torus"
                elif topology_rule in ["abAb", "abaB"]:
                    prefix = "klein"
                elif topology_rule in ["abBA", "aAbB"]:
                    prefix = "sphere"
                elif topology_rule == "abab":
                    prefix = "projective"
                else:
                    # Default: use first 8 chars of topology rule (lowercase)
                    prefix = topology_rule.lower()[:8]
            
            dataset_name = f"{prefix}_{args.topology}_n{args.n}_k{args.K_edge}_iter{args.iters}"
            
            save_dataset(final_nodes, final_edges, args, dataset_name=dataset_name, output_dir=args.output_dir)
        except Exception as e:
            print(f"[Topology Error] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
