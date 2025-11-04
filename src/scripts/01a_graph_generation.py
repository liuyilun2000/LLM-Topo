#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_graph.py - Generate graph representation for topology analysis

This script generates the graph structure (nodes, edges, adjacency matrix,
coordinates, and distance matrix) for a given topology and saves it to files
for later use in random walk generation.

Output files:
  - A_{topology}_{H}x{W}_labeled.csv : Adjacency matrix with node labels
  - A_{topology}_{H}x{W}.npy : Adjacency matrix as numpy array
  - nodes_{topology}_{H}x{W}.csv : Node information (node_id, rowcol_index, i, j, layer)
  - coords_{topology}_{H}x{W}.csv : 3D coordinates for visualization
  - coords_{topology}_{H}x{W}.npy : Coordinates as numpy array
  - distance_matrix_{topology}_{H}x{W}.npy : Shortest path distance matrix
  - graph_info_{topology}_{H}x{W}.json : Graph metadata (H, W, topology, neigh, etc.)
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix


def parse_topos(arg: str):
    """Parse topology string: single topology, comma-separated, or 'all'"""
    all_topos = ["plane", "cylinder_x", "cylinder_y", "mobius_x", "mobius_y",
                 "torus", "klein_x", "klein_y", "proj_plane", "sphere_two",
                 "hemisphere_n", "hemisphere_s", "sphere"]
    if arg.strip().lower() == "all":
        return all_topos
    return [t.strip() for t in arg.split(",") if t.strip()]


def compute_distance_matrix(A: np.ndarray, method: str = "shortest_path"):
    """
    Compute distance matrix from adjacency matrix.
    
    Args:
        A: Adjacency matrix (N x N, binary)
        method: 'shortest_path' or 'euclidean'
        
    Returns:
        Distance matrix (N x N)
    """
    if method == "shortest_path":
        # Use shortest path distances
        # Convert to sparse matrix for efficiency
        graph = csr_matrix(A.astype(float))
        # Compute shortest paths (unweighted, so edge weight = 1)
        dist_matrix = shortest_path(
            graph, 
            method='auto', 
            directed=False,
            unweighted=True
        )
        # Replace infinities (unreachable nodes) with a large value
        dist_matrix[np.isinf(dist_matrix)] = np.nan
        return dist_matrix
    else:
        raise ValueError(f"Unknown method: {method}")


def generate_graph(H: int, W: int, topo: str, neigh: int, output_dir: str):
    """
    Generate graph representation for a given topology.
    
    Args:
        H: Grid height
        W: Grid width
        topo: Topology name
        neigh: Neighborhood type (4 or 8)
        output_dir: Output directory
    """
    try:
        import quotient_space_topology as t
    except Exception as e:
        raise SystemExit(f"Failed to import quotient_space_topology: {e}")
    
    print(f"[*] Generating graph: {topo} (H={H}, W={W}, neigh={neigh})")
    
    # Generate adjacency matrix and nodes
    A, nodes, mapdf = t.build_adj(H, W, topo, neigh=neigh, undirected=True)
    
    # Sort nodes by rowcol_index to ensure consistent ordering
    mapdf_sorted = mapdf.sort_values("rowcol_index").reset_index(drop=True)
    labels = mapdf_sorted["node_id"].astype(str).tolist()
    
    # Generate coordinates
    X = t.coords_for_topo(H, W, topo)
    
    N = A.shape[0]
    print(f"    Nodes: {N}, Edges: {A.sum() // 2}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Base filename
    base = f"{topo}_{H}x{W}"
    
    # Save adjacency matrix (labeled)
    A_labeled_path = os.path.join(output_dir, f"A_{base}_labeled.csv")
    dfA_labeled = pd.DataFrame(A, index=labels, columns=labels)
    dfA_labeled.to_csv(A_labeled_path, encoding="utf-8")
    print(f"    Saved: {A_labeled_path}")
    
    # Save adjacency matrix (numpy)
    A_npy_path = os.path.join(output_dir, f"A_{base}.npy")
    np.save(A_npy_path, A)
    print(f"    Saved: {A_npy_path}")
    
    # Save nodes information
    nodes_path = os.path.join(output_dir, f"nodes_{base}.csv")
    mapdf_sorted.to_csv(nodes_path, index=False)
    print(f"    Saved: {nodes_path}")
    
    # Save coordinates (CSV)
    coords_csv_path = os.path.join(output_dir, f"coords_{base}.csv")
    np.savetxt(coords_csv_path, X, delimiter=",")
    print(f"    Saved: {coords_csv_path}")
    
    # Save coordinates (numpy)
    coords_npy_path = os.path.join(output_dir, f"coords_{base}.npy")
    np.save(coords_npy_path, X)
    print(f"    Saved: {coords_npy_path}")
    
    # Compute and save distance matrix (shortest path)
    print(f"    Computing distance matrix...")
    try:
        dist_matrix = compute_distance_matrix(A, method="shortest_path")
        dist_npy_path = os.path.join(output_dir, f"distance_matrix_{base}.npy")
        np.save(dist_npy_path, dist_matrix)
        print(f"    Saved: {dist_npy_path}")
    except Exception as e:
        print(f"    Warning: Failed to compute distance matrix: {e}")
        dist_matrix = None
    
    # Save graph metadata
    graph_info = {
        "topology": topo,
        "H": H,
        "W": W,
        "neigh": neigh,
        "num_nodes": int(N),
        "num_edges": int(A.sum() // 2),
        "avg_degree": float((A.sum(axis=0).mean())),
        "files": {
            "adjacency_matrix_labeled": f"A_{base}_labeled.csv",
            "adjacency_matrix_npy": f"A_{base}.npy",
            "nodes": f"nodes_{base}.csv",
            "coords_csv": f"coords_{base}.csv",
            "coords_npy": f"coords_{base}.npy",
            "distance_matrix": f"distance_matrix_{base}.npy" if dist_matrix is not None else None
        }
    }
    
    info_path = os.path.join(output_dir, f"graph_info_{base}.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(graph_info, f, indent=2)
    print(f"    Saved: {info_path}")
    
    print(f"    ✅ Graph generation complete for {topo}")
    return graph_info


def main():
    ap = argparse.ArgumentParser(
        description="Generate graph representation (adjacency matrix, nodes, coordinates, distance matrix)"
    )
    
    ap.add_argument("--H", type=int, required=True, help="Grid height")
    ap.add_argument("--W", type=int, required=True, help="Grid width")
    ap.add_argument("--topology", type=str, required=True,
                    help="Topology name, comma-separated list, or 'all'")
    ap.add_argument("--neigh", type=int, default=4, choices=[4, 8], 
                    help="Neighborhood type (4 or 8)")
    ap.add_argument("--output_dir", type=str, default="./data/graphs",
                    help="Output directory for graph files")
    
    args = ap.parse_args()
    
    topo_list = parse_topos(args.topology)
    
    print(f"Generating graphs for {len(topo_list)} topologie(s): {', '.join(topo_list)}")
    print(f"Output directory: {args.output_dir}\n")
    
    for topo in topo_list:
        try:
            generate_graph(args.H, args.W, topo, args.neigh, args.output_dir)
        except Exception as e:
            print(f"    ❌ Error generating {topo}: {e}")
            continue
        print()
    
    print(f"✅ All graphs generated. See: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
