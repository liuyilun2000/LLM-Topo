"""
Graph utility functions for loading and saving graph representations
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
from scipy.sparse.csgraph import shortest_path


def load_graph(graph_dir: str, dataset_name: str) -> Tuple[np.ndarray, List[str], pd.DataFrame, Optional[np.ndarray]]:
    """
    Load graph representation from saved files.
    
    Args:
        graph_dir: Directory containing graph files
        dataset_name: Name of the dataset (e.g., "torus_abAB_N800_iter200")
        
    Returns:
        A: Adjacency matrix (N x N)
        labels: Node labels (list of strings)
        nodes_df: Node information DataFrame
        coords: Coordinates (N x 3) or None
    """
    # Load adjacency matrix
    A_labeled_path = os.path.join(graph_dir, f"A_{dataset_name}_labeled.csv")
    if not os.path.exists(A_labeled_path):
        raise FileNotFoundError(
            f"Graph file not found: {A_labeled_path}\n"
            f"Please run the generation script first with matching parameters."
        )
    
    print(f"Loading adjacency matrix from: {A_labeled_path}")
    dfA = pd.read_csv(A_labeled_path, index_col=0)
    labels = dfA.index.astype(str).tolist()
    A = dfA.values.astype(np.int8)
    
    # Load nodes information
    nodes_path = os.path.join(graph_dir, f"nodes_{dataset_name}.csv")
    if os.path.exists(nodes_path):
        nodes_df = pd.read_csv(nodes_path, index_col=0)
    else:
        nodes_df = pd.DataFrame({"node_id": labels})
    
    # Load coordinates (optional)
    coords_path = os.path.join(graph_dir, f"coords_{dataset_name}.csv")
    coords = None
    if os.path.exists(coords_path):
        try:
            df_coords = pd.read_csv(coords_path, index_col=0)
            if set(['x', 'y', 'z']).issubset(df_coords.columns):
                coords = df_coords[['x', 'y', 'z']].values
            else:
                print(f"[Warn] Coords file found but columns mismatch. Expected x,y,z.")
        except Exception as e:
            print(f"[Warn] Failed to load coords: {e}")
    
    return A, labels, nodes_df, coords


def load_adjacency_matrix(graph_dir: str, dataset_name: str) -> Optional[np.ndarray]:
    """Load adjacency matrix from graph directory"""
    graph_dir = Path(graph_dir)
    adjacency_file = graph_dir / f'A_{dataset_name}.npy'
    
    if not adjacency_file.exists():
        return None
    
    adjacency = np.load(adjacency_file)
    # Ensure binary (0/1)
    adjacency = (adjacency > 0).astype(int)
    return adjacency


def load_distance_matrix(graph_dir: str, dataset_name: str) -> Optional[np.ndarray]:
    """Load distance matrix from graph directory"""
    graph_dir = Path(graph_dir)
    distance_file = graph_dir / f'distance_matrix_{dataset_name}.npy'
    
    if not distance_file.exists():
        return None
    
    return np.load(distance_file)


def load_dataset_matrix(graph_dir: str, dataset_name: str, matrix_type: str = 'auto') -> Tuple[np.ndarray, str]:
    """
    Load neighboring matrix from graph directory (distance or adjacency).
    
    Args:
        graph_dir: Path to graph directory
        dataset_name: Dataset name
        matrix_type: 'auto', 'distance', or 'adjacency'
    
    Returns:
        distance_matrix: Distance matrix for UMAP (always returns distance matrix)
        matrix_type_used: Which matrix type was actually used
    """
    graph_dir = Path(graph_dir)
    distance_file = graph_dir / f'distance_matrix_{dataset_name}.npy'
    adjacency_file = graph_dir / f'A_{dataset_name}.npy'
    
    if matrix_type == 'auto':
        if distance_file.exists():
            matrix = np.load(distance_file)
            print(f"  Loaded distance matrix from: {distance_file.name}")
            print(f"  Shape: {matrix.shape}")
            return matrix, 'distance'
        elif adjacency_file.exists():
            adjacency = np.load(adjacency_file)
            print(f"  Loaded adjacency matrix from: {adjacency_file.name}")
            print(f"  Shape: {adjacency.shape}")
            # Convert adjacency to distance matrix
            print(f"  Converting adjacency matrix to distance matrix...")
            distance_matrix = shortest_path(
                csgraph=adjacency,
                directed=False,
                unweighted=False,
                method='auto'
            )
            # Replace infinities with large finite value
            distance_matrix[np.isinf(distance_matrix)] = np.max(distance_matrix[np.isfinite(distance_matrix)]) * 2
            print(f"  Distance matrix shape: {distance_matrix.shape}")
            return distance_matrix, 'adjacency'
        else:
            raise FileNotFoundError(
                f"Neither distance matrix nor adjacency matrix found for {dataset_name} in {graph_dir}\n"
                f"  Expected: {distance_file.name} or {adjacency_file.name}"
            )
    elif matrix_type == 'distance':
        if not distance_file.exists():
            raise FileNotFoundError(f"Distance matrix not found: {distance_file}")
        matrix = np.load(distance_file)
        print(f"  Loaded distance matrix from: {distance_file.name}")
        print(f"  Shape: {matrix.shape}")
        return matrix, 'distance'
    elif matrix_type == 'adjacency':
        if not adjacency_file.exists():
            raise FileNotFoundError(f"Adjacency matrix not found: {adjacency_file}")
        adjacency = np.load(adjacency_file)
        print(f"  Loaded adjacency matrix from: {adjacency_file.name}")
        print(f"  Shape: {adjacency.shape}")
        # Convert adjacency to distance matrix
        print(f"  Converting adjacency matrix to distance matrix...")
        distance_matrix = shortest_path(
            csgraph=adjacency,
            directed=False,
            unweighted=False,
            method='auto'
        )
        # Replace infinities with large finite value
        distance_matrix[np.isinf(distance_matrix)] = np.max(distance_matrix[np.isfinite(distance_matrix)]) * 2
        print(f"  Distance matrix shape: {distance_matrix.shape}")
        return distance_matrix, 'adjacency'
    else:
        raise ValueError(f"Invalid matrix_type: {matrix_type} (must be 'auto', 'distance', or 'adjacency')")


def build_neighbors_from_A(A: np.ndarray) -> List[np.ndarray]:
    """Build neighbor list from adjacency matrix"""
    return [np.where(A[i] == 1)[0].astype(np.int64) for i in range(A.shape[0])]
