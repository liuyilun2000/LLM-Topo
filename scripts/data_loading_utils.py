"""
Data loading utility functions for PCA, UMAP, fuzzy distance matrices, trajectories, and token representations
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_pca_data(pca_dir: str, key: str, use_downsampled: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load PCA data (regular or downsampled).
    
    Args:
        pca_dir: Directory containing PCA results
        key: Representation name
        use_downsampled: If True, load downsampled PCA data
    
    Returns:
        pca_reduced: PCA-reduced data array
        selected_indices: Indices of selected points (if downsampled) or None
    """
    pca_dir = Path(pca_dir)
    
    if use_downsampled:
        npz_file = pca_dir / f'{key}_pca_downsampled.npz'
        if not npz_file.exists():
            raise FileNotFoundError(f"Downsampled PCA results not found for {key} in {pca_dir}")
    else:
        npz_file = pca_dir / f'{key}_pca.npz'
        if not npz_file.exists():
            raise FileNotFoundError(f"PCA results not found for {key} in {pca_dir}")
    
    data = np.load(npz_file)
    pca_reduced = data['pca_reduced']
    selected_indices = data.get('selected_indices', None)
    
    return pca_reduced, selected_indices


def load_fuzzy_distance_matrix(fuzzy_dir: str, key: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load fuzzy neighborhood distance matrix.
    
    Args:
        fuzzy_dir: Directory containing fuzzy distance matrices
        key: Representation name
    
    Returns:
        distance_matrix: Fuzzy distance matrix
        pca_data: Original PCA data (optional, for reference)
    """
    fuzzy_dir = Path(fuzzy_dir)
    npz_file = fuzzy_dir / f'{key}_fuzzy_dist.npz'
    
    if not npz_file.exists():
        raise FileNotFoundError(f"Fuzzy distance matrix not found for {key} in {fuzzy_dir}")
    
    data = np.load(npz_file)
    distance_matrix = data['distance_matrix']
    pca_data = data.get('pca_reduced', None)
    
    return distance_matrix, pca_data


def load_data_representation(data_dir: str, key: str, data_type: str = 'auto') -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Load data representation (PCA, UMAP, or downsampled PCA) with auto-detection.
    
    Priority order for auto-detection:
    1. UMAP results ({key}_umap_{n}d.npz)
    2. Downsampled PCA ({key}_pca_downsampled.npz)
    3. PCA ({key}_pca.npz)
    
    Args:
        data_dir: Directory containing results
        key: Representation name
        data_type: 'auto', 'pca', 'umap', or 'downsampled'
    
    Returns:
        data: Embeddings array or None if not found
        selected_indices: Indices of selected points (if downsampled) or None
        source_type: 'pca', 'umap', 'downsampled', or None
    """
    data_dir = Path(data_dir)
    
    if data_type == 'downsampled':
        npz_file = data_dir / f'{key}_pca_downsampled.npz'
        if not npz_file.exists():
            raise FileNotFoundError(f"Downsampled PCA results not found for {key}")
        data = np.load(npz_file)
        return data['pca_reduced'], data.get('selected_indices', None), 'downsampled'
    
    # Try UMAP results
    if data_type in ['auto', 'umap']:
        umap_files = list(data_dir.glob(f'{key}_umap_*d.npz'))
        if umap_files:
            npz_file = sorted(umap_files)[0]
            data = np.load(npz_file)
            if 'umap_reduced' in data:
                return data['umap_reduced'], None, 'umap'
    
    # Try PCA
    if data_type in ['auto', 'pca']:
        npz_file = data_dir / f'{key}_pca.npz'
        if npz_file.exists():
            data = np.load(npz_file)
            if 'pca_reduced' in data:
                return data['pca_reduced'], data.get('selected_indices', None), 'pca'
    
    return None, None, None


def load_representations(representation_dir: str) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    """
    Load token representation files.
    
    Args:
        representation_dir: Path to directory containing token_representations.npz and token_metadata.json
    
    Returns:
        representations: Dict mapping representation names to numpy arrays
        metadata: List of token metadata dictionaries
    """
    representation_dir = Path(representation_dir)
    representation_file = representation_dir / 'token_representations.npz'
    metadata_file = representation_dir / 'token_metadata.json'
    
    if not representation_file.exists():
        raise FileNotFoundError(f"Representation file not found: {representation_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    print(f"Loading representations from {representation_file}")
    data = np.load(representation_file)
    
    print(f"Loading metadata from {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    representations = {key: data[key] for key in data.keys()}
    
    print(f"\nLoaded {len(representations)} representations:")
    for key, arr in representations.items():
        print(f"  {key}: {arr.shape}")
    
    return representations, metadata


def load_summary(representation_dir: str) -> Optional[Dict]:
    """
    Load extraction summary information.
    
    Args:
        representation_dir: Path to directory containing extraction_summary.json
    
    Returns:
        summary: Dict with model and extraction info, or None if not found
    """
    summary_file = Path(representation_dir) / 'extraction_summary.json'
    if not summary_file.exists():
        return None
    
    with open(summary_file, 'r') as f:
        return json.load(f)


def load_vocab_size(representation_dir: str) -> Optional[int]:
    """
    Load vocab size from extraction summary.
    
    Args:
        representation_dir: Path to directory containing extraction_summary.json
    
    Returns:
        vocab_size: Integer vocabulary size, or None if not found
    """
    summary = load_summary(representation_dir)
    return summary.get('vocab_size') if summary else None


def load_trajectory(walks_csv: str, walk_id: Optional[int] = None, trajectory_idx: Optional[int] = None) -> Tuple[List[int], int]:
    """
    Load a trajectory from walks CSV file.
    
    Args:
        walks_csv: Path to walks CSV file
        walk_id: Specific walk_id to load (if None, uses trajectory_idx)
        trajectory_idx: Index of trajectory in CSV (0-based, if walk_id is None)
    
    Returns:
        trajectory: List of node IDs (as integers)
        walk_id: The walk_id used
    """
    df = pd.read_csv(walks_csv)
    
    if walk_id is not None:
        row = df[df['walk_id'] == walk_id]
        if row.empty:
            raise ValueError(f"Walk ID {walk_id} not found in {walks_csv}")
    elif trajectory_idx is not None:
        if trajectory_idx >= len(df):
            raise ValueError(f"Trajectory index {trajectory_idx} out of range (max: {len(df)-1})")
        row = df.iloc[[trajectory_idx]]
        walk_id = row.iloc[0]['walk_id']
    else:
        raise ValueError("Either walk_id or trajectory_idx must be provided")
    
    sequence_labels = row.iloc[0]['sequence_labels']
    trajectory = [int(x) for x in sequence_labels.split()]
    
    return trajectory, walk_id
