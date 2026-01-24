"""
Visualize UMAP embedding from fuzzy neighborhood distance matrix with trajectory overlay
Creates a 3D plot with point cloud and trajectory colored by time
"""
import argparse
import json
import numpy as np
from pathlib import Path

from data_loading_utils import load_fuzzy_distance_matrix, load_trajectory
from visualization_utils import apply_umap, create_interactive_3d_html
from graph_utils import load_adjacency_matrix


def main():
    parser = argparse.ArgumentParser(
        description='Visualize UMAP embedding from fuzzy neighborhood with trajectory overlay'
    )
    parser.add_argument('--fuzzy_dir', type=str, required=True,
                      help='Directory with fuzzy neighborhood distance matrices')
    parser.add_argument('--key', type=str, required=True,
                      help='Representation key (e.g., layer_1_after_block)')
    parser.add_argument('--walks_csv', type=str, required=True,
                      help='Path to walks CSV file (e.g., sequences/walks_*.csv)')
    parser.add_argument('--walk_id', type=int, default=None,
                      help='Walk ID to use (if not specified, uses --trajectory_idx)')
    parser.add_argument('--trajectory_idx', type=int, default=0,
                      help='Index of trajectory in CSV (0-based, default: 0)')
    parser.add_argument('--output_dir', type=str, default='./umap_plot',
                      help='Output directory for plots')
    parser.add_argument('--umap_n_components', type=int, default=3,
                      help='UMAP target dimensions (default: 3)')
    parser.add_argument('--umap_min_dist', type=float, default=0.2,
                      help='UMAP min_dist parameter')
    parser.add_argument('--umap_n_neighbors', type=int, default=20,
                      help='UMAP n_neighbors parameter')
    parser.add_argument('--umap_random_state', type=int, default=None,
                      help='Random seed for UMAP (default: None, for reproducibility set to integer)')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Maximum number of points to plot in trajectory (default: 128)')
    parser.add_argument('--graph_dir', type=str, default=None,
                      help='Directory with graph files (for adjacency matrix visualization). Optional.')
    
    args = parser.parse_args()
    
    if args.umap_n_components != 3:
        print("Warning: This script is designed for 3D visualization. Using 3D anyway.")
        args.umap_n_components = 3
    
    print("="*60)
    print("UMAP Visualization with Trajectory")
    print("="*60)
    
    # Load trajectory first (needed for 2D plot, doesn't need UMAP)
    print(f"\nLoading trajectory from: {args.walks_csv}")
    trajectory, walk_id = load_trajectory(
        args.walks_csv,
        walk_id=args.walk_id,
        trajectory_idx=args.trajectory_idx
    )
    print(f"  Walk ID: {walk_id}")
    print(f"  Trajectory length: {len(trajectory)} nodes")
    
    # Limit trajectory to max_length for 2D plot
    limited_trajectory = trajectory[:args.max_length] if len(trajectory) > args.max_length else trajectory
    trajectory_positions = np.arange(len(limited_trajectory))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_base = f"{args.key}_walk_{walk_id}_umap_3d"
    
    # Load adjacency matrix for neighborhood edges (optional)
    adjacency_matrix = None
    if args.graph_dir:
        try:
            # Extract dataset name from walks CSV path or use a default
            graph_dir = Path(args.graph_dir)
            # Try to find dataset name from nodes files
            nodes_files = list(graph_dir.glob('nodes_*.csv'))
            if nodes_files:
                dataset_name = nodes_files[0].stem.replace('nodes_', '')
                adjacency_matrix = load_adjacency_matrix(str(graph_dir), dataset_name)
                if adjacency_matrix is not None:
                    print(f"  Loaded adjacency matrix: shape {adjacency_matrix.shape}")
        except Exception as e:
            print(f"  Warning: Could not load adjacency matrix: {e}")
    
    # Load fuzzy distance matrix for UMAP
    print(f"\nLoading fuzzy distance matrix for key: {args.key}")
    distance_matrix, pca_data = load_fuzzy_distance_matrix(args.fuzzy_dir, args.key)
    print(f"  Distance matrix shape: {distance_matrix.shape}")
    
    # Apply UMAP
    print(f"\nApplying UMAP...")
    umap_data, umap_info = apply_umap(
        distance_matrix=distance_matrix,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        n_neighbors=args.umap_n_neighbors,
        metric='precomputed',
        random_state=args.umap_random_state
    )
    
    # Map trajectory node IDs to indices in UMAP data
    # Assuming node IDs are 0-indexed or can be mapped directly
    # If node IDs don't match indices, we need to handle mapping
    try:
        trajectory_indices = np.array(limited_trajectory, dtype=int)
        # Check if indices are valid
        if trajectory_indices.max() >= len(umap_data):
            print(f"  Warning: Some trajectory node IDs exceed UMAP data size")
            print(f"  Max node ID: {trajectory_indices.max()}, UMAP size: {len(umap_data)}")
            # Filter valid indices
            valid_mask = trajectory_indices < len(umap_data)
            trajectory_indices = trajectory_indices[valid_mask]
            trajectory_positions = trajectory_positions[valid_mask]
            print(f"  Using {len(trajectory_indices)} valid nodes")
        else:
            trajectory_indices = trajectory_indices
    except Exception as e:
        print(f"  Error mapping trajectory: {e}")
        trajectory_indices = np.array([], dtype=int)
        trajectory_positions = np.array([])
    
    # Create title
    title = f"{args.key}\nWalk {walk_id} ({len(trajectory_indices)} nodes)"
    
    # Save HTML
    html_file = output_dir / f"{output_base}.html"
    create_interactive_3d_html(
        umap_data=umap_data,
        trajectory_indices=trajectory_indices,
        trajectory_positions=trajectory_positions,
        adjacency_matrix=adjacency_matrix,
        token_ids=None,  # Use node indices for coloring
        output_file=str(html_file),
        title=title
    )
    
    # Save metadata
    metadata = {
        'key': args.key,
        'walk_id': int(walk_id),
        'trajectory_length': len(trajectory),
        'trajectory_nodes': trajectory[:100] if len(trajectory) <= 100 else trajectory[:100] + ['...'],  # First 100 nodes
        'valid_trajectory_length': len(trajectory_indices),
        'umap_shape': list(umap_data.shape),
        'distance_matrix_shape': list(distance_matrix.shape),
        'umap_params': {
            'n_components': args.umap_n_components,
            'min_dist': args.umap_min_dist,
            'n_neighbors': args.umap_n_neighbors,
            'metric': 'precomputed',
            'random_state': args.umap_random_state
        },
        'trajectory_params': {
            'max_length': args.max_length,
            'actual_length': len(trajectory_indices),
            'walks_csv': args.walks_csv
        },
        'output_files': {
            'html': str(html_file.name) if html_file.exists() else None
        }
    }
    
    metadata_file = output_dir / f"{output_base}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_file.name}")
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    if html_file.exists():
        print(f"  HTML: {html_file.name}")
    print(f"  Metadata: {metadata_file.name}")


if __name__ == '__main__':
    main()

