"""
Apply UMAP using the original graph distance matrix
This is used for validation - comparing LLM representations against ground truth graph topology
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Try to import umap and plotly
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Error: umap-learn not installed. Install with: pip install umap-learn")
    raise

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: plotly not available. Install with: pip install plotly")
    print("Interactive HTML visualizations will be skipped.")


def load_graph_distance_matrix(graph_dir, dataset_name):
    """
    Load the original graph distance matrix
    
    Args:
        graph_dir: directory containing graph files
        dataset_name: dataset name (e.g., "cylinder_x_30x40")
    
    Returns:
        distance_matrix: numpy array of shape [n_samples, n_samples]
    """
    graph_dir = Path(graph_dir)
    
    distance_matrix_file = graph_dir / f"distance_matrix_{dataset_name}.npy"
    if not distance_matrix_file.exists():
        raise FileNotFoundError(f"Graph distance matrix not found: {distance_matrix_file}")
    
    distance_matrix = np.load(distance_matrix_file)
    print(f"Loaded graph distance matrix: {distance_matrix.shape}")
    
    # Handle NaN values (unreachable nodes) - replace with large value
    if np.isnan(distance_matrix).any():
        print(f"  Warning: Found NaN values in distance matrix (unreachable nodes)")
        max_finite = np.nanmax(distance_matrix)
        distance_matrix = np.nan_to_num(distance_matrix, nan=max_finite * 2)
    
    return distance_matrix


def umap_analysis(distance_matrix, n_components=2, min_dist=0.1, n_neighbors=15, **kwargs):
    """
    Perform UMAP dimensionality reduction using precomputed distance matrix
    
    Args:
        distance_matrix: precomputed distance matrix of shape [n_samples, n_samples]
        n_components: number of dimensions (2 or 3)
        min_dist: minimum distance between points (lower=more local, higher=more global)
        n_neighbors: number of neighbors (lower=more local, higher=more global)
        **kwargs: additional UMAP parameters
    
    Returns:
        reduced_data: UMAP-transformed data
        info: information dictionary
        umap_reducer: fitted UMAP object
    """
    if not HAS_UMAP:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    
    n_samples = distance_matrix.shape[0]
    print(f"  Using precomputed graph distance matrix: shape {distance_matrix.shape}")
    # Ensure n_neighbors doesn't exceed data size
    n_neighbors = min(n_neighbors, n_samples - 1)
    
    print(f"  Original shape: {n_samples} samples (precomputed distances)")
    print(f"  Target dimensions: {n_components}")
    print(f"  min_dist: {min_dist} (lower=more local, higher=more global)")
    print(f"  n_neighbors: {n_neighbors} (lower=more local, higher=more global)")
    print(f"  metric: precomputed")
    print(f"  init: spectral (default)")
    
    umap_reducer = umap.UMAP(
        n_components=n_components, 
        min_dist=min_dist, 
        n_neighbors=n_neighbors,
        metric='precomputed',
        **kwargs
    )
    reduced = umap_reducer.fit_transform(distance_matrix)
    
    print(f"  Reduced shape: {reduced.shape}")
    
    info = {
        'n_components': n_components,
        'min_dist': min_dist,
        'n_neighbors': n_neighbors,
        'metric': 'precomputed',
        'use_precomputed': True
    }
    
    return reduced, info, umap_reducer


def create_3d_plot(reduced_data, title, output_file):
    """Create 3D scatter plot with beautiful rainbow colors and uniform scale"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use index as color for visualization
    token_ids = np.arange(reduced_data.shape[0])
    
    scatter = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        reduced_data[:, 2],
        c=token_ids,
        cmap='Spectral_r',
        alpha=0.7,
        s=50,
        edgecolors='none',
        linewidth=0
    )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_zlabel('UMAP 3', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Set uniform scale
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
    z_min, z_max = reduced_data[:, 2].min(), reduced_data[:, 2].max()
    
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2
    
    ax.set_xlim([x_center - max_range/2, x_center + max_range/2])
    ax.set_ylim([y_center - max_range/2, y_center + max_range/2])
    ax.set_zlim([z_center - max_range/2, z_center + max_range/2])
    
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass
    
    plt.colorbar(scatter, ax=ax, label='Node Index', shrink=0.6, pad=0.1)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def create_interactive_3d_html(reduced_data, title, output_file):
    """Create interactive 3D plot with plotly (HTML format) with uniform scale"""
    if not PLOTLY_AVAILABLE:
        print("  Skipping interactive HTML (plotly not available)")
        return
    
    import pandas as pd
    token_ids = np.arange(reduced_data.shape[0])
    
    df = pd.DataFrame({
        'x': reduced_data[:, 0],
        'y': reduced_data[:, 1],
        'z': reduced_data[:, 2],
        'node_index': token_ids
    })
    
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
    z_min, z_max = reduced_data[:, 2].min(), reduced_data[:, 2].max()
    
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2
    
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='node_index',
        color_continuous_scale='Spectral_r',
        title=title,
        labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
        hover_data={'node_index': True},
        opacity=0.7
    )
    
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            bgcolor='white',
            xaxis=dict(backgroundcolor="white", range=[x_center - max_range/2, x_center + max_range/2]),
            yaxis=dict(backgroundcolor="white", range=[y_center - max_range/2, y_center + max_range/2]),
            zaxis=dict(backgroundcolor="white", range=[z_center - max_range/2, z_center + max_range/2]),
            aspectmode='cube',
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    fig.write_html(output_file)
    print(f"  Saved interactive HTML to {Path(output_file).name}")


def create_2d_plot(reduced_data, title, output_file):
    """Create 2D scatter plot with beautiful rainbow colors"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    token_ids = np.arange(reduced_data.shape[0])
    
    scatter = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=token_ids,
        cmap='Spectral_r',
        alpha=0.7,
        s=50,
        edgecolors='none',
        linewidth=0
    )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Node Index')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_reduction_result(key, reduced_data, output_dir, title_suffix=""):
    """Visualize a single reduction result"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_dims = reduced_data.shape[1]
    title = f'{key}{title_suffix}\n{reduced_data.shape[0]} samples → {n_dims}D (Graph Ground Truth)'
    
    if n_dims == 3:
        output_file = output_dir / f'{key}_umap_3d.png'
        create_3d_plot(reduced_data, title, output_file)
        print(f"  Saved PNG to {output_file.name}")
        
        if PLOTLY_AVAILABLE:
            html_file = output_dir / f'{key}_umap_3d.html'
            create_interactive_3d_html(reduced_data, title, html_file)
    elif n_dims == 2:
        output_file = output_dir / f'{key}_umap_2d.png'
        create_2d_plot(reduced_data, title, output_file)
        print(f"  Saved to {output_file.name}")
    else:
        output_file = output_dir / f'{key}_umap_{n_dims}d.png'
        print(f"  Warning: Unsupported dimensions ({n_dims}D) for visualization, skipping plot")
        return


def main():
    parser = argparse.ArgumentParser(
        description='Apply UMAP using the original graph distance matrix (for validation)'
    )
    parser.add_argument('--graph_dir', type=str, required=True,
                      help='Directory with graph distance matrix')
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='Dataset name (e.g., cylinder_x_30x40)')
    parser.add_argument('--output_dir', type=str, default='./umap_visualization',
                      help='Output directory for plots')
    parser.add_argument('--umap_n_components', type=int, default=3,
                      help='Target dimensions for UMAP (any positive integer; visualization only supports 2D/3D)')
    parser.add_argument('--umap_min_dist', type=float, default=0.1,
                      help='UMAP min_dist parameter (lower=more local, higher=more global)')
    parser.add_argument('--umap_n_neighbors', type=int, default=15,
                      help='UMAP n_neighbors parameter (lower=more local, higher=more global)')
    parser.add_argument('--save_umap_result', action='store_true',
                      help='Save UMAP embeddings as .npz files (default: False)')
    parser.add_argument('--generate_visualizations', action='store_true',
                      help='Generate visualization plots/images (default: False)')
    
    args = parser.parse_args()
    
    # Validate dimensions for visualization
    if args.generate_visualizations and args.umap_n_components not in [2, 3]:
        print(f"Error: Visualization only supports 2D or 3D, but {args.umap_n_components}D was requested")
        print("  Set --generate_visualizations to False if you only want to save higher-dimensional data")
        return
    
    # Validate dimensions are positive
    if args.umap_n_components < 1:
        print(f"Error: --umap_n_components must be positive, got {args.umap_n_components}")
        return
    
    print("="*60)
    print("UMAP + Visualization (Graph Ground Truth)")
    print("="*60)
    
    # Load graph distance matrix
    print(f"\nLoading graph distance matrix...")
    distance_matrix = load_graph_distance_matrix(args.graph_dir, args.dataset_name)
    
    # Use "ground_truth" as the key name
    key = "ground_truth"
    
    print(f"\n{'='*60}")
    print(f"Processing: {key} (Graph Distance Matrix)")
    print(f"{'='*60}")
    
    # Apply UMAP with precomputed distance matrix
    print("\nApplying UMAP with precomputed graph distance matrix...")
    umap_data, umap_info, umap_reducer = umap_analysis(
        distance_matrix,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        n_neighbors=args.umap_n_neighbors
    )
    
    # Save UMAP result if requested
    if args.save_umap_result:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        umap_file = output_dir / f'{key}_umap_{args.umap_n_components}d.npz'
        info_file = output_dir / f'{key}_umap_{args.umap_n_components}d_info.json'
        
        # Save numpy array
        np.savez(umap_file,
                umap_reduced=umap_data,
                n_points=umap_data.shape[0],
                n_dims=umap_data.shape[1])
        
        # Save info as JSON
        info_dict = {
            'n_components': int(umap_info['n_components']),
            'min_dist': float(umap_info['min_dist']),
            'n_neighbors': int(umap_info['n_neighbors']),
            'metric': str(umap_info['metric']),
            'use_precomputed': bool(umap_info['use_precomputed']),
            'n_points': int(umap_data.shape[0]),
            'n_dims': int(umap_data.shape[1]),
            'source': 'graph_distance_matrix',
            'dataset_name': args.dataset_name
        }
        with open(info_file, 'w') as f:
            json.dump(info_dict, f, indent=2)
        
        print(f"\nSaved UMAP result: {umap_file.name}")
        print(f"  Shape: {umap_data.shape[0]} points × {umap_data.shape[1]} dimensions")
        print(f"  Info saved to: {info_file.name}")
    
    # Visualize if requested
    if args.generate_visualizations:
        print("\nGenerating visualizations...")
        visualize_reduction_result(
            key=key,
            reduced_data=umap_data,
            output_dir=args.output_dir
        )
    
    print(f"\n{'='*60}")
    print("UMAP + Visualization complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_dir}")
    if args.save_umap_result:
        print(f"\nUMAP embeddings saved as: {key}_umap_{args.umap_n_components}d.npz")
        print(f"  Each file contains: umap_reduced (shape: [n_points, {args.umap_n_components}])")
        if args.umap_n_components > 3:
            print(f"  Note: {args.umap_n_components}D data saved (visualization only supports 2D/3D)")


if __name__ == '__main__':
    main()

