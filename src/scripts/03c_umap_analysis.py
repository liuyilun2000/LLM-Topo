"""
Apply UMAP and visualize dimensionality reduction results
Can work with either PCA or downsampled PCA data
Similar to manifold.ipynb visualization
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


def load_pca_data(pca_dir, key, use_downsampled=False):
    """Load PCA data (from 05a) or downsampled PCA data (from 05b)"""
    pca_dir = Path(pca_dir)
    
    if use_downsampled:
        # Try to load downsampled data first
        npz_file = pca_dir / f'{key}_pca_downsampled.npz'
        if not npz_file.exists():
            raise FileNotFoundError(f"Downsampled PCA results not found for {key} in {pca_dir}")
        data = np.load(npz_file)
        pca_reduced = data['pca_reduced']
        selected_indices = data.get('selected_indices', None)
    else:
        # Load regular PCA data
        npz_file = pca_dir / f'{key}_pca.npz'
        if not npz_file.exists():
            raise FileNotFoundError(f"PCA results not found for {key} in {pca_dir}")
        data = np.load(npz_file)
        pca_reduced = data['pca_reduced']
        selected_indices = None
    
    return pca_reduced, selected_indices


def load_fuzzy_distance_matrix(fuzzy_dir, key):
    """Load fuzzy neighborhood distance matrix (from 05b_fuzzy_neighborhood)"""
    fuzzy_dir = Path(fuzzy_dir)
    
    npz_file = fuzzy_dir / f'{key}_fuzzy_dist.npz'
    if not npz_file.exists():
        raise FileNotFoundError(f"Fuzzy distance matrix not found for {key} in {fuzzy_dir}")
    
    data = np.load(npz_file)
    distance_matrix = data['distance_matrix']
    pca_data = data.get('pca_reduced', None)  # Optional, for reference
    
    return distance_matrix, pca_data


def umap_analysis(data, n_components=2, min_dist=0.1, n_neighbors=15, metric='cosine', 
                  distance_matrix=None, random_state=None, **kwargs):
    """
    Perform UMAP dimensionality reduction
    
    Args:
        data: numpy array of shape [n_samples, n_features] (used when metric != 'precomputed')
        n_components: number of dimensions (2 or 3)
        min_dist: minimum distance between points (lower=more local, higher=more global)
        n_neighbors: number of neighbors (lower=more local, higher=more global)
        metric: distance metric to use (default: 'cosine', also supports 'euclidean', 'manhattan', 'precomputed')
        distance_matrix: precomputed distance matrix of shape [n_samples, n_samples] (when metric='precomputed')
        random_state: random seed for reproducibility (default: None)
        **kwargs: additional UMAP parameters
    
    Returns:
        reduced_data: UMAP-transformed data
        info: information dictionary
        umap_reducer: fitted UMAP object
    """
    if not HAS_UMAP:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    
    # Determine if using precomputed distances
    use_precomputed = (metric == 'precomputed' or distance_matrix is not None)
    
    if use_precomputed:
        if distance_matrix is None:
            raise ValueError("distance_matrix must be provided when using metric='precomputed'")
        n_samples = distance_matrix.shape[0]
        print(f"  Using precomputed distance matrix: shape {distance_matrix.shape}")
        # Ensure n_neighbors doesn't exceed data size
        n_neighbors = min(n_neighbors, n_samples - 1)
        input_data = distance_matrix
    else:
        if data is None:
            raise ValueError("data must be provided when not using precomputed distances")
        # Ensure n_neighbors doesn't exceed data size
        n_neighbors = min(n_neighbors, data.shape[0] - 1)
        input_data = data
    
    print(f"  Original shape: {input_data.shape if not use_precomputed else f'{n_samples} samples (precomputed distances)'}")
    print(f"  Target dimensions: {n_components}")
    print(f"  min_dist: {min_dist} (lower=more local, higher=more global)")
    print(f"  n_neighbors: {n_neighbors} (lower=more local, higher=more global)")
    print(f"  metric: {metric}")
    if random_state is not None:
        print(f"  random_state: {random_state}")
    else:
        print(f"  random_state: (random)")
    print(f"  init: spectral (default)")
    
    umap_reducer = umap.UMAP(
        n_components=n_components, 
        min_dist=min_dist, 
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        **kwargs
    )
    reduced = umap_reducer.fit_transform(input_data)
    
    print(f"  Reduced shape: {reduced.shape}")
    
    info = {
        'n_components': n_components,
        'min_dist': min_dist,
        'n_neighbors': n_neighbors,
        'metric': metric,
        'use_precomputed': use_precomputed,
        'random_state': random_state
    }
    
    return reduced, info, umap_reducer


def create_3d_plot(reduced_data, token_ids, title, output_file):
    """Create 3D scatter plot with beautiful rainbow colors and uniform scale"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        reduced_data[:, 2],
        c=token_ids,
        cmap='Spectral_r',  # Beautiful rainbow colormap (reversed)
        alpha=0.7,
        s=50,
        edgecolors='none',  # No black edges for smoother rainbow
        linewidth=0
    )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_zlabel('UMAP 3', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Set uniform scale (same unit length for all axes)
    # Get the data range
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
    z_min, z_max = reduced_data[:, 2].min(), reduced_data[:, 2].max()
    
    # Find the maximum range
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Calculate centers
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2
    
    # Set same range for all axes
    ax.set_xlim([x_center - max_range/2, x_center + max_range/2])
    ax.set_ylim([y_center - max_range/2, y_center + max_range/2])
    ax.set_zlim([z_center - max_range/2, z_center + max_range/2])
    
    # Alternatively, try setting box aspect (if supported)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass  # Older matplotlib versions don't support set_box_aspect
    
    plt.colorbar(scatter, ax=ax, label='Token ID', shrink=0.6, pad=0.1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def create_interactive_3d_html(reduced_data, token_ids, title, output_file):
    """Create interactive 3D plot with plotly (HTML format) with uniform scale"""
    if not PLOTLY_AVAILABLE:
        print("  Skipping interactive HTML (plotly not available)")
        return
    
    # Create a DataFrame for better handling
    import pandas as pd
    df = pd.DataFrame({
        'x': reduced_data[:, 0],
        'y': reduced_data[:, 1],
        'z': reduced_data[:, 2],
        'token_id': token_ids
    })
    
    # Calculate uniform scale range
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
    z_min, z_max = reduced_data[:, 2].min(), reduced_data[:, 2].max()
    
    # Find the maximum range
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Calculate centers
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2
    
    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='token_id',
        color_continuous_scale='Spectral_r',
        title=title,
        labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
        hover_data={'token_id': True},
        opacity=0.7
    )
    
    # Update marker size
    fig.update_traces(
        marker=dict(size=4, line=dict(width=0))
    )
    
    # Update layout for better viewing with uniform scale
    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            bgcolor='white',
            xaxis=dict(
                backgroundcolor="white",
                range=[x_center - max_range/2, x_center + max_range/2]
            ),
            yaxis=dict(
                backgroundcolor="white",
                range=[y_center - max_range/2, y_center + max_range/2]
            ),
            zaxis=dict(
                backgroundcolor="white",
                range=[z_center - max_range/2, z_center + max_range/2]
            ),
            aspectmode='cube',  # Ensure uniform scaling
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    # Save as HTML
    fig.write_html(output_file)
    print(f"  Saved interactive HTML to {Path(output_file).name}")


def create_2d_plot(reduced_data, token_ids, title, output_file):
    """Create 2D scatter plot with beautiful rainbow colors"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=token_ids,
        cmap='Spectral_r',  # Beautiful rainbow colormap (reversed)
        alpha=0.7,
        s=50,
        edgecolors='none',  # No black edges for smoother rainbow
        linewidth=0
    )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Token ID')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_reduction_result(key, reduced_data, token_ids, output_dir, 
                               title_suffix=""):
    """Visualize a single reduction result"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_dims = reduced_data.shape[1]
    title = f'{key}{title_suffix}\n{reduced_data.shape[0]} samples → {n_dims}D'
    
    if n_dims == 3:
        # Save PNG version
        output_file = output_dir / f'{key}_umap_3d.png'
        create_3d_plot(reduced_data, token_ids, title, output_file)
        print(f"  Saved PNG to {output_file.name}")
        
        # Save interactive HTML version
        if PLOTLY_AVAILABLE:
            html_file = output_dir / f'{key}_umap_3d.html'
            create_interactive_3d_html(reduced_data, token_ids, title, html_file)
    elif n_dims == 2:
        output_file = output_dir / f'{key}_umap_2d.png'
        create_2d_plot(reduced_data, token_ids, title, output_file)
        print(f"  Saved to {output_file.name}")
    else:
        # Support arbitrary dimensions in filename
        output_file = output_dir / f'{key}_umap_{n_dims}d.png'
        print(f"  Warning: Unsupported dimensions ({n_dims}D) for visualization, skipping plot")
        return


def main():
    parser = argparse.ArgumentParser(
        description='Apply UMAP and visualize dimensionality reduction results'
    )
    parser.add_argument('--pca_dir', type=str, default=None,
                      help='Directory with PCA results from 05a or downsampled results from 05b')
    parser.add_argument('--fuzzy_dir', type=str, default=None,
                      help='Directory with fuzzy neighborhood distance matrices from 05b_fuzzy_neighborhood')
    parser.add_argument('--representation_dir', type=str, required=True,
                      help='Directory with extracted representations (for metadata)')
    parser.add_argument('--output_dir', type=str, default='./umap_visualization',
                      help='Output directory for plots')
    parser.add_argument('--use_downsampled', action='store_true',
                      help='Use downsampled PCA data from 05b instead of regular PCA from 05a')
    parser.add_argument('--use_fuzzy', action='store_true',
                      help='Use fuzzy neighborhood distance matrices from 05b_fuzzy_neighborhood (metric will be precomputed)')
    parser.add_argument('--umap_n_components', type=int, default=3,
                      help='Target dimensions for UMAP (any positive integer; visualization only supports 2D/3D)')
    parser.add_argument('--umap_min_dist', type=float, default=0.1,
                      help='UMAP min_dist parameter (lower=more local, higher=more global)')
    parser.add_argument('--umap_n_neighbors', type=int, default=15,
                      help='UMAP n_neighbors parameter (lower=more local, higher=more global)')
    parser.add_argument('--umap_metric', type=str, default='cosine',
                      help='UMAP distance metric (default: cosine, also supports euclidean, manhattan, etc.)')
    parser.add_argument('--umap_random_state', type=int, default=None,
                      help='Random seed for UMAP (default: None, for reproducibility set to integer)')
    parser.add_argument('--save_umap_result', action='store_true',
                      help='Save UMAP embeddings as .npz files (default: False)')
    parser.add_argument('--generate_visualizations', action='store_true',
                      help='Generate visualization plots/images (default: False)')
    parser.add_argument('--keys', type=str, nargs='+', default=None,
                      help='Specific keys to visualize (default: all)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_fuzzy:
        if args.fuzzy_dir is None:
            print("Error: --fuzzy_dir is required when using --use_fuzzy")
            return
        if args.pca_dir is not None and not args.use_downsampled:
            print("Warning: --pca_dir is ignored when using --use_fuzzy")
    else:
        if args.pca_dir is None:
            print("Error: --pca_dir is required when not using --use_fuzzy")
            return
    
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
    print("UMAP + Visualization")
    print("="*60)
    
    # Load metadata for token IDs
    from utils import load_representations
    _, metadata = load_representations(args.representation_dir)
    token_ids = np.array([m['token_id'] for m in metadata])
    
    # Find available results
    if args.use_fuzzy:
        fuzzy_dir = Path(args.fuzzy_dir)
        if args.keys:
            keys_to_process = args.keys
        else:
            # Find fuzzy distance matrix files
            npz_files = list(fuzzy_dir.glob('*_fuzzy_dist.npz'))
            keys_to_process = sorted(set(
                f.name.replace('_fuzzy_dist.npz', '') for f in npz_files
            ))
        
        if not keys_to_process:
            print(f"Error: No fuzzy distance matrices found in {fuzzy_dir}")
            print(f"Please run ./05b_fuzzy_neighborhood.sh first")
            return
    else:
        pca_dir = Path(args.pca_dir)
        if args.keys:
            keys_to_process = args.keys
        else:
            if args.use_downsampled:
                # Find downsampled files
                npz_files = list(pca_dir.glob('*_pca_downsampled.npz'))
                keys_to_process = sorted(set(
                    f.name.replace('_pca_downsampled.npz', '') for f in npz_files
                ))
            else:
                # Find regular PCA files
                npz_files = list(pca_dir.glob('*_pca.npz'))
                keys_to_process = sorted(set(
                    f.name.replace('_pca.npz', '') for f in npz_files
                ))
        
        if not keys_to_process:
            data_type = "downsampled" if args.use_downsampled else "PCA"
            print(f"Error: No {data_type} results found in {pca_dir}")
            print(f"Please run ./05a_pca_analysis.sh first" + 
                  (" or ./05b_density_downsampling.sh if using downsampled" if args.use_downsampled else ""))
            return
    
    print(f"\nProcessing {len(keys_to_process)} representations...")
    print(f"Output directory: {args.output_dir}")
    if args.use_fuzzy:
        print(f"Using fuzzy neighborhood distance matrices (precomputed)")
    else:
        print(f"Using {'downsampled' if args.use_downsampled else 'PCA'} data")
    print(f"UMAP target dimensions: {args.umap_n_components}D")
    print(f"UMAP min_dist: {args.umap_min_dist}")
    print(f"UMAP n_neighbors: {args.umap_n_neighbors}")
    if args.use_fuzzy:
        print(f"UMAP metric: precomputed (from fuzzy neighborhood)")
    else:
        print(f"UMAP metric: {args.umap_metric}")
    if args.umap_random_state is not None:
        print(f"UMAP random_state: {args.umap_random_state}")
    else:
        print(f"UMAP random_state: (random)")
    print(f"Save UMAP result: {args.save_umap_result}")
    print(f"Generate visualizations: {args.generate_visualizations}")
    if args.save_umap_result and args.umap_n_components > 3:
        print(f"  Note: Saving {args.umap_n_components}D data (visualization only supports 2D/3D)")
    
    # Process each representation
    for key in keys_to_process:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {key}")
            print(f"{'='*60}")
            
            if args.use_fuzzy:
                # Load fuzzy neighborhood distance matrix
                distance_matrix, pca_data = load_fuzzy_distance_matrix(args.fuzzy_dir, key)
                print(f"Loaded fuzzy distance matrix: {distance_matrix.shape}")
                if pca_data is not None:
                    print(f"  (Original PCA data shape: {pca_data.shape})")
                
                # Use all token_ids (no downsampling with fuzzy neighborhood)
                token_ids_subset = token_ids
                
                # Apply UMAP with precomputed distance matrix
                print("\nApplying UMAP with precomputed distance matrix...")
                umap_data, umap_info, umap_reducer = umap_analysis(
                    data=None,  # Not needed when using precomputed distances
                    n_components=args.umap_n_components,
                    min_dist=args.umap_min_dist,
                    n_neighbors=args.umap_n_neighbors,
                    metric='precomputed',
                    distance_matrix=distance_matrix,
                    random_state=args.umap_random_state
                )
            else:
                # Load PCA or downsampled PCA data
                pca_data, selected_indices = load_pca_data(
                    args.pca_dir, key, use_downsampled=args.use_downsampled
                )
                print(f"Loaded data: {pca_data.shape}")
                
                # If downsampling was applied, subset token_ids accordingly
                if selected_indices is not None:
                    print(f"  (Downsampled from original, {len(selected_indices)} samples selected)")
                    token_ids_subset = token_ids[selected_indices]
                else:
                    token_ids_subset = token_ids
                
                # Apply UMAP
                print("\nApplying UMAP...")
                umap_data, umap_info, umap_reducer = umap_analysis(
                    pca_data,
                    n_components=args.umap_n_components,
                    min_dist=args.umap_min_dist,
                    n_neighbors=args.umap_n_neighbors,
                    metric=args.umap_metric,
                    random_state=args.umap_random_state
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
                
                # Save info as JSON (convert numpy types to Python types)
                info_dict = {
                    'n_components': int(umap_info['n_components']),
                    'min_dist': float(umap_info['min_dist']),
                    'n_neighbors': int(umap_info['n_neighbors']),
                    'metric': str(umap_info['metric']),
                    'use_precomputed': bool(umap_info['use_precomputed']),
                    'random_state': umap_info.get('random_state', None),
                    'n_points': int(umap_data.shape[0]),
                    'n_dims': int(umap_data.shape[1])
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
                    token_ids=token_ids_subset,
                    output_dir=args.output_dir
                )
            
        except FileNotFoundError as e:
            print(f"  Skipping {key}: {e}")
    
    print(f"\n{'='*60}")
    print("UMAP + Visualization complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_dir}")
    if args.save_umap_result:
        print(f"\nUMAP embeddings saved as: {{key}}_umap_{args.umap_n_components}d.npz")
        print(f"  Each file contains: umap_reduced (shape: [n_points, {args.umap_n_components}])")
        if args.umap_n_components > 3:
            print(f"  Note: {args.umap_n_components}D data saved (visualization only supports 2D/3D)")
    if args.generate_visualizations:
        print(f"\nVisualizations saved as: {{key}}_umap_{args.umap_n_components}d.{{png,html}}")


if __name__ == '__main__':
    main()

