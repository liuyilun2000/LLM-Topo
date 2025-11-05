"""
Visualize UMAP embedding from fuzzy neighborhood distance matrix with trajectory overlay
Creates a 3D plot with point cloud and trajectory colored by time
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import pandas as pd

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


def load_fuzzy_distance_matrix(fuzzy_dir, key):
    """Load fuzzy neighborhood distance matrix"""
    fuzzy_dir = Path(fuzzy_dir)
    
    npz_file = fuzzy_dir / f'{key}_fuzzy_dist.npz'
    if not npz_file.exists():
        raise FileNotFoundError(f"Fuzzy distance matrix not found for {key} in {fuzzy_dir}")
    
    data = np.load(npz_file)
    distance_matrix = data['distance_matrix']
    pca_data = data.get('pca_reduced', None)
    
    return distance_matrix, pca_data


def load_trajectory(walks_csv, walk_id=None, trajectory_idx=None):
    """
    Load a trajectory from walks CSV file
    
    Args:
        walks_csv: path to walks CSV file
        walk_id: specific walk_id to load (if None, uses trajectory_idx)
        trajectory_idx: index of trajectory in CSV (0-based, if walk_id is None)
    
    Returns:
        trajectory: list of node IDs (as integers)
        walk_id: the walk_id used
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


def apply_umap(distance_matrix, n_components=3, min_dist=0.2, n_neighbors=20, random_state=None):
    """Apply UMAP to distance matrix"""
    if not HAS_UMAP:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    
    n_samples = distance_matrix.shape[0]
    n_neighbors = min(n_neighbors, n_samples - 1)
    
    print(f"  Applying UMAP to distance matrix: shape {distance_matrix.shape}")
    print(f"  Target dimensions: {n_components}D")
    print(f"  min_dist: {min_dist}")
    print(f"  n_neighbors: {n_neighbors}")
    if random_state is not None:
        print(f"  random_state: {random_state}")
    
    umap_reducer = umap.UMAP(
        n_components=n_components,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        metric='precomputed',
        random_state=random_state
    )
    reduced = umap_reducer.fit_transform(distance_matrix)
    
    print(f"  Reduced shape: {reduced.shape}")
    return reduced


def load_graph_info(graph_dir):
    """Load graph nodes and H, W dimensions"""
    graph_dir = Path(graph_dir)
    
    # Try to find graph_info JSON first (contains H, W)
    graph_info_files = list(graph_dir.glob('graph_info_*.json'))
    if graph_info_files:
        with open(graph_info_files[0], 'r') as f:
            graph_info = json.load(f)
            H = graph_info.get('H')
            W = graph_info.get('W')
            topology = graph_info.get('topology')
            dataset_name = f"{topology}_{H}x{W}"
    else:
        # Infer from filename pattern
        # Try to find nodes file
        nodes_files = list(graph_dir.glob('nodes_*.csv'))
        if not nodes_files:
            raise FileNotFoundError(f"Could not find graph files in {graph_dir}")
        # Extract dataset name from nodes file
        dataset_name = nodes_files[0].stem.replace('nodes_', '')
        # Try to parse H and W from dataset name (format: topology_HxW)
        parts = dataset_name.split('_')
        if len(parts) >= 2:
            hw = parts[-1]  # e.g., "30x40"
            if 'x' in hw:
                H, W = map(int, hw.split('x'))
            else:
                raise ValueError(f"Could not parse H, W from dataset name: {dataset_name}")
        else:
            raise ValueError(f"Could not parse dataset name: {dataset_name}")
    
    # Load nodes file
    nodes_file = graph_dir / f"nodes_{dataset_name}.csv"
    if not nodes_file.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
    
    nodes_df = pd.read_csv(nodes_file)
    
    # Create mapping from node_id to (i, j) grid coordinates
    # node_id is the actual node ID used in walks
    # i, j are the grid coordinates (row, column)
    node_to_grid = {}
    for _, row in nodes_df.iterrows():
        node_id = int(row['node_id'])
        i = int(row['i'])
        j = int(row['j'])
        node_to_grid[node_id] = (i, j)
    
    return node_to_grid, H, W


def create_2d_grid_plot(node_to_grid, H, W, trajectory, trajectory_positions, output_file):
    """Create 2D grid plot showing H×W surface with trajectory using plotly"""
    if not PLOTLY_AVAILABLE:
        print("  Skipping 2D grid plot (plotly not available)")
        return
    
    # Create grid of all nodes
    grid_colors = np.zeros((H, W))
    grid_positions = {}
    
    # Fill grid with node indices (for coloring)
    for node_id, (i, j) in node_to_grid.items():
        if 0 <= i < H and 0 <= j < W:
            grid_colors[i, j] = node_id
            grid_positions[node_id] = (i, j)
    
    # Create figure
    fig = go.Figure()
    
    # Plot background points with Viridis colormap (like 3D plot)
    x_bg_points = []
    y_bg_points = []
    color_bg_points = []
    for node_id, (i, j) in grid_positions.items():
        x_bg_points.append(j)  # x-coordinate is column (j)
        y_bg_points.append(i)  # y-coordinate is row (i)
        color_bg_points.append(node_id)  # Color by node_id
    
    fig.add_trace(go.Scatter(
        x=x_bg_points,
        y=y_bg_points,
        mode='markers',
        marker=dict(
            size=10,  # Similar to 3D plot
            color=color_bg_points,
            colorscale='Viridis',  # Same as 3D plot
            opacity=0.6,  # Similar to 3D plot
            line=dict(width=0)
        ),
        name='Point cloud',
        hovertemplate='Node: %{text}<extra></extra>',
        text=color_bg_points,
        showlegend=False
    ))
    
    # Plot trajectory if present
    if len(trajectory) > 0 and len(trajectory_positions) > 0:
        # Get grid coordinates for trajectory nodes
        traj_grid_coords = []
        valid_positions = []
        for idx, node_id in enumerate(trajectory[:len(trajectory_positions)]):
            if node_id in grid_positions:
                i, j = grid_positions[node_id]
                traj_grid_coords.append((j, i))  # Plotly uses (x, y) = (j, i)
                valid_positions.append(trajectory_positions[idx])
        
        if len(traj_grid_coords) > 0:
            traj_grid_coords = np.array(traj_grid_coords)
            valid_positions = np.array(valid_positions)
            
            # Plot trajectory: first add black connecting line (beneath markers)
            fig.add_trace(go.Scatter(
                x=traj_grid_coords[:, 0],
                y=traj_grid_coords[:, 1],
                mode='lines',
                line=dict(
                    width=4,
                    color='black'
                ),
                line_shape='spline',
                hoverinfo='skip',
                showlegend=False,
                opacity=0.2 
            ))
            
            # Then add gradient-colored markers (on top of line)
            fig.add_trace(go.Scatter(
                x=traj_grid_coords[:, 0],
                y=traj_grid_coords[:, 1],
                mode='markers',
                marker=dict(
                    size=16,  # Larger markers
                    color=valid_positions.tolist(),
                    colorscale='Hot',
                    showscale=False,
                    line=dict(color='rgba(0,0,0,0.2)', width=2)  # <--- Black outline with opacity 0.5
                ),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # Update layout - remove axes, labels, ticks, titles
    fig.update_layout(
        title='',
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            range=[-0.5, W-0.5]
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            range=[-0.5, H-0.5],
            scaleanchor='x',
            scaleratio=1
        ),
        width=1000,
        height=int(1000 * H / W),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Save as PDF (plotly can save as PDF, requires kaleido)
    try:
        fig.write_image(output_file, format='pdf', width=1000, height=int(1000 * H / W))
        print(f"  Saved 2D grid PDF to {Path(output_file).name}")
    except Exception as e:
        print(f"  Warning: Could not save PDF (kaleido may not be installed): {e}")
        print(f"  Saving as HTML instead...")
        html_file = str(output_file).replace('.pdf', '.html')
        fig.write_html(html_file)
        print(f"  Saved 2D grid HTML to {Path(html_file).name}")

def create_interactive_3d_html(umap_data, trajectory_indices, trajectory_positions,
                               output_file, title=""):
    """Create interactive 3D HTML plot with plotly"""
    if not PLOTLY_AVAILABLE:
        print("  Skipping interactive HTML (plotly not available)")
        return
    
    
    # Create DataFrame for background points
    node_indices = np.arange(len(umap_data))
    df_bg = pd.DataFrame({
        'x': umap_data[:, 0],
        'y': umap_data[:, 1],
        'z': umap_data[:, 2],
        'node_index': node_indices
    })
    
    # Create figure
    fig = go.Figure()
    
    # Add background point cloud with twilight colormap (plotly built-in)
    fig.add_trace(go.Scatter3d(
        x=df_bg['x'],
        y=df_bg['y'],
        z=df_bg['z'],
        mode='markers',
        marker=dict(
            size=4,
            color=df_bg['node_index'],
            colorscale='Viridis',  # Use plotly built-in colorscale directly
            opacity=0.6,
            line=dict(width=0),
            showscale=False  # Hide colorbar
        ),
        name='Point cloud',
        hovertemplate='Node: %{text}<extra></extra>',
        text=df_bg['node_index'],
        showlegend=False
    ))
    
    # Add trajectory if present
    if len(trajectory_indices) > 0:
        traj_points = umap_data[trajectory_indices]
        
        # Create trajectory line
        fig.add_trace(go.Scatter3d(
            x=traj_points[:, 0],
            y=traj_points[:, 1],
            z=traj_points[:, 2],
            mode='lines',
            line=dict(
                width=10,
                color=trajectory_positions,
                colorscale='Hot',  # Use plotly built-in colorscale directly
                showscale=False  # Hide colorbar
            ),
            opacity=0.8,
            name='Trajectory',
            hovertemplate='Position: %{marker.color:.0f}<extra></extra>',
            showlegend=False
        ))
    
    # Set uniform scale
    x_min, x_max = umap_data[:, 0].min(), umap_data[:, 0].max()
    y_min, y_max = umap_data[:, 1].min(), umap_data[:, 1].max()
    z_min, z_max = umap_data[:, 2].min(), umap_data[:, 2].max()
    
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2
    
    fig.update_layout(
        title='',  # No title
        scene=dict(
            xaxis=dict(
                range=[x_center - max_range/2, x_center + max_range/2],
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                zeroline=False
            ),
            yaxis=dict(
                range=[y_center - max_range/2, y_center + max_range/2],
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                zeroline=False
            ),
            zaxis=dict(
                range=[z_center - max_range/2, z_center + max_range/2],
                showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                title='',
                zeroline=False
            ),
            aspectmode='cube',
            bgcolor='white'  # White background
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0),  # No margins
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    fig.write_html(output_file)
    print(f"  Saved interactive HTML to {Path(output_file).name}")


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
                      help='Directory with graph files (nodes CSV, graph_info JSON). If not provided, will try to infer from paths.')
    
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
    
    # Try to load graph info for 2D plot (before UMAP)
    graph_dir = None
    node_to_grid = None
    H = None
    W = None
    
    if args.graph_dir:
        graph_dir = Path(args.graph_dir)
    else:
        # Try to infer graph directory from fuzzy_dir or walks_csv
        fuzzy_path = Path(args.fuzzy_dir)
        walks_path = Path(args.walks_csv)
        
        # Look for graph directory in common locations
        possible_graph_dirs = [
            fuzzy_path.parent.parent / 'graph',  # results/DATASET/graph
            walks_path.parent.parent / 'graph',   # results/DATASET/graph
            fuzzy_path.parent / 'graph',           # fuzzy_dir/../graph
            walks_path.parent / 'graph',           # walks_dir/../graph
        ]
        
        for gd in possible_graph_dirs:
            if gd.exists() and (list(gd.glob('graph_info_*.json')) or list(gd.glob('nodes_*.csv'))):
                graph_dir = gd
                break
    
    if graph_dir and graph_dir.exists():
        try:
            print(f"\nLoading graph info from: {graph_dir}")
            node_to_grid, H, W = load_graph_info(graph_dir)
            print(f"  Grid dimensions: H={H}, W={W}")
            print(f"  Loaded {len(node_to_grid)} node mappings")
        except Exception as e:
            print(f"  Warning: Could not load graph info: {e}")
            print(f"  Skipping 2D grid plot")
            graph_dir = None
    else:
        print(f"\nWarning: Graph directory not found. Skipping 2D grid plot.")
        print(f"  To enable 2D plot, provide --graph_dir argument")
    
    # Save 2D grid plot if graph info is available (before UMAP)
    pdf_file = None
    if graph_dir and node_to_grid is not None:
        pdf_file = output_dir / f"{output_base}_grid_2d.pdf"
        create_2d_grid_plot(
            node_to_grid, H, W, limited_trajectory, trajectory_positions,
            str(pdf_file)
        )
    
    # Load fuzzy distance matrix for UMAP
    print(f"\nLoading fuzzy distance matrix for key: {args.key}")
    distance_matrix, pca_data = load_fuzzy_distance_matrix(args.fuzzy_dir, args.key)
    print(f"  Distance matrix shape: {distance_matrix.shape}")
    
    # Apply UMAP
    print(f"\nApplying UMAP...")
    umap_data = apply_umap(
        distance_matrix,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        n_neighbors=args.umap_n_neighbors,
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
    if PLOTLY_AVAILABLE:
        html_file = output_dir / f"{output_base}.html"
        create_interactive_3d_html(
            umap_data, trajectory_indices, trajectory_positions,
            str(html_file), title=title
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
            'html': str(html_file.name) if PLOTLY_AVAILABLE else None,
            'pdf_2d': str(pdf_file.name) if graph_dir and node_to_grid is not None else None
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
    if PLOTLY_AVAILABLE:
        print(f"  HTML: {html_file.name}")
    if graph_dir and node_to_grid is not None:
        print(f"  PDF (2D grid): {pdf_file.name}")
    print(f"  Metadata: {metadata_file.name}")


if __name__ == '__main__':
    main()

