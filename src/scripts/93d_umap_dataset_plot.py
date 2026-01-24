"""
Visualize UMAP embedding directly from dataset neighboring matrix with trajectory overlay
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
from scipy.sparse.csgraph import shortest_path

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


def load_adjacency_matrix(graph_dir, dataset_name):
    """Load adjacency matrix from graph directory"""
    graph_dir = Path(graph_dir)
    adjacency_file = graph_dir / f'A_{dataset_name}.npy'
    
    if not adjacency_file.exists():
        return None
    
    adjacency = np.load(adjacency_file)
    # Make sure it's binary (convert to 0/1)
    adjacency = (adjacency > 0).astype(int)
    return adjacency


def load_dataset_matrix(graph_dir, dataset_name, matrix_type='auto'):
    """
    Load neighboring matrix from graph directory
    
    Args:
        graph_dir: path to graph directory
        dataset_name: dataset name (e.g., "torus_30x40")
        matrix_type: 'auto', 'distance', or 'adjacency'
    
    Returns:
        distance_matrix: distance matrix for UMAP (always returns distance matrix)
        matrix_type_used: which matrix type was actually used
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
            # Convert adjacency to distance matrix using shortest paths
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
        # Convert adjacency to distance matrix using shortest paths
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


def load_graph_info(graph_dir, dataset_name):
    """
    Load graph nodes information.
    
    Note: Polygon-based graphs don't have H, W grid dimensions or (i, j) coordinates.
    This function returns None for grid-related data since it's not applicable.
    """
    graph_dir = Path(graph_dir)
    
    # Load nodes file
    nodes_file = graph_dir / f"nodes_{dataset_name}.csv"
    if not nodes_file.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
    
    nodes_df = pd.read_csv(nodes_file)
    
    # Polygon-based graphs don't have grid coordinates (i, j)
    # Return None to indicate grid-based visualization is not available
    return None, None, None


def create_2d_grid_plot(node_to_grid, H, W, trajectory, trajectory_positions, adjacency_matrix, output_file):
    """
    Create 2D grid plot showing H×W surface with trajectory using plotly.
    
    Note: This function is not applicable for polygon-based graphs which don't have
    grid coordinates. Returns without creating plot.
    """
    if node_to_grid is None or H is None or W is None:
        print("  Skipping 2D grid plot (not applicable for polygon-based graphs)")
        return
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
    
    # Plot neighborhood edges (semi-transparent lines) if adjacency matrix is available
    if adjacency_matrix is not None:
        # Find all edges (non-zero entries in upper triangle of adjacency matrix)
        n_nodes = adjacency_matrix.shape[0]
        edge_x = []
        edge_y = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adjacency_matrix[i, j] > 0:
                    # Check if both nodes are in grid_positions
                    if i in grid_positions and j in grid_positions:
                        i_coord, j_coord_i = grid_positions[i]
                        i_coord_j, j_coord_j = grid_positions[j]
                        # Add edge coordinates (j is x, i is y)
                        edge_x.extend([j_coord_i, j_coord_j, None])
                        edge_y.extend([i_coord, i_coord_j, None])
        
        if len(edge_x) > 0:
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(
                    width=2.5,
                    color='rgba(80, 80, 80, 0.5)'  # Darker, more visible gray
                ),
                hoverinfo='skip',
                showlegend=False
            ))
    
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
    
    # Plot trajectory if present (bold gradient line only)
    if len(trajectory) > 0 and len(trajectory_positions) > 0:
        # Get grid coordinates for trajectory nodes
        traj_grid_coords = []
        valid_positions = []
        for idx, node_id in enumerate(trajectory[:len(trajectory_positions)]):
            if node_id in grid_positions:
                i, j = grid_positions[node_id]
                traj_grid_coords.append((j, i))  # Plotly uses (x, y) = (j, i)
                valid_positions.append(trajectory_positions[idx])
        
        if len(traj_grid_coords) > 1:
            traj_grid_coords = np.array(traj_grid_coords)
            valid_positions = np.array(valid_positions)
            
            # Normalize positions to [0, 1] for color mapping
            if valid_positions.max() > valid_positions.min():
                normalized_positions = (valid_positions - valid_positions.min()) / (valid_positions.max() - valid_positions.min())
            else:
                normalized_positions = np.zeros_like(valid_positions)
            
            # Create gradient line by plotting segments between consecutive points
            # Each segment gets its color from the starting position
            # Get Hot colormap
            hot_cmap = plt.cm.get_cmap('hot')
            
            # Plot segments for smooth gradient effect with curvature
            for i in range(len(traj_grid_coords) - 1):
                pos = normalized_positions[i]
                # Get RGB color from Hot colormap
                rgba = hot_cmap(pos)
                color_str = f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})'
                
                # Interpolate intermediate points for smoother curves
                # Use 3 points: start, middle, end for better spline curvature
                x_start, y_start = traj_grid_coords[i, 0], traj_grid_coords[i, 1]
                x_end, y_end = traj_grid_coords[i+1, 0], traj_grid_coords[i+1, 1]
                
                # Create a slight intermediate point for curvature
                # Offset perpendicular to the line direction
                dx = x_end - x_start
                dy = y_end - y_start
                # Perpendicular direction
                perp_x = -dy
                perp_y = dx
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    # Normalize perpendicular
                    perp_x /= length
                    perp_y /= length
                    # Curvature proportional to line length - longer lines get more curvature
                    curvature_offset = length * 0.25  # Proportional to length, no cap
                    x_mid = (x_start + x_end) / 2 + perp_x * curvature_offset
                    y_mid = (y_start + y_end) / 2 + perp_y * curvature_offset
                else:
                    x_mid = (x_start + x_end) / 2
                    y_mid = (y_start + y_end) / 2
                
                # Plot segment with 3 points for smooth spline curve
                fig.add_trace(go.Scatter(
                    x=[x_start, x_mid, x_end],
                    y=[y_start, y_mid, y_end],
                    mode='lines',
                    line=dict(
                        width=8,  # Bold line
                        color=color_str,
                        smoothing=1.0  # Maximum smoothing for spline
                    ),
                    line_shape='spline',
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
                               adjacency_matrix, output_file, title=""):
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
    
    # Plot neighborhood edges (semi-transparent lines) if adjacency matrix is available
    if adjacency_matrix is not None:
        n_nodes = min(len(umap_data), adjacency_matrix.shape[0])
        edge_x = []
        edge_y = []
        edge_z = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adjacency_matrix[i, j] > 0:
                    # Add edge coordinates
                    edge_x.extend([umap_data[i, 0], umap_data[j, 0], None])
                    edge_y.extend([umap_data[i, 1], umap_data[j, 1], None])
                    edge_z.extend([umap_data[i, 2], umap_data[j, 2], None])
        
        if len(edge_x) > 0:
            fig.add_trace(go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                line=dict(
                    width=3,
                    color='rgba(80, 80, 80, 0.5)'  # Darker, more visible gray
                ),
                hoverinfo='skip',
                showlegend=False
            ))
    
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
        description='Visualize UMAP embedding from dataset neighboring matrix with trajectory overlay'
    )
    parser.add_argument('--graph_dir', type=str, required=True,
                      help='Directory with graph files (adjacency/distance matrices, nodes CSV)')
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='Dataset name (e.g., torus_30x40)')
    parser.add_argument('--matrix_type', type=str, default='auto',
                      choices=['auto', 'distance', 'adjacency'],
                      help='Matrix type to use: auto (prefer distance), distance, or adjacency')
    parser.add_argument('--walks_csv', type=str, required=True,
                      help='Path to walks CSV file (e.g., sequences/walks_*.csv)')
    parser.add_argument('--walk_id', type=int, default=None,
                      help='Walk ID to use (if not specified, uses --trajectory_idx)')
    parser.add_argument('--trajectory_idx', type=int, default=0,
                      help='Index of trajectory in CSV (0-based, default: 0)')
    parser.add_argument('--output_dir', type=str, default='./umap_dataset_plot',
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
    
    args = parser.parse_args()
    
    if args.umap_n_components != 3:
        print("Warning: This script is designed for 3D visualization. Using 3D anyway.")
        args.umap_n_components = 3
    
    print("="*60)
    print("UMAP Visualization from Dataset Matrix with Trajectory")
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
    output_base = f"dataset_{args.matrix_type}_walk_{walk_id}_umap_3d"
    
    # Load graph info for 2D plot (before UMAP)
    graph_dir = Path(args.graph_dir)
    node_to_grid = None
    H = None
    W = None
    
    try:
        print(f"\nLoading graph info from: {graph_dir}")
        node_to_grid, H, W = load_graph_info(graph_dir, args.dataset_name)
        print(f"  Grid dimensions: H={H}, W={W}")
        print(f"  Loaded {len(node_to_grid)} node mappings")
    except Exception as e:
        print(f"  Warning: Could not load graph info: {e}")
        print(f"  Skipping 2D grid plot")
        node_to_grid = None
    
    # Load adjacency matrix for neighborhood edges
    adjacency_matrix = None
    try:
        print(f"\nLoading adjacency matrix for neighborhood edges...")
        adjacency_matrix = load_adjacency_matrix(args.graph_dir, args.dataset_name)
        if adjacency_matrix is not None:
            print(f"  Loaded adjacency matrix: shape {adjacency_matrix.shape}")
        else:
            print(f"  Adjacency matrix not found, skipping neighborhood edges")
    except Exception as e:
        print(f"  Warning: Could not load adjacency matrix: {e}")
        print(f"  Neighborhood edges will be skipped")
    
    # Save 2D grid plot if graph info is available (before UMAP)
    pdf_file = None
    if node_to_grid is not None:
        pdf_file = output_dir / f"{output_base}_grid_2d.pdf"
        create_2d_grid_plot(
            node_to_grid, H, W, limited_trajectory, trajectory_positions,
            adjacency_matrix, str(pdf_file)
        )
    
    # Load dataset matrix for UMAP
    print(f"\nLoading dataset matrix (type: {args.matrix_type})...")
    distance_matrix, matrix_type_used = load_dataset_matrix(
        args.graph_dir, args.dataset_name, args.matrix_type
    )
    print(f"  Matrix type used: {matrix_type_used}")
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
    title = f"Dataset Matrix ({matrix_type_used})\nWalk {walk_id} ({len(trajectory_indices)} nodes)"
    
    # Save HTML
    if PLOTLY_AVAILABLE:
        html_file = output_dir / f"{output_base}.html"
        create_interactive_3d_html(
            umap_data, trajectory_indices, trajectory_positions,
            adjacency_matrix, str(html_file), title=title
        )
    else:
        html_file = None
    
    # Save metadata
    metadata = {
        'dataset_name': args.dataset_name,
        'matrix_type': matrix_type_used,
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
            'html': str(html_file.name) if html_file and PLOTLY_AVAILABLE else None,
            'pdf_2d': str(pdf_file.name) if node_to_grid is not None else None
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
    if PLOTLY_AVAILABLE and html_file:
        print(f"  HTML: {html_file.name}")
    if node_to_grid is not None and pdf_file:
        print(f"  PDF (2D grid): {pdf_file.name}")
    print(f"  Metadata: {metadata_file.name}")


if __name__ == '__main__':
    main()
