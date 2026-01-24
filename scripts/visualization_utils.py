"""
Visualization utility functions for UMAP, 3D plots, and interactive HTML
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, List, Tuple

# Try to import optional dependencies
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def apply_umap(
    data: Optional[np.ndarray] = None,
    distance_matrix: Optional[np.ndarray] = None,
    n_components: int = 3,
    min_dist: float = 0.2,
    n_neighbors: int = 20,
    metric: str = 'precomputed',
    random_state: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Apply UMAP dimensionality reduction.
    
    Args:
        data: Input data array (used when metric != 'precomputed')
        distance_matrix: Precomputed distance matrix (used when metric='precomputed')
        n_components: Target dimensions
        min_dist: Minimum distance between points
        n_neighbors: Number of neighbors
        metric: Distance metric ('precomputed', 'euclidean', 'cosine', etc.)
        random_state: Random seed for reproducibility
        **kwargs: Additional UMAP parameters
    
    Returns:
        reduced_data: UMAP-transformed data
        info: Information dictionary
    """
    if not HAS_UMAP:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    
    use_precomputed = (metric == 'precomputed' or distance_matrix is not None)
    
    if use_precomputed:
        if distance_matrix is None:
            raise ValueError("distance_matrix must be provided when using metric='precomputed'")
        n_samples = distance_matrix.shape[0]
        print(f"  Using precomputed distance matrix: shape {distance_matrix.shape}")
        n_neighbors = min(n_neighbors, n_samples - 1)
        input_data = distance_matrix
    else:
        if data is None:
            raise ValueError("data must be provided when not using precomputed distances")
        n_neighbors = min(n_neighbors, data.shape[0] - 1)
        input_data = data
    
    print(f"  Original shape: {input_data.shape if not use_precomputed else f'{n_samples} samples (precomputed distances)'}")
    print(f"  Target dimensions: {n_components}D")
    print(f"  min_dist: {min_dist}")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  metric: {metric}")
    if random_state is not None:
        print(f"  random_state: {random_state}")
    
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
    
    return reduced, info


def create_3d_scatter_plot(
    reduced_data: np.ndarray,
    token_ids: np.ndarray,
    title: str,
    output_file: str,
    colormap: str = 'Spectral_r'
) -> None:
    """
    Create 3D scatter plot with uniform scale.
    
    Args:
        reduced_data: 3D coordinates array (N, 3)
        token_ids: Token IDs for coloring
        title: Plot title
        output_file: Output file path
        colormap: Colormap name
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        reduced_data[:, 2],
        c=token_ids,
        cmap=colormap,
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
        pass  # Older matplotlib versions don't support set_box_aspect
    
    plt.colorbar(scatter, ax=ax, label='Token ID', shrink=0.6, pad=0.1)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def create_2d_scatter_plot(
    reduced_data: np.ndarray,
    token_ids: np.ndarray,
    title: str,
    output_file: str,
    colormap: str = 'Spectral_r'
) -> None:
    """
    Create 2D scatter plot.
    
    Args:
        reduced_data: 2D coordinates array (N, 2)
        token_ids: Token IDs for coloring
        title: Plot title
        output_file: Output file path
        colormap: Colormap name
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=token_ids,
        cmap=colormap,
        alpha=0.7,
        s=50,
        edgecolors='none',
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


def create_interactive_3d_html(
    umap_data: np.ndarray,
    trajectory_indices: Optional[np.ndarray] = None,
    trajectory_positions: Optional[np.ndarray] = None,
    adjacency_matrix: Optional[np.ndarray] = None,
    token_ids: Optional[np.ndarray] = None,
    output_file: str = "",
    title: str = ""
) -> None:
    """
    Create interactive 3D HTML plot with plotly.
    
    Args:
        umap_data: 3D UMAP coordinates (N, 3)
        trajectory_indices: Indices of trajectory nodes (optional)
        trajectory_positions: Position values for trajectory coloring (optional)
        adjacency_matrix: Adjacency matrix for edge visualization (optional)
        token_ids: Token IDs for point cloud coloring (optional)
        output_file: Output HTML file path
        title: Plot title
    """
    if not PLOTLY_AVAILABLE:
        print("  Skipping interactive HTML (plotly not available)")
        return
    
    # Create DataFrame for background points
    import pandas as pd
    node_indices = np.arange(len(umap_data))
    if token_ids is None:
        token_ids = node_indices
    
    df_bg = pd.DataFrame({
        'x': umap_data[:, 0],
        'y': umap_data[:, 1],
        'z': umap_data[:, 2],
        'node_index': node_indices,
        'token_id': token_ids
    })
    
    fig = go.Figure()
    
    # Plot neighborhood edges if available
    if adjacency_matrix is not None:
        n_nodes = min(len(umap_data), adjacency_matrix.shape[0])
        edge_x, edge_y, edge_z = [], [], []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adjacency_matrix[i, j] > 0:
                    edge_x.extend([umap_data[i, 0], umap_data[j, 0], None])
                    edge_y.extend([umap_data[i, 1], umap_data[j, 1], None])
                    edge_z.extend([umap_data[i, 2], umap_data[j, 2], None])
        
        if len(edge_x) > 0:
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(width=3, color='rgba(80, 80, 80, 0.5)'),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # Add background point cloud
    fig.add_trace(go.Scatter3d(
        x=df_bg['x'], y=df_bg['y'], z=df_bg['z'],
        mode='markers',
        marker=dict(
            size=4,
            color=df_bg['token_id'],
            colorscale='Viridis',
            opacity=0.6,
            line=dict(width=0),
            showscale=False
        ),
        name='Point cloud',
        hovertemplate='Node: %{text}<extra></extra>',
        text=df_bg['node_index'],
        showlegend=False
    ))
    
    # Add trajectory if present
    if trajectory_indices is not None and len(trajectory_indices) > 0:
        traj_points = umap_data[trajectory_indices]
        traj_colors = trajectory_positions if trajectory_positions is not None else np.arange(len(traj_points))
        
        fig.add_trace(go.Scatter3d(
            x=traj_points[:, 0],
            y=traj_points[:, 1],
            z=traj_points[:, 2],
            mode='lines',
            line=dict(
                width=10,
                color=traj_colors,
                colorscale='Hot',
                showscale=False
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
        title='',
        scene=dict(
            xaxis=dict(
                range=[x_center - max_range/2, x_center + max_range/2],
                showbackground=False, showgrid=False, showline=False,
                showticklabels=False, title='', zeroline=False
            ),
            yaxis=dict(
                range=[y_center - max_range/2, y_center + max_range/2],
                showbackground=False, showgrid=False, showline=False,
                showticklabels=False, title='', zeroline=False
            ),
            zaxis=dict(
                range=[z_center - max_range/2, z_center + max_range/2],
                showbackground=False, showgrid=False, showline=False,
                showticklabels=False, title='', zeroline=False
            ),
            aspectmode='cube',
            bgcolor='white'
        ),
        width=1000, height=800,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    if output_file:
        fig.write_html(output_file)
        print(f"  Saved interactive HTML to {Path(output_file).name}")


# Aliases for backward compatibility
create_3d_plot = create_3d_scatter_plot
create_2d_plot = create_2d_scatter_plot


def visualize_reduction_result(
    key: str,
    reduced_data: np.ndarray,
    token_ids: np.ndarray,
    output_dir: str,
    title_suffix: str = ""
) -> None:
    """
    Visualize a single reduction result (wrapper for create_3d_plot/create_2d_plot).
    
    Args:
        key: Representation name
        reduced_data: Reduced data array (N, n_dims)
        token_ids: Token IDs for coloring
        output_dir: Output directory
        title_suffix: Optional title suffix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_dims = reduced_data.shape[1]
    title = f'{key}{title_suffix}\n{reduced_data.shape[0]} samples → {n_dims}D'
    
    if n_dims == 3:
        # Save PNG version
        output_file = output_dir / f'{key}_umap_3d.png'
        create_3d_scatter_plot(reduced_data, token_ids, title, str(output_file))
        print(f"  Saved PNG to {output_file.name}")
        
        # Save interactive HTML version
        if PLOTLY_AVAILABLE:
            html_file = output_dir / f'{key}_umap_3d.html'
            create_interactive_3d_html(
                umap_data=reduced_data,
                token_ids=token_ids,
                output_file=str(html_file),
                title=title
            )
    elif n_dims == 2:
        output_file = output_dir / f'{key}_umap_2d.png'
        create_2d_scatter_plot(reduced_data, token_ids, title, str(output_file))
        print(f"  Saved to {output_file.name}")
    else:
        # Support arbitrary dimensions in filename
        output_file = output_dir / f'{key}_umap_{n_dims}d.png'
        print(f"  Warning: Unsupported dimensions ({n_dims}D) for visualization, skipping plot")
        return
