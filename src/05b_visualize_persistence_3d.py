#!/usr/bin/env python3
"""
Visualize Topological Features in 3D (Steps x Layers)

This script creates 3D plots showing the emergence of topological features
across training checkpoints (x-axis: steps) and layers (y-axis: representations)
for each dimension instance (z-axis: persistence values).
"""

import json
import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go


def parse_checkpoint_number(checkpoint_dir: str) -> Optional[int]:
    """Extract checkpoint number from directory name.
    
    Returns:
        Checkpoint number (int) for checkpoint-XXX, None for final_model
    """
    match = re.search(r'checkpoint-(\d+)', checkpoint_dir)
    if match:
        return int(match.group(1))
    elif 'final_model' in checkpoint_dir:
        return None  # Will be sorted last
    return None


def find_checkpoints(work_dir: str) -> List[Tuple[str, Optional[int]]]:
    """Find all checkpoint directories in the work directory.
    
    Returns:
        List of (checkpoint_dir, checkpoint_number) tuples, sorted by number
    """
    checkpoints = []
    work_path = Path(work_dir)
    
    if not work_path.exists():
        raise ValueError(f"Work directory does not exist: {work_dir}")
    
    for item in work_path.iterdir():
        if item.is_dir():
            checkpoint_num = parse_checkpoint_number(item.name)
            checkpoints.append((item.name, checkpoint_num))
    
    # Sort: numeric checkpoints first (by number), then final_model
    checkpoints.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))
    
    return checkpoints


def load_persistence_data(checkpoint_path: Path, representation: str) -> Optional[Dict]:
    """Load persistence statistics JSON for a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        representation: Name of representation (e.g., 'final_hidden', 'input_embeds')
    
    Returns:
        Dictionary with persistence data or None if file doesn't exist
    """
    json_file = checkpoint_path / "persistence_barcode" / f"{representation}_statistics.json"
    
    if not json_file.exists():
        return None
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Failed to load {json_file}: {e}", file=sys.stderr)
        return None


def extract_top_k_bars(data: Dict, dimension: str, k: int) -> List[float]:
    """Extract top k longest persistence bars for a dimension.
    
    Args:
        data: Persistence statistics dictionary
        dimension: Dimension name (e.g., 'H0', 'H1')
        k: Number of top bars to extract
    
    Returns:
        List of persistence values (sorted descending)
    """
    if dimension not in data:
        return []
    
    dim_data = data[dimension]
    if 'significant_bars' not in dim_data:
        return []
    
    bars = dim_data['significant_bars']
    # Extract persistence values and sort descending
    persistences = [bar['persistence'] for bar in bars]
    persistences.sort(reverse=True)
    
    # Return top k
    return persistences[:k]


def get_available_dimensions(data: Dict) -> List[str]:
    """Get list of available dimensions in the data."""
    return [dim for dim in data.keys() if dim.startswith('H')]


def get_dimension_color(dimension: str) -> str:
    """Get color for a dimension using the same palette as 05_visualize_persistence_across_checkpoints.py.
    
    Colors: Blue for H0, Green for H1, Red for H2
    """
    colors = {
        'H0': '#3562E8',  # Blue
        'H1': '#059669',  # Green
        'H2': '#DC2626',  # Red
    }
    return colors.get(dimension, '#000000')


def create_3d_plot(
    data_matrix: np.ndarray,
    steps: List[int],
    layers: List[str],
    dimension: str,
    bar_index: int,
    output_dir: Path
):
    """Create a 3D plot for a specific dimension instance.
    
    Args:
        data_matrix: 2D numpy array with shape (num_layers, num_steps) containing persistence values
        steps: List of step numbers (checkpoint numbers)
        layers: List of layer/representation names
        dimension: Dimension name (e.g., 'H0', 'H1')
        bar_index: Index of the bar (0-indexed, 0 = 1st bar, 1 = 2nd bar, etc.)
        output_dir: Directory to save output files
    """
    # Create layer indices for y-axis
    layer_indices = list(range(len(layers)))
    
    # Create meshgrid for surface plot
    X, Y = np.meshgrid(steps, layer_indices)
    Z = data_matrix
    
    # Get color for this dimension
    color = get_dimension_color(dimension)
    
    # Create a colorscale based on the dimension color
    # Convert hex to RGB
    color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    color_rgb_normalized = [c/255.0 for c in color_rgb]
    
    # Create a colorscale that goes from white/light to the dimension color
    colorscale = [
        [0, 'rgba(255, 255, 255, 0.8)'],  # Light/white at minimum
        [0.5, f'rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, 0.6)'],  # Mid-tone
        [1, color]  # Full color at maximum
    ]
    
    # Create hovertext matrix with layer names
    hovertext_matrix = np.empty(Z.shape, dtype=object)
    for i, layer in enumerate(layers):
        for j, step in enumerate(steps):
            persistence_val = Z[i, j]
            if not np.isnan(persistence_val):
                hovertext_matrix[i, j] = f'Step: {step}<br>Layer: {layer}<br>Persistence: {persistence_val:.4f}'
            else:
                hovertext_matrix[i, j] = f'Step: {step}<br>Layer: {layer}<br>Persistence: N/A'
    
    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title="Persistence"),
        hovertext=hovertext_matrix,
        hovertemplate='%{hovertext}<extra></extra>',
    )])
    
    # Calculate z-axis range (starting from 0)
    z_min = 0
    z_max = np.nanmax(Z) if not np.all(np.isnan(Z)) else 1.0
    if np.isnan(z_max) or z_max <= 0:
        z_max = 1.0
    
    # Update layout
    fig.update_layout(
        title=f'{dimension} Bar {bar_index + 1} - 3D Evolution',
        scene=dict(
            xaxis_title='Step',
            yaxis_title='Layer',
            zaxis_title='Persistence',
            xaxis=dict(type='log'),  # Log scale for steps
            yaxis=dict(
                tickmode='array',
                tickvals=layer_indices,
                ticktext=layers,
                title='Layer'
            ),
            zaxis=dict(range=[z_min, z_max]),  # Set z-axis range starting from 0
        ),
        width=1000,
        height=800,
        font=dict(size=12),
    )
    
    # Create output filenames
    safe_dim = dimension.lower()
    safe_bar = f"bar_{bar_index + 1}"
    base_name = f"3d_{safe_dim}_{safe_bar}"
    
    # Save as HTML
    html_path = output_dir / f"{base_name}.html"
    fig.write_html(str(html_path))
    print(f"  Saved HTML: {html_path}")
    
    # Save as PDF (using kaleido or orca)
    pdf_path = output_dir / f"{base_name}.pdf"
    try:
        fig.write_image(str(pdf_path), width=1000, height=800)
        print(f"  Saved PDF: {pdf_path}")
    except Exception as e:
        print(f"  Warning: Could not save PDF (kaleido/orca may not be installed): {e}", file=sys.stderr)
        print(f"  Install with: pip install kaleido", file=sys.stderr)


def create_combined_3d_plot(
    surfaces_data: List[Tuple[np.ndarray, str, int]],
    steps: List[int],
    layers: List[str],
    output_dir: Path
):
    """Create a combined 3D plot with all surfaces.
    
    Args:
        surfaces_data: List of tuples (data_matrix, dimension, bar_index)
        steps: List of step numbers (checkpoint numbers)
        layers: List of layer/representation names
        output_dir: Directory to save output files
    """
    # Create layer indices for y-axis
    layer_indices = list(range(len(layers)))
    
    # Create meshgrid for surface plot
    X, Y = np.meshgrid(steps, layer_indices)
    
    # Create figure
    fig = go.Figure()
    
    # Find global z-axis range (starting from 0)
    z_min = 0
    z_max = 0
    for data_matrix, _, _ in surfaces_data:
        if not np.all(np.isnan(data_matrix)):
            max_val = np.nanmax(data_matrix)
            if not np.isnan(max_val) and max_val > z_max:
                z_max = max_val
    
    if z_max <= 0:
        z_max = 1.0
    
    # Add each surface to the plot
    for data_matrix, dimension, bar_index in surfaces_data:
        # Skip if all NaN
        if np.all(np.isnan(data_matrix)):
            continue
        
        Z = data_matrix
        
        # Get color for this dimension
        color = get_dimension_color(dimension)
        
        # Create a colorscale based on the dimension color
        color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        
        # Create a colorscale that goes from white/light to the dimension color
        colorscale = [
            [0, 'rgba(255, 255, 255, 0.8)'],  # Light/white at minimum
            [0.5, f'rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, 0.6)'],  # Mid-tone
            [1, color]  # Full color at maximum
        ]
        
        # Create hovertext matrix with layer names
        hovertext_matrix = np.empty(Z.shape, dtype=object)
        for i, layer in enumerate(layers):
            for j, step in enumerate(steps):
                persistence_val = Z[i, j]
                if not np.isnan(persistence_val):
                    hovertext_matrix[i, j] = f'{dimension} Bar {bar_index + 1}<br>Step: {step}<br>Layer: {layer}<br>Persistence: {persistence_val:.4f}'
                else:
                    hovertext_matrix[i, j] = f'{dimension} Bar {bar_index + 1}<br>Step: {step}<br>Layer: {layer}<br>Persistence: N/A'
        
        # Add surface to plot
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=colorscale,
            showscale=False,  # Don't show individual colorbars
            hovertext=hovertext_matrix,
            hovertemplate='%{hovertext}<extra></extra>',
            name=f'{dimension} Bar {bar_index + 1}',
            opacity=0.7,  # Slightly transparent so overlapping surfaces are visible
        ))
    
    # Update layout
    fig.update_layout(
        title='All Topological Features - Combined 3D Evolution',
        scene=dict(
            xaxis_title='Step',
            yaxis_title='Layer',
            zaxis_title='Persistence',
            xaxis=dict(type='log'),  # Log scale for steps
            yaxis=dict(
                tickmode='array',
                tickvals=layer_indices,
                ticktext=layers,
                title='Layer'
            ),
            zaxis=dict(range=[z_min, z_max]),  # Set z-axis range starting from 0
        ),
        width=1200,
        height=900,
        font=dict(size=12),
    )
    
    # Create output filenames
    base_name = "3d_combined_all_features"
    
    # Save as HTML
    html_path = output_dir / f"{base_name}.html"
    fig.write_html(str(html_path))
    print(f"  Saved combined HTML: {html_path}")
    
    # Save as PDF (using kaleido or orca)
    pdf_path = output_dir / f"{base_name}.pdf"
    try:
        fig.write_image(str(pdf_path), width=1200, height=900)
        print(f"  Saved combined PDF: {pdf_path}")
    except Exception as e:
        print(f"  Warning: Could not save PDF (kaleido/orca may not be installed): {e}", file=sys.stderr)
        print(f"  Install with: pip install kaleido", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize topological features in 3D (steps x layers)'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        required=True,
        help='Work directory (results/DATASET_NAME/RUN_NAME)'
    )
    parser.add_argument(
        '--representation',
        type=str,
        action='append',
        required=True,
        help='Representation name (can specify multiple times, e.g., --representation final_hidden --representation input_embeds)'
    )
    parser.add_argument(
        '--k',
        type=int,
        nargs='+',
        help='Top k values for each dimension (e.g., --k 5 3 2 for H0, H1, H2). '
             'If single value provided, applies to all dimensions.'
    )
    parser.add_argument(
        '--k-H0',
        type=int,
        help='Top k for H0 dimension'
    )
    parser.add_argument(
        '--k-H1',
        type=int,
        help='Top k for H1 dimension'
    )
    parser.add_argument(
        '--k-H2',
        type=int,
        help='Top k for H2 dimension'
    )
    
    args = parser.parse_args()
    
    # Determine k values for each dimension
    k_values = {}
    
    # First, try dimension-specific k values
    if args.k_H0 is not None:
        k_values['H0'] = args.k_H0
    if args.k_H1 is not None:
        k_values['H1'] = args.k_H1
    if args.k_H2 is not None:
        k_values['H2'] = args.k_H2
    
    # Then, use --k argument if provided
    if args.k:
        if len(args.k) == 1:
            # Single value: apply to all dimensions
            # We'll determine dimensions from first checkpoint
            work_path = Path(args.work_dir)
            checkpoints = find_checkpoints(args.work_dir)
            
            if not checkpoints:
                print(f"Error: No checkpoints found in {args.work_dir}", file=sys.stderr)
                sys.exit(1)
            
            # Load first checkpoint to get dimensions
            first_checkpoint_path = work_path / checkpoints[0][0]
            first_data = None
            for rep in args.representation:
                first_data = load_persistence_data(first_checkpoint_path, rep)
                if first_data:
                    break
            
            if first_data:
                available_dims = get_available_dimensions(first_data)
                for dim in available_dims:
                    k_values[dim] = args.k[0]
        else:
            # Multiple values: map to dimensions
            available_dims = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5']
            for idx, k_val in enumerate(args.k):
                if idx < len(available_dims):
                    k_values[available_dims[idx]] = k_val
    
    if not k_values:
        print("Error: Must specify k values using --k or --k-H0/--k-H1/--k-H2", file=sys.stderr)
        sys.exit(1)
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.work_dir)
    
    if not checkpoints:
        print(f"Error: No checkpoints found in {args.work_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Processing {len(args.representation)} representations: {args.representation}")
    
    # Load data from all checkpoints and representations
    work_path = Path(args.work_dir)
    
    # Get checkpoint numbers (steps) - filter out None (final_model)
    steps = []
    checkpoint_names = []
    for checkpoint_name, checkpoint_num in checkpoints:
        if checkpoint_num is not None:
            steps.append(checkpoint_num)
            checkpoint_names.append(checkpoint_name)
    
    if not steps:
        print("Error: No valid checkpoints found (only final_model?)", file=sys.stderr)
        sys.exit(1)
    
    # Get available dimensions from first checkpoint and representation
    # Try to find data in any checkpoint/representation combination
    first_data = None
    first_checkpoint_path = None
    first_representation = None
    
    print("Searching for persistence data files...")
    for checkpoint_name in checkpoint_names:
        checkpoint_path = work_path / checkpoint_name
        for rep in args.representation:
            json_file = checkpoint_path / "persistence_barcode" / f"{rep}_statistics.json"
            if json_file.exists():
                data = load_persistence_data(checkpoint_path, rep)
                if data:
                    first_data = data
                    first_checkpoint_path = checkpoint_path
                    first_representation = rep
                    print(f"  Found data in: {checkpoint_name}/{rep}")
                    break
            else:
                print(f"  Missing: {checkpoint_name}/persistence_barcode/{rep}_statistics.json")
        if first_data:
            break
    
    if not first_data:
        print(f"\nError: Could not load persistence data from any checkpoint/representation combination", file=sys.stderr)
        print(f"  Expected files: {work_path}/checkpoint-*/persistence_barcode/*_statistics.json", file=sys.stderr)
        print(f"  Representations checked: {args.representation}", file=sys.stderr)
        print(f"  Checkpoints found: {len(checkpoint_names)}", file=sys.stderr)
        print(f"\n  Hint: Make sure you've run the persistence barcode analysis (04b_persistence_barcode.sh)", file=sys.stderr)
        sys.exit(1)
    
    available_dims = get_available_dimensions(first_data)
    
    # Filter k_values to only include available dimensions
    filtered_k_values = {}
    for dim, k_val in k_values.items():
        if dim in available_dims:
            filtered_k_values[dim] = k_val
    
    k_values = filtered_k_values
    
    if not k_values:
        print(f"Error: No valid dimensions found. Available: {available_dims}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing dimensions with k-values: {k_values}")
    
    # Create output directory
    output_dir = work_path / "persistence_3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all surfaces data for combined plot
    all_surfaces_data = []
    
    # For each dimension and bar index, create a 3D plot
    for dimension, k in k_values.items():
        if k <= 0:
            continue
        
        print(f"\nProcessing dimension {dimension} with k={k}")
        
        # For each bar index (1st, 2nd, etc.)
        for bar_idx in range(k):
            print(f"  Creating plot for {dimension} Bar {bar_idx + 1}")
            
            # Create data matrix: rows = layers (representations), cols = steps (checkpoints)
            data_matrix = np.full((len(args.representation), len(steps)), np.nan)
            
            # Fill the matrix
            for layer_idx, representation in enumerate(args.representation):
                for step_idx, checkpoint_name in enumerate(checkpoint_names):
                    checkpoint_path = work_path / checkpoint_name
                    data = load_persistence_data(checkpoint_path, representation)
                    
                    if data is not None:
                        top_k_bars = extract_top_k_bars(data, dimension, k)
                        if bar_idx < len(top_k_bars):
                            data_matrix[layer_idx, step_idx] = top_k_bars[bar_idx]
            
            # Check if we have any valid data
            if np.all(np.isnan(data_matrix)):
                print(f"    Warning: No data available for {dimension} Bar {bar_idx + 1}")
                continue
            
            # Store data for combined plot
            all_surfaces_data.append((data_matrix.copy(), dimension, bar_idx))
            
            # Create individual 3D plot
            create_3d_plot(
                data_matrix=data_matrix,
                steps=steps,
                layers=args.representation,
                dimension=dimension,
                bar_index=bar_idx,
                output_dir=output_dir
            )
    
    # Create combined plot with all surfaces
    if all_surfaces_data:
        print(f"\nCreating combined plot with {len(all_surfaces_data)} surfaces...")
        create_combined_3d_plot(
            surfaces_data=all_surfaces_data,
            steps=steps,
            layers=args.representation,
            output_dir=output_dir
        )
    
    print(f"\n✓ All 3D visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
