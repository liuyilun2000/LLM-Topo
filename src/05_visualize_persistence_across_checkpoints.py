#!/usr/bin/env python3
"""
Visualize Top K-Longest Persistence Bars Across Checkpoints

This script loads persistence barcode statistics from multiple checkpoints
and visualizes how the top k-longest bars vary across training checkpoints.
"""

import json
import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


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


def plot_persistence_across_checkpoints(
    checkpoint_data: List[Tuple[str, Optional[int], Dict]],
    representation: str,
    k_values: Dict[str, int],
    output_path: str
):
    """Plot top k persistence bars across checkpoints.
    
    Args:
        checkpoint_data: List of (checkpoint_name, checkpoint_num, data_dict) tuples
        representation: Name of representation being visualized
        k_values: Dictionary mapping dimension -> k value
        output_path: Path to save the plot
    """
    # Get all available dimensions from first checkpoint
    if not checkpoint_data:
        print("Error: No checkpoint data available", file=sys.stderr)
        return
    
    available_dims = get_available_dimensions(checkpoint_data[0][2])
    
    # Filter k_values to only include available dimensions
    k_values = {dim: k_values[dim] for dim in k_values if dim in available_dims}
    
    if not k_values:
        print(f"Error: No valid dimensions found. Available: {available_dims}", file=sys.stderr)
        return
    
    # Prepare data for plotting - map checkpoint indices to actual checkpoint numbers
    checkpoint_x_coords = []  # Actual x-coordinates (checkpoint numbers)
    checkpoint_labels = []  # Labels for display
    
    # Find max checkpoint number to handle final_model
    max_checkpoint_num = 0
    for checkpoint_name, checkpoint_num, _ in checkpoint_data:
        if checkpoint_num is not None and checkpoint_num > max_checkpoint_num:
            max_checkpoint_num = checkpoint_num
    
    for checkpoint_name, checkpoint_num, _ in checkpoint_data:
        if checkpoint_num is not None:
            checkpoint_x_coords.append(checkpoint_num)
            checkpoint_labels.append(str(checkpoint_num))  # Only number, no "checkpoint-"
        else:
            # For final_model, use a value slightly larger than max checkpoint
            final_x = max_checkpoint_num * 1.1 if max_checkpoint_num > 0 else 1000
            checkpoint_x_coords.append(final_x)
            checkpoint_labels.append("final")
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Beautiful color palette: blue, red, green for H0, H1, H2
    colors = ['#3562E8', '#059669', '#DC2626']  # Blue, Green, Red
    line_styles = ['-', '--', '-.', ':']
    
    # Plot each dimension
    for dim_idx, (dim, k) in enumerate(k_values.items()):
        color = colors[dim_idx % len(colors)]
        line_style = line_styles[dim_idx % len(line_styles)]
        
        # Extract top k bars for each checkpoint
        all_persistences = []  # List of lists, one per checkpoint
        all_averages = []  # List of mean persistence values
        
        for checkpoint_name, checkpoint_num, data in checkpoint_data:
            top_k = extract_top_k_bars(data, dim, k)
            all_persistences.append(top_k)
            
            # Extract mean persistence
            if dim in data and 'mean_persistence' in data[dim]:
                all_averages.append(data[dim]['mean_persistence'])
            else:
                all_averages.append(None)
        
        # Plot top k bars with 100% opacity
        max_bars = max(len(p) for p in all_persistences)
        
        for bar_idx in range(max_bars):
            values = []
            x_coords = []
            
            for checkpoint_idx, persistences in enumerate(all_persistences):
                if bar_idx < len(persistences):
                    values.append(persistences[bar_idx])
                    x_coords.append(checkpoint_x_coords[checkpoint_idx])
            
            if values:
                # Use actual checkpoint numbers for x-axis
                label = f'{dim} Bar {bar_idx + 1}'
                ax.plot(x_coords, values, 
                       marker='o', markersize=3, alpha=1.0,  # 100% opacity
                       color=color, linestyle=line_style,
                       linewidth=1.5,
                       label=label)
        
        # Plot average line with 50% opacity
        avg_values = []
        avg_x_coords = []
        
        for checkpoint_idx, avg_val in enumerate(all_averages):
            if avg_val is not None:
                avg_values.append(avg_val)
                avg_x_coords.append(checkpoint_x_coords[checkpoint_idx])
        
        if avg_values:
            label = f'{dim} Average'
            ax.plot(avg_x_coords, avg_values,
                   marker='', alpha=0.5,  # 50% opacity, no markers
                   color=color, linestyle='-',
                   linewidth=1.5,
                   label=label)

    # Set x-axis to logarithmic scale
    ax.ticklabel_format(useMathText=True)
    ax.set_xscale('log')

    # Set x-axis ticks and labels
    #ax.set_xticks(checkpoint_x_coords)
    #ax.set_xticklabels(checkpoint_labels, rotation=45, ha='right')
    ax.set_xlabel('Step', fontsize=15)
    ax.set_ylabel('Persistence', fontsize=15)
    #ax.set_title(f'Persistence Bars Evolution: {representation}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    #ax.set_xlim(left=1, right=max_checkpoint_num)
    ax.set_ylim(bottom=0, top=5.2) 

    # Make tick labels larger
    ax.tick_params(axis='both', labelsize=12)

    # Create single legend with all lines
    handles, labels = ax.get_legend_handles_labels()
    
    # Sort labels by dimension, then by type (bars first, then average)
    def sort_key(label):
        parts = label.split()
        dim = parts[0]  # H0, H1, H2, etc.
        dim_num = int(dim[1:]) if len(dim) > 1 else 0  # Extract number from H0, H1, etc.
        
        # Check if it's an average or a bar
        if len(parts) > 1 and parts[1] == 'Average':
            # Average comes after all bars
            bar_num = 9999
        else:
            # Bar number
            bar_num = int(parts[2]) if len(parts) > 2 else 0
        
        return (dim_num, bar_num)
    
    # Sort handles and labels together
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: sort_key(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_pairs)
    # Create legend inside plot, top left, with padding from border
    '''
    ax.legend(
        sorted_handles, sorted_labels,
        loc='upper left',
        #bbox_to_anchor=(0.03, 0.97),  # More space from left/top border
        borderaxespad=1.5,            # Increase space between legend and axes
        ncol=1,
        fontsize=15,
        framealpha=1.0
    )
    '''
    plt.tight_layout()
    
    # Save as PDF
    if not output_path.endswith('.pdf'):
        output_path = output_path.rsplit('.', 1)[0] + '.pdf'
    
    plt.savefig(output_path, bbox_inches='tight', format='pdf')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize top k-longest persistence bars across checkpoints'
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
        required=True,
        help='Representation name (e.g., final_hidden, input_embeds, layer_0_after_block)'
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
    parser.add_argument(
        '--output',
        type=str,
        help='Output plot path (default: work_dir/persistence_evolution_{representation}.png)'
    )
    
    args = parser.parse_args()
    
    # Determine k values for each dimension
    k_values = {}
    
    # First, try dimension-specific k values
    if args.k_H0:
        k_values['H0'] = args.k_H0
    if args.k_H1:
        k_values['H1'] = args.k_H1
    if args.k_H2:
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
            first_data = load_persistence_data(first_checkpoint_path, args.representation)
            
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
    
    # Load data from each checkpoint
    checkpoint_data = []
    work_path = Path(args.work_dir)
    
    for checkpoint_name, checkpoint_num in checkpoints:
        checkpoint_path = work_path / checkpoint_name
        data = load_persistence_data(checkpoint_path, args.representation)
        
        if data is not None:
            checkpoint_data.append((checkpoint_name, checkpoint_num, data))
            print(f"  Loaded: {checkpoint_name}")
        else:
            print(f"  Skipped: {checkpoint_name} (no data found)")
    
    if not checkpoint_data:
        print(f"Error: No valid checkpoint data found for representation '{args.representation}'", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(work_path / f"persistence_evolution_{args.representation}.pdf")
    
    # Create plot
    plot_persistence_across_checkpoints(
        checkpoint_data,
        args.representation,
        k_values,
        output_path
    )


if __name__ == '__main__':
    main()

