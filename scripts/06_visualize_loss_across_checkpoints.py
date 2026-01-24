#!/usr/bin/env python3
"""
Visualize Training and Evaluation Loss Across Checkpoints

This script loads trainer_state.json from multiple checkpoints
and visualizes how training loss and evaluation loss vary across training checkpoints.
"""

import json
import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def load_trainer_state(checkpoint_path: Path) -> Optional[Dict]:
    """Load trainer_state.json for a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
    
    Returns:
        Dictionary with trainer state data or None if file doesn't exist
    """
    json_file = checkpoint_path / "trainer_state.json"
    
    if not json_file.exists():
        return None
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Failed to load {json_file}: {e}", file=sys.stderr)
        return None


def extract_loss_data(data: Dict, checkpoint_step: Optional[int] = None) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Extract training loss, evaluation loss, and learning rate from trainer state.
    
    Args:
        data: Trainer state dictionary
        checkpoint_step: Step number of the checkpoint (to find matching learning rate)
    
    Returns:
        Tuple of (last_train_loss, last_eval_loss, best_metric, learning_rate)
        - last_train_loss: Last training loss from log_history
        - last_eval_loss: Last evaluation loss from log_history
        - best_metric: Best metric (best eval_loss) from the checkpoint
        - learning_rate: Learning rate at or before checkpoint_step, or first available
    """
    last_train_loss = None
    last_eval_loss = None
    best_metric = None
    learning_rate = None
    
    if 'log_history' in data:
        # Find the last training loss, last eval loss
        # For learning rate, find the one at or before checkpoint_step
        first_lr = None
        best_lr = None
        best_step = -1
        
        for entry in data['log_history']:
            if 'loss' in entry and 'eval_loss' not in entry:
                # Training loss (not eval)
                last_train_loss = entry['loss']
            elif 'eval_loss' in entry:
                # Evaluation loss
                last_eval_loss = entry['eval_loss']
            
            if 'learning_rate' in entry:
                # Store first learning rate encountered
                if first_lr is None:
                    first_lr = entry['learning_rate']
                
                # If we have a checkpoint step, find the learning rate at or before that step
                if checkpoint_step is not None and 'step' in entry:
                    entry_step = entry['step']
                    # Prefer exact match, then closest before, then closest after
                    if entry_step == checkpoint_step:
                        best_lr = entry['learning_rate']
                        best_step = entry_step
                        break  # Exact match found, use it
                    elif entry_step <= checkpoint_step and entry_step > best_step:
                        # This is the closest step before or at checkpoint_step
                        best_lr = entry['learning_rate']
                        best_step = entry_step
                    elif best_lr is None and entry_step > checkpoint_step:
                        # No exact or before match found yet, use first after
                        best_lr = entry['learning_rate']
                        best_step = entry_step
        
        # Use best_lr if found, otherwise use first
        if best_lr is not None:
            learning_rate = best_lr
        elif first_lr is not None:
            learning_rate = first_lr
    
    if 'best_metric' in data:
        best_metric = data['best_metric']
    
    return last_train_loss, last_eval_loss, best_metric, learning_rate


def plot_loss_across_checkpoints(
    checkpoint_data: List[Tuple[str, Optional[int], Dict]],
    output_path: str
):
    """Plot training and evaluation loss across checkpoints.
    
    Args:
        checkpoint_data: List of (checkpoint_name, checkpoint_num, data_dict) tuples
        output_path: Path to save the plot
    """
    if not checkpoint_data:
        print("Error: No checkpoint data available", file=sys.stderr)
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
            checkpoint_labels.append(str(checkpoint_num))
        else:
            # For final_model, use a value slightly larger than max checkpoint
            final_x = max_checkpoint_num * 1.1 if max_checkpoint_num > 0 else 1000
            checkpoint_x_coords.append(final_x)
            checkpoint_labels.append("final")
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Colors: blue for train loss, red for eval loss, green for learning rate
    train_color = '#3562E8'  # Blue
    eval_color = '#DC2626'   # Red
    lr_color = '#059669'     # Green
    
    # Collect loss values and learning rate at each checkpoint
    train_losses = []
    eval_losses = []
    best_metrics = []
    learning_rates = []
    checkpoint_nums = []
    
    for checkpoint_name, checkpoint_num, data in checkpoint_data:
        last_train_loss, last_eval_loss, best_metric, learning_rate = extract_loss_data(data, checkpoint_num)
        
        # Determine x-coordinate
        if checkpoint_num is not None:
            x_coord = checkpoint_num
        else:
            x_coord = max_checkpoint_num * 1.1 if max_checkpoint_num > 0 else 1000
        
        checkpoint_nums.append(x_coord)
        
        if last_train_loss is not None:
            train_losses.append(last_train_loss)
        else:
            train_losses.append(None)
        
        if last_eval_loss is not None:
            eval_losses.append(last_eval_loss)
        else:
            eval_losses.append(None)
        
        if best_metric is not None:
            best_metrics.append(best_metric)
        else:
            best_metrics.append(None)
        
        if learning_rate is not None:
            learning_rates.append(learning_rate)
        else:
            learning_rates.append(None)
    
    # Filter out None values for plotting
    train_x = []
    train_y = []
    eval_x = []
    eval_y = []
    best_x = []
    best_y = []
    lr_x = []
    lr_y = []
    
    for i, (x, train, eval_val, best, lr) in enumerate(zip(checkpoint_nums, train_losses, eval_losses, best_metrics, learning_rates)):
        if train is not None:
            train_x.append(x)
            train_y.append(train)
        if eval_val is not None:
            eval_x.append(x)
            eval_y.append(eval_val)
        if best is not None:
            best_x.append(x)
            best_y.append(best)
        if lr is not None:
            lr_x.append(x)
            lr_y.append(lr)
    
    # Create secondary y-axis for learning rate
    ax2 = ax.twinx()
    
    # Plot training loss
    if train_x:
        ax.plot(train_x, train_y,
               marker='o', markersize=3, alpha=1.0,
               color=train_color, linestyle='-',
               linewidth=1.5,
               label='Train Loss')
    
    # Plot evaluation loss
    if eval_x:
        ax.plot(best_x, best_y,
               marker='s', markersize=4, alpha=1.0,
               color=eval_color, linestyle='-',
               linewidth=1.5,
               label='Eval Loss')
    
    # Plot learning rate on secondary y-axis
    if lr_x:
        ax2.plot(lr_x, lr_y,
                marker='^', markersize=0, alpha=1.0,
                color=lr_color, linestyle='--',
                linewidth=1.5,
                label='Learning Rate')
    
    # Plot best metric (best_metric from each checkpoint)
    #if best_x:
    #    ax.plot(best_x, best_y,
    #           marker='', alpha=0.5,
    #           color=eval_color, linestyle='--',
    #           linewidth=1.0,
    #           label='Best Eval Loss')
    
    # Set only x-axis to logarithmic scale (y-axes are linear)
    ax.set_xscale('log')
    # y-axes are linear (no log scale)
    
    ax.set_xlabel('Step', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15)
    ax2.set_ylabel('Learning Rate', fontsize=15, labelpad=-18)
    ax2.tick_params(axis='y')

    ax2.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f'{x:.0e}'))

    # Only show min and max on ax2 y-axis
    if lr_y:  # Only try if there is lr data
        min_lr = min(lr_y)
        max_lr = max(lr_y)
        if min_lr != max_lr:
            ax2.set_yticks([min_lr, max_lr])
            ax2.set_yticklabels([f"{min_lr:.0e}", f"{max_lr:.0e}"])
        else:
            # Only one value, show it
            ax2.set_yticks([min_lr])
            ax2.set_yticklabels([f"{min_lr:.0e}"])

    # Enable grid (both horizontal and vertical)
    ax.grid(True, alpha=0.3)
    
    # Make tick labels larger
    ax.tick_params(axis='both', labelsize=12)
    
    # Create legend - combine handles from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='lower left', fontsize=15, framealpha=1.0)
    
    plt.tight_layout()
    
    # Save as PDF
    if not output_path.endswith('.pdf'):
        output_path = output_path.rsplit('.', 1)[0] + '.pdf'
    
    plt.savefig(output_path, bbox_inches='tight', format='pdf')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training and evaluation loss across checkpoints'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        required=True,
        help='Work directory (results/DATASET_NAME/RUN_NAME)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output plot path (default: work_dir/loss_evolution.pdf)'
    )
    
    args = parser.parse_args()
    
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
        data = load_trainer_state(checkpoint_path)
        
        if data is not None:
            checkpoint_data.append((checkpoint_name, checkpoint_num, data))
            print(f"  Loaded: {checkpoint_name}")
        else:
            print(f"  Skipped: {checkpoint_name} (no trainer_state.json found)")
    
    if not checkpoint_data:
        print(f"Error: No valid checkpoint data found", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(work_path / "loss_evolution.pdf")
    
    # Create plot
    plot_loss_across_checkpoints(
        checkpoint_data,
        output_path
    )


if __name__ == '__main__':
    main()

