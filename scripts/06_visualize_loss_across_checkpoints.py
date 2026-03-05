#!/usr/bin/env python3
"""
Visualize Training and Evaluation Loss (Full Log History)

This script loads trainer_state.json from the checkpoint that has the longest
log_history (typically the latest checkpoint) and plots all available points:
- Train loss at every logging step
- Eval loss at every eval step
- Learning rate at every step
"""

import json
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
    Returns None for final_model.
    """
    match = re.search(r'checkpoint-(\d+)', checkpoint_dir)
    if match:
        return int(match.group(1))
    return None


def find_checkpoints(work_dir: str) -> List[Tuple[str, Optional[int]]]:
    """Find all checkpoint directories. Returns (dir_name, step_or_None) sorted by step."""
    work_path = Path(work_dir)
    if not work_path.exists():
        raise ValueError(f"Work directory does not exist: {work_dir}")
    checkpoints = []
    for item in work_path.iterdir():
        if item.is_dir():
            num = parse_checkpoint_number(item.name)
            checkpoints.append((item.name, num))
    checkpoints.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))
    return checkpoints


def load_trainer_state(checkpoint_path: Path) -> Optional[Dict]:
    """Load trainer_state.json from a checkpoint directory."""
    json_file = checkpoint_path / "trainer_state.json"
    if not json_file.exists():
        return None
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {json_file}: {e}", file=sys.stderr)
        return None


def extract_full_log_history(data: Dict) -> Tuple[
    List[float], List[float],  # train steps, train losses
    List[float], List[float],  # eval steps, eval losses
    List[float], List[float],  # lr steps, lr values
]:
    """Extract all train loss, eval loss, and learning rate from log_history.
    Returns (train_steps, train_losses), (eval_steps, eval_losses), (lr_steps, lr_values).
    """
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    lr_steps, lr_values = [], []

    if 'log_history' not in data:
        return (train_steps, train_losses), (eval_steps, eval_losses), (lr_steps, lr_values)

    for entry in data['log_history']:
        step = entry.get('step')
        if step is None:
            continue
        if 'loss' in entry and 'eval_loss' not in entry:
            train_steps.append(float(step))
            train_losses.append(float(entry['loss']))
        if 'eval_loss' in entry:
            eval_steps.append(float(step))
            eval_losses.append(float(entry['eval_loss']))
        if 'learning_rate' in entry:
            lr_steps.append(float(step))
            lr_values.append(float(entry['learning_rate']))

    return (train_steps, train_losses), (eval_steps, eval_losses), (lr_steps, lr_values)


def plot_full_history(
    train_steps: List[float], train_losses: List[float],
    eval_steps: List[float], eval_losses: List[float],
    lr_steps: List[float], lr_values: List[float],
    output_path: str,
):
    """Plot all train loss, eval loss, and learning rate points."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    train_color = '#3562E8'
    eval_color = '#DC2626'
    lr_color = '#059669'

    # Train loss: line only (many points), no markers
    if train_steps and train_losses:
        ax.plot(train_steps, train_losses, color=train_color, linestyle='-', linewidth=1.0, label='Train Loss', alpha=0.9)
    if eval_steps and eval_losses:
        ax.plot(eval_steps, eval_losses, color=eval_color, linestyle='-', linewidth=1.2, label='Eval Loss', alpha=0.9)

    ax2 = ax.twinx()
    if lr_steps and lr_values:
        ax2.plot(lr_steps, lr_values, color=lr_color, linestyle='--', linewidth=1.0, label='Learning Rate', alpha=0.9)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12, color=lr_color)
    ax2.tick_params(axis='y', labelcolor=lr_color)
    ax2.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0e}'))
    if lr_values:
        ax2.set_ylim(min(lr_values), max(lr_values) * 1.05 if max(lr_values) > min(lr_values) else max(lr_values) * 1.2)

    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=10)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=11, framealpha=1.0)

    if not output_path.endswith('.pdf'):
        output_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', format='pdf')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training and evaluation loss (full log history)'
    )
    parser.add_argument('--work-dir', type=str, required=True,
                        help='Work directory (results/DATASET_NAME/RUN_NAME)')
    parser.add_argument('--output', type=str, help='Output plot path (default: work_dir/loss_evolution.pdf)')
    args = parser.parse_args()

    work_path = Path(args.work_dir)
    checkpoints = find_checkpoints(args.work_dir)
    if not checkpoints:
        print(f"Error: No checkpoints found in {args.work_dir}", file=sys.stderr)
        sys.exit(1)

    # Load all checkpoint trainer_states and pick the one with longest log_history (full run)
    best_data = None
    best_len = 0
    best_name = None
    for name, num in checkpoints:
        path = work_path / name
        data = load_trainer_state(path)
        if data is None:
            continue
        lh = data.get('log_history', [])
        if len(lh) > best_len:
            best_len = len(lh)
            best_data = data
            best_name = name
    if best_data is None:
        print("Error: No trainer_state.json found in any checkpoint", file=sys.stderr)
        sys.exit(1)

    print(f"Using full log history from: {best_name} ({best_len} log entries)")

    (train_steps, train_losses), (eval_steps, eval_losses), (lr_steps, lr_values) = extract_full_log_history(best_data)
    print(f"  Train points: {len(train_steps)}, Eval points: {len(eval_steps)}, LR points: {len(lr_steps)}")

    output_path = args.output if args.output else str(work_path / "loss_evolution.pdf")
    plot_full_history(
        train_steps, train_losses,
        eval_steps, eval_losses,
        lr_steps, lr_values,
        output_path,
    )


if __name__ == '__main__':
    main()

