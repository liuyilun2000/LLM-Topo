#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01b_sequence_generation.py - Generate random walk sequences from saved graph representation

This script generates random walk sequences on a pre-computed graph structure.
The graph representation should be generated first using 01a_graph_generation.py.

Input files (from 01a_graph_generation.py):
  - A_{topology}_{H}x{W}_labeled.csv : Adjacency matrix with node labels
  - nodes_{topology}_{H}x{W}.csv : Node information
  - coords_{topology}_{H}x{W}.csv : Coordinates (optional, for visualization)

Output files:
  - walks_{topology}_{H}x{W}.csv : Random walk sequences
  - visit_counts_{topology}_{H}x{W}.csv : Node visit statistics
"""

import argparse
import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_graph(graph_dir: str, topology: str, H: int, W: int):
    """
    Load graph representation from saved files.
    
    Args:
        graph_dir: Directory containing graph files
        topology: Topology name
        H: Grid height
        W: Grid width
        
    Returns:
        A: Adjacency matrix (N x N)
        labels: Node labels (list of strings)
        nodes_df: Node information DataFrame
        coords: Coordinates (N x 3) or None
    """
    base = f"{topology}_{H}x{W}"
    
    # Load adjacency matrix
    A_labeled_path = os.path.join(graph_dir, f"A_{base}_labeled.csv")
    if not os.path.exists(A_labeled_path):
        raise FileNotFoundError(f"Graph file not found: {A_labeled_path}\n"
                               f"Please run 01a_graph_generation.py first to create the graph representation.")
    
    dfA = pd.read_csv(A_labeled_path, index_col=0)
    labels = dfA.index.astype(str).tolist()
    A = dfA.values.astype(np.int8)
    
    # Load nodes information
    nodes_path = os.path.join(graph_dir, f"nodes_{base}.csv")
    if os.path.exists(nodes_path):
        nodes_df = pd.read_csv(nodes_path)
    else:
        # Fallback: create from labels
        nodes_df = pd.DataFrame({"node_id": labels})
    
    # Load coordinates (optional)
    coords_path = os.path.join(graph_dir, f"coords_{base}.csv")
    if os.path.exists(coords_path):
        coords = np.loadtxt(coords_path, delimiter=",")
    else:
        coords = None
    
    return A, labels, nodes_df, coords


def build_neighbors_from_A(A: np.ndarray):
    """Build neighbor list from adjacency matrix."""
    return [np.where(A[i] == 1)[0].astype(np.int64) for i in range(A.shape[0])]


def deficit_weights(visits: np.ndarray, target: float, temperature: float):
    """
    Compute sampling weights based on visit deficit.
    Nodes with fewer visits get higher weights.
    """
    deficits = np.maximum(0, target - visits.astype(float))
    w = np.power(deficits + 1e-8, 1.0 / temperature)
    s = w.sum()
    return (w / s) if s > 0 else (np.ones_like(w) / len(w))


def choose_start_index(rng, visits: np.ndarray, target: float, temperature: float):
    """Choose start index based on visit deficit."""
    p = deficit_weights(visits, target, temperature)
    return int(rng.choice(len(visits), p=p))


def random_walk_with_restart_balanced(neighbors, start_idx: int, max_len: int,
                                      restart_prob: float,
                                      visits: np.ndarray, target: float, temp: float,
                                      rng: np.random.Generator,
                                      no_repeat_window: int = 0):
    """
    Generate a single random walk with restart and balanced sampling.
    
    Args:
        neighbors: List of neighbor arrays for each node
        start_idx: Starting node index
        max_len: Maximum walk length
        restart_prob: Probability of restarting at each step
        visits: Array of visit counts for each node
        target: Target visit count per node
        temp: Temperature for deficit weighting
        rng: Random number generator
        no_repeat_window: Avoid revisiting nodes in last N steps (0 to disable)
        
    Returns:
        path: List of node indices visited
    """
    path = [start_idx]
    cur = start_idx
    
    # Short-term memory queue to avoid small loops
    recent_visited = []
    if no_repeat_window > 0:
        recent_visited.append(start_idx)
    
    for _ in range(max_len - 1):
        if rng.random() < restart_prob:
            # Restart: choose based on deficit
            p = deficit_weights(visits.astype(float), target, temp)
            cur = int(rng.choice(len(visits), p=p))
            path.append(cur)
            # Update short-term memory
            if no_repeat_window > 0:
                recent_visited.append(cur)
                if len(recent_visited) > no_repeat_window:
                    recent_visited.pop(0)
            continue
        
        neigh = neighbors[cur]
        if len(neigh) == 0:
            # Isolated node: force jump
            p = deficit_weights(visits.astype(float), target, temp)
            cur = int(rng.choice(len(visits), p=p))
        else:
            # Apply short-term memory filtering
            if no_repeat_window > 0 and len(recent_visited) > 0:
                # Filter neighbors not in short-term memory
                recent_set = set(recent_visited)
                valid_neigh = [n for n in neigh if n not in recent_set]
                
                # If valid neighbors available, use them; otherwise allow revisiting
                if len(valid_neigh) > 0:
                    cur = int(valid_neigh[rng.integers(len(valid_neigh))])
                else:
                    # All neighbors in memory, allow revisiting (avoid deadlock)
                    cur = int(neigh[rng.integers(len(neigh))])
            else:
                # No memory restriction: uniform random
                cur = int(neigh[rng.integers(len(neigh))])
        
        path.append(cur)
        
        # Update short-term memory queue (fixed window FIFO)
        if no_repeat_window > 0:
            recent_visited.append(cur)
            if len(recent_visited) > no_repeat_window:
                recent_visited.pop(0)
    
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Generate random walk sequences from saved graph representation"
    )
    
    # Graph loading
    ap.add_argument("--graph_dir", type=str, default="./data/graphs",
                    help="Directory containing graph files (from 01a_graph_generation.py)")
    ap.add_argument("--topology", type=str, required=True,
                    help="Topology name (must match graph files)")
    ap.add_argument("--H", type=int, required=True, help="Grid height")
    ap.add_argument("--W", type=int, required=True, help="Grid width")
    
    # Walk generation parameters
    ap.add_argument("--max_length", type=int, default=128,
                    help="Maximum walk length")
    ap.add_argument("--max_seqs", type=int, default=120000,
                    help="Maximum number of sequences")
    ap.add_argument("--min_visits_per_node", type=int, default=10000000000,
                    help="Minimum visits per node before stopping (default: very large, effectively disabled)")
    ap.add_argument("--restart_prob", type=float, default=0.0,
                    help="Restart probability (0.0 = no restart)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Temperature for deficit sampling (lower = more aggressive)")
    ap.add_argument("--no_repeat_window", type=int, default=32,
                    help="Avoid revisiting nodes in last N steps (0 to disable)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV file (default: auto-generated from topology)")
    ap.add_argument("--counts_out", type=str, default=None,
                    help="Visit counts output CSV (default: auto-generated)")
    ap.add_argument("--flush_every", type=int, default=2000,
                    help="Flush output file every N sequences")
    
    args = ap.parse_args()
    
    # Auto-generate output paths if not provided
    if args.out is None:
        args.out = f"./data/walks_{args.topology}_{args.H}x{args.W}.csv"
    if args.counts_out is None:
        args.counts_out = f"./data/visit_counts_{args.topology}_{args.H}x{args.W}.csv"
    
    print("=" * 60)
    print("Random Walk Generation")
    print("=" * 60)
    print(f"Graph directory: {args.graph_dir}")
    print(f"Topology: {args.topology}")
    print(f"Grid size: {args.H}x{args.W}")
    print(f"Output: {args.out}")
    print(f"Counts output: {args.counts_out}")
    print()
    
    # Load graph
    print("Loading graph representation...")
    try:
        A, labels, nodes_df, coords = load_graph(args.graph_dir, args.topology, args.H, args.W)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    N = A.shape[0]
    neighbors = build_neighbors_from_A(A)
    visits = np.zeros(N, dtype=np.int64)
    total_steps = 0
    
    print(f"Graph loaded: {N} nodes, {A.sum() // 2} edges")
    print()
    
    # Initialize output file
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, 'w', newline='', encoding='utf-8')
    writer = csv.writer(fout)
    writer.writerow(["walk_id", "length", "sequence_labels"])
    
    # Generate walks
    rng = np.random.default_rng(args.seed)
    seq_count = 0
    
    pbar = tqdm(total=args.max_seqs, desc="Generating walks", unit="seq")
    
    try:
        while seq_count < args.max_seqs:
            # Compute target visits per node
            target = (total_steps / max(1, N)) if total_steps > 0 else 0
            
            # Choose start node based on deficit
            start = choose_start_index(rng, visits.astype(float), target, args.temperature)
            
            # Generate walk
            path = random_walk_with_restart_balanced(
                neighbors, start, args.max_length,
                args.restart_prob, visits.astype(float), target, args.temperature, rng,
                no_repeat_window=args.no_repeat_window
            )
            
            # Update visit counts
            for idx in path:
                visits[idx] += 1
            total_steps += len(path)
            
            # Write walk to file
            sequence_str = " ".join(labels[i] for i in path)
            writer.writerow([seq_count, len(path), sequence_str])
            
            if seq_count % args.flush_every == 0:
                fout.flush()
            
            seq_count += 1
            pbar.update(1)
            pbar.set_postfix({
                'visits': f"{int(visits.min())}/{int(visits.mean()):.0f}/{int(visits.max())}",
                'steps': total_steps
            })
            
            # Check stopping condition
            if args.min_visits_per_node > 0:
                if visits.min() >= args.min_visits_per_node:
                    print(f"\nStopping: all nodes have at least {args.min_visits_per_node} visits")
                    break
    
    finally:
        pbar.close()
        fout.close()
    
    # Save visit counts
    os.makedirs(os.path.dirname(args.counts_out) or ".", exist_ok=True)
    counts_df = pd.DataFrame({
        "node_id": labels,
        "visit_count": visits,
        "visit_fraction": visits / max(1, visits.sum())
    })
    counts_df.to_csv(args.counts_out, index=False)
    
    print()
    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)
    print(f"Sequences generated: {seq_count}")
    print(f"Total steps: {total_steps}")
    print(f"Visit statistics:")
    print(f"  Min: {visits.min()}, Mean: {visits.mean():.1f}, Max: {visits.max()}")
    print(f"Output files:")
    print(f"  Walks: {args.out}")
    print(f"  Visit counts: {args.counts_out}")
    
    return 0


if __name__ == "__main__":
    exit(main())
