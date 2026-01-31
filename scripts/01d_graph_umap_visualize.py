"""
UMAP on graph distance/adjacency matrix - visualization and/or analysis.
Like 03c_umap_analysis.py for model representations, this handles both:
  - Visualization mode: 3D UMAP + trajectory overlay (HTML)
  - Analysis mode: save embeddings (e.g. 6D) for topology analysis
"""
import argparse
import json
import numpy as np
from pathlib import Path

from graph_utils import load_adjacency_matrix, load_dataset_matrix
from data_loading_utils import load_trajectory
from visualization_utils import apply_umap, create_interactive_3d_html


def _fix_infinite_distances(distance_matrix):
    """Replace infinite distances with large finite value."""
    num_inf = np.sum(np.isinf(distance_matrix))
    if num_inf > 0:
        finite_vals = distance_matrix[np.isfinite(distance_matrix)]
        max_finite = np.max(finite_vals) if len(finite_vals) > 0 else distance_matrix.shape[0] * 2
        max_finite = max(max_finite + 1, distance_matrix.shape[0] * 2)
        distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, max_finite)
        np.fill_diagonal(distance_matrix, 0.0)
        print(f"  Fixed {num_inf} infinite distances (replaced with {max_finite:.2f})")
    return distance_matrix


def main():
    parser = argparse.ArgumentParser(
        description='UMAP on graph matrix: visualization (3D+trajectory) and/or analysis (save embeddings)'
    )
    parser.add_argument('--graph_dir', type=str, required=True,
                      help='Directory with graph files (distance/adjacency matrices)')
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='Dataset name (e.g., torus_abABcdCD_N1000_iter200)')
    parser.add_argument('--matrix_type', type=str, default='auto',
                      choices=['auto', 'distance', 'adjacency'],
                      help='Matrix type: auto (prefer distance), distance, or adjacency')
    parser.add_argument('--output_dir', type=str, default='./graph_umap_visualize',
                      help='Output directory')
    parser.add_argument('--umap_n_components', type=int, default=3,
                      help='UMAP target dimensions (default: 3 for viz, use 6 for topology)')
    parser.add_argument('--umap_min_dist', type=float, default=0.2,
                      help='UMAP min_dist parameter')
    parser.add_argument('--umap_n_neighbors', type=int, default=200,
                      help='UMAP n_neighbors parameter')
    parser.add_argument('--umap_random_state', type=int, default=42,
                      help='Random seed for UMAP (default: 42)')
    parser.add_argument('--save_umap_result', action='store_true',
                      help='Save UMAP embeddings as .npz for topology analysis')
    parser.add_argument('--generate_visualizations', action='store_true',
                      help='Generate 3D HTML visualization with trajectory overlay')
    # Trajectory args (required when generate_visualizations)
    parser.add_argument('--walks_csv', type=str, default=None,
                      help='Path to walks CSV (required for --generate_visualizations)')
    parser.add_argument('--walk_id', type=int, default=None,
                      help='Walk ID to use')
    parser.add_argument('--trajectory_idx', type=int, default=128,
                      help='Trajectory index if walk_id not set (default: 128)')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Max points in trajectory for visualization (default: 128)')

    args = parser.parse_args()

    if args.generate_visualizations and not args.walks_csv:
        parser.error("--walks_csv is required when using --generate_visualizations")

    if args.generate_visualizations and args.umap_n_components not in [2, 3]:
        print("Warning: Visualization requires 2D or 3D. Setting umap_n_components=3")
        args.umap_n_components = 3

    do_viz = args.generate_visualizations
    do_save = args.save_umap_result
    if not do_viz and not do_save:
        parser.error("At least one of --save_umap_result or --generate_visualizations must be set")

    print("="*60)
    if do_viz and do_save:
        print("Graph UMAP: Visualization + Save embeddings")
    elif do_viz:
        print("Graph UMAP Visualization (3D + trajectory)")
    else:
        print("Graph UMAP Analysis (save embeddings for topology)")
    print("="*60)

    # Load graph matrix
    print(f"\nLoading graph matrix (type: {args.matrix_type})...")
    distance_matrix, matrix_type_used = load_dataset_matrix(
        args.graph_dir, args.dataset_name, args.matrix_type
    )
    print(f"  Matrix type used: {matrix_type_used}")
    print(f"  Shape: {distance_matrix.shape}")
    distance_matrix = _fix_infinite_distances(distance_matrix)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key = f"graph_{matrix_type_used}"

    # Apply UMAP
    print(f"\nApplying UMAP ({args.umap_n_components}D)...")
    umap_data, umap_info = apply_umap(
        distance_matrix=distance_matrix,
        n_components=args.umap_n_components,
        min_dist=args.umap_min_dist,
        n_neighbors=args.umap_n_neighbors,
        metric='precomputed',
        random_state=args.umap_random_state
    )

    # Save UMAP result (for topology analysis)
    if do_save:
        npz_file = output_dir / f'{key}_umap_{args.umap_n_components}d.npz'
        np.savez(npz_file,
                 umap_reduced=umap_data,
                 n_points=umap_data.shape[0],
                 n_dims=umap_data.shape[1])
        print(f"\nSaved: {npz_file.name}")
        print(f"  Shape: {umap_data.shape[0]} points × {umap_data.shape[1]} dimensions")

        info_dict = {
            'dataset_name': args.dataset_name,
            'matrix_type': matrix_type_used,
            'n_points': int(umap_data.shape[0]),
            'n_dims': int(umap_data.shape[1]),
            'umap_params': {
                'n_components': args.umap_n_components,
                'min_dist': args.umap_min_dist,
                'n_neighbors': args.umap_n_neighbors,
                'metric': 'precomputed',
                'random_state': args.umap_random_state
            }
        }
        info_file = output_dir / f'{key}_umap_{args.umap_n_components}d_info.json'
        with open(info_file, 'w') as f:
            json.dump(info_dict, f, indent=2)
        print(f"  Info: {info_file.name}")

    # Generate visualization with trajectory overlay
    if do_viz:
        print(f"\nLoading trajectory from: {args.walks_csv}")
        trajectory, walk_id = load_trajectory(
            args.walks_csv,
            walk_id=args.walk_id,
            trajectory_idx=args.trajectory_idx
        )
        print(f"  Walk ID: {walk_id}")
        limited = trajectory[:args.max_length] if len(trajectory) > args.max_length else trajectory
        trajectory_positions = np.arange(len(limited))

        try:
            trajectory_indices = np.array(limited, dtype=int)
            if trajectory_indices.max() >= len(umap_data):
                valid_mask = trajectory_indices < len(umap_data)
                trajectory_indices = trajectory_indices[valid_mask]
                trajectory_positions = trajectory_positions[valid_mask]
        except Exception as e:
            print(f"  Error mapping trajectory: {e}")
            trajectory_indices = np.array([], dtype=int)
            trajectory_positions = np.array([])

        adjacency_matrix = None
        try:
            adjacency_matrix = load_adjacency_matrix(args.graph_dir, args.dataset_name)
        except Exception:
            pass

        output_base = f"graph_{matrix_type_used}_walk_{walk_id}_umap_3d"
        html_file = output_dir / f"{output_base}.html"
        title = f"Graph Matrix ({matrix_type_used})\nWalk {walk_id} ({len(trajectory_indices)} nodes)"
        create_interactive_3d_html(
            umap_data=umap_data,
            trajectory_indices=trajectory_indices,
            trajectory_positions=trajectory_positions,
            adjacency_matrix=adjacency_matrix,
            token_ids=None,
            output_file=str(html_file),
            title=title
        )

        metadata = {
            'dataset_name': args.dataset_name,
            'matrix_type': matrix_type_used,
            'walk_id': int(walk_id),
            'umap_shape': list(umap_data.shape),
            'umap_params': {'n_components': args.umap_n_components, 'min_dist': args.umap_min_dist,
                          'n_neighbors': args.umap_n_neighbors, 'random_state': args.umap_random_state},
        }
        metadata_file = output_dir / f"{output_base}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved HTML: {html_file.name}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    if do_save:
        print(f"Next step (topology): ./01f_graph_topology_analysis.sh")


if __name__ == '__main__':
    main()
