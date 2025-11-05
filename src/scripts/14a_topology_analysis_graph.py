"""
Generate and save persistence diagrams from the original graph distance matrix
This is used for validation - comparing LLM representations against ground truth graph topology
"""
import argparse
import numpy as np
import json
from pathlib import Path

# Try to import ripser for persistent homology
try:
    from ripser import ripser
    from persim import plot_diagrams
    import matplotlib.pyplot as plt
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("Warning: ripser not available. Install with: pip install ripser")


def load_graph_distance_matrix(graph_dir, dataset_name):
    """
    Load the original graph distance matrix
    
    Args:
        graph_dir: directory containing graph files
        dataset_name: dataset name (e.g., "cylinder_x_30x40")
    
    Returns:
        distance_matrix: numpy array of shape [n_samples, n_samples]
    """
    graph_dir = Path(graph_dir)
    
    distance_matrix_file = graph_dir / f"distance_matrix_{dataset_name}.npy"
    if not distance_matrix_file.exists():
        raise FileNotFoundError(f"Graph distance matrix not found: {distance_matrix_file}")
    
    distance_matrix = np.load(distance_matrix_file)
    print(f"Loaded graph distance matrix: {distance_matrix.shape}")
    
    # Handle NaN values (unreachable nodes) - replace with large value
    if np.isnan(distance_matrix).any():
        print(f"  Warning: Found NaN values in distance matrix (unreachable nodes)")
        max_finite = np.nanmax(distance_matrix)
        distance_matrix = np.nan_to_num(distance_matrix, nan=max_finite * 2)
    
    return distance_matrix


def generate_persistence_diagram(distance_matrix, max_dim=2, thresh=None, coeff=47, save_diagrams_path=None):
    """
    Generate and save persistence diagrams (no Betti number calculation)
    
    Args:
        distance_matrix: distance matrix of shape [n_samples, n_samples]
        max_dim: maximum homology dimension (0 for connected components, 1 for loops, etc.)
        thresh: threshold for filtration (None or inf = full filtration)
        coeff: compute homology with coefficients in the prime field Z/pZ for p=coeff (default: 47)
        save_diagrams_path: path to save persistence diagrams plot
    
    Returns:
        dict with persistence_diagrams data (for JSON saving) or None if failed
    """
    if not RIPSER_AVAILABLE:
        print("  Warning: ripser not available. Cannot generate persistence diagrams.")
        return None
    
    if save_diagrams_path is None:
        print("  Warning: No save path provided for persistence diagram")
        return None
    
    try:
        # Compute persistent homology using ripser
        ripser_params = {
            'maxdim': max_dim,
            'coeff': coeff,
            'distance_matrix': True
        }
        
        # Only add thresh if specified (None/inf = full filtration)
        if thresh is not None:
            ripser_params['thresh'] = thresh
        
        print(f"  Computing persistent homology with maxdim={max_dim}, coeff={coeff}")
        if thresh is not None:
            print(f"  Using threshold: {thresh}")
        else:
            print(f"  Using full filtration (no threshold)")
        
        dgms = ripser(distance_matrix, **ripser_params)
        
        # Extract persistence diagrams data (for JSON saving)
        persistence_diagrams = {}
        for dim in range(max_dim + 1):
            dgm = dgms['dgms'][dim]
            # Convert to list and replace inf with -1.0 for JSON serialization
            diagram_list = []
            for persistence_pair in dgm:
                birth, death = persistence_pair
                if not np.isfinite(death):
                    diagram_list.append([float(birth), -1.0])  # Use -1.0 for inf
                else:
                    diagram_list.append([float(birth), float(death)])
            persistence_diagrams[f'H{dim}'] = diagram_list
        
        # Save persistence diagrams plot
        try:
            plt.figure(figsize=(10, 10))
            plot_diagrams(dgms['dgms'], show=False)  # Don't pop up, just save
            plt.savefig(save_diagrams_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved persistence diagram to {Path(save_diagrams_path).name}")
            return {'persistence_diagrams': persistence_diagrams}
        except Exception as e:
            print(f"  Warning: Could not save persistence diagram: {e}")
            return {'persistence_diagrams': persistence_diagrams}  # Return data even if plot fails
        
    except Exception as e:
        print(f"  Warning: Failed to generate persistence diagram: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Generate and save persistence diagrams from graph distance matrix (for validation)'
    )
    parser.add_argument('--graph_dir', type=str, required=True,
                      help='Directory with graph distance matrix')
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='Dataset name (e.g., cylinder_x_30x40)')
    parser.add_argument('--output_dir', type=str, default='./topology_analysis',
                      help='Output directory for persistence diagrams')
    parser.add_argument('--ripser_thresh', type=float, default=None,
                      help='Threshold for ripser filtration (default: None = full filtration)')
    parser.add_argument('--ripser_maxdim', type=int, default=2,
                      help='Maximum homology dimension (default: 2, computes up to H²)')
    parser.add_argument('--ripser_coeff', type=int, default=47,
                      help='Compute homology with coefficients in Z/pZ for p=coeff (default: 47)')
    
    args = parser.parse_args()
    
    if not RIPSER_AVAILABLE:
        parser.error("ripser is required but not available. Install with: pip install ripser")
    
    print("="*60)
    print("Persistence Diagram Generation (Graph Ground Truth)")
    print("="*60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load graph distance matrix
    print(f"\nLoading graph distance matrix...")
    distance_matrix = load_graph_distance_matrix(args.graph_dir, args.dataset_name)
    
    # Use "ground_truth" as the key name
    key = "ground_truth"
    
    print(f"\n{'='*60}")
    print(f"Processing: {key} (Graph Distance Matrix)")
    print(f"{'='*60}")
    print(f"  Distance matrix shape: {distance_matrix.shape}")
    
    # Generate and save persistence diagram
    diagrams_path = output_dir / f'{key}_persistence_diagram.png'
    json_path = output_dir / f'{key}_topology.json'
    
    result = generate_persistence_diagram(
        distance_matrix,
        max_dim=args.ripser_maxdim,
        thresh=args.ripser_thresh,
        coeff=args.ripser_coeff,
        save_diagrams_path=diagrams_path
    )
    
    if result:
        # Save JSON data
        try:
            # Add metadata to result
            result['source'] = 'graph_distance_matrix'
            result['dataset_name'] = args.dataset_name
            result['distance_matrix_shape'] = list(distance_matrix.shape)
            
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved persistence diagram data to {Path(json_path).name}")
        except Exception as e:
            print(f"  Warning: Could not save persistence diagram data to JSON: {e}")
        
        print(f"  ✓ Successfully generated persistence diagram and data")
    else:
        print(f"  ✗ Failed to generate persistence diagram")
    
    print(f"\n{'='*60}")
    print("Persistence Diagram Generation Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()

