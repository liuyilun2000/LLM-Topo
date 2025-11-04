"""
Generate and save persistence diagrams from data representations or distance matrices
No topology analysis or Betti number calculation - only persistence diagram generation
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



def load_data_representation(data_dir, key, data_type='auto'):
    """
    Load data representation (PCA, UMAP, or downsampled PCA) for persistence diagram generation
    
    Args:
        data_dir: directory containing results
        key: representation name
        data_type: 'pca', 'umap', 'downsampled', or 'auto' (tries all)
    
    Returns:
        data: numpy array of embeddings
        selected_indices: indices of selected points (if downsampled) or None
        source_type: string indicating the source type ('pca', 'umap', 'downsampled')
    """
    data_dir = Path(data_dir)
    
    # Priority order for auto-detection:
    # 1. UMAP results (new format: {key}_umap_{n}d.npz)
    # 2. Downsampled PCA ({key}_pca_downsampled.npz)
    # 3. PCA ({key}_pca.npz)
    # 4. Legacy formats ({key}_reduced_{n}d.npz)
    
    if data_type == 'downsampled':
        # Load downsampled PCA data
        npz_file = data_dir / f'{key}_pca_downsampled.npz'
        if not npz_file.exists():
            raise FileNotFoundError(f"Downsampled PCA results not found for {key}")
        data = np.load(npz_file)
        embeddings = data['pca_reduced']  # Keep key name for compatibility
        selected_indices = data.get('selected_indices', None)
        return embeddings, selected_indices, 'downsampled'
    
    # Try to find UMAP results (pattern: {key}_umap_{n}d.npz)
    if data_type in ['auto', 'umap']:
        umap_files = list(data_dir.glob(f'{key}_umap_*d.npz'))
        if umap_files:
            # Use the first found (could have multiple dims, user should specify)
            npz_file = sorted(umap_files)[0]  # Sort to get consistent ordering
            data = np.load(npz_file)
            if 'umap_reduced' in data:
                embeddings = data['umap_reduced']  # Keep key name for compatibility
                selected_indices = None  # UMAP results don't have selected_indices
                return embeddings, selected_indices, 'umap'
    
    # Try PCA
    if data_type in ['auto', 'pca']:
        npz_file_pca = data_dir / f'{key}_pca.npz'
        if npz_file_pca.exists():
            data = np.load(npz_file_pca)
            if 'pca_reduced' in data:
                embeddings = data['pca_reduced']  # Keep key name for compatibility
                selected_indices = data.get('selected_indices', None)
                return embeddings, selected_indices, 'pca'
    
    # Try legacy formats (old naming: {key}_reduced_{n}d.npz)
    if data_type == 'auto':
        npz_file_2d = data_dir / f'{key}_reduced_2d.npz'
        npz_file_3d = data_dir / f'{key}_reduced_3d.npz'
        npz_file = None
        if npz_file_2d.exists():
            npz_file = npz_file_2d
        elif npz_file_3d.exists():
            npz_file = npz_file_3d
        
        if npz_file:
            data = np.load(npz_file)
            if 'reduced' in data:
                embeddings = data['reduced']
                selected_indices = data.get('selected_indices', None)
                return embeddings, selected_indices, 'legacy'
    
    return None, None, None


def load_fuzzy_distance_matrix(fuzzy_dir, key):
    """
    Load fuzzy neighborhood distance matrix
    
    Args:
        fuzzy_dir: directory containing fuzzy distance matrices
        key: representation name
    
    Returns:
        distance_matrix: numpy array of shape [n_samples, n_samples]
        pca_data: original PCA data (for compatibility)
    """
    fuzzy_dir = Path(fuzzy_dir)
    
    npz_file = fuzzy_dir / f'{key}_fuzzy_dist.npz'
    if not npz_file.exists():
        raise FileNotFoundError(f"Fuzzy distance matrix not found for {key}")
    
    data = np.load(npz_file)
    distance_matrix = data['distance_matrix']
    pca_data = data.get('pca_reduced', None)
    
    return distance_matrix, pca_data


def generate_persistence_diagram(data, max_dim=2, thresh=None, coeff=47, save_diagrams_path=None, distance_matrix=None):
    """
    Generate and save persistence diagrams (no Betti number calculation)
    
    Args:
        data: numpy array [n_samples, dim] or None if distance_matrix provided
        max_dim: maximum homology dimension (0 for connected components, 1 for loops, etc.)
        thresh: threshold for filtration (None or inf = full filtration)
        coeff: compute homology with coefficients in the prime field Z/pZ for p=coeff (default: 47)
        save_diagrams_path: path to save persistence diagrams plot
        distance_matrix: optional distance matrix of shape [n_samples, n_samples]
    
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
            'coeff': coeff
        }
        
        # Only add thresh if specified (None/inf = full filtration)
        if thresh is not None:
            ripser_params['thresh'] = thresh
        
        # Use distance matrix if provided, otherwise compute from points
        if distance_matrix is not None:
            ripser_params['distance_matrix'] = True
            dgms = ripser(distance_matrix, **ripser_params)
        else:
            ripser_params['metric'] = 'euclidean'
            dgms = ripser(data, **ripser_params)
        
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
        return None


def generate_persistence_diagrams_only(data, save_diagrams_path, save_json_path=None, ripser_thresh=None, ripser_maxdim=2, ripser_coeff=47, distance_matrix=None):
    """
    Generate and save persistence diagrams only (no analysis, no Betti numbers)
    Also saves persistence diagram data to JSON for barcode plotting
    
    Args:
        data: numpy array [n_samples, dim] or None if using distance_matrix
        save_diagrams_path: path to save persistence diagrams plot (PNG)
        save_json_path: path to save persistence diagrams data (JSON) for barcode script
        ripser_thresh: threshold for ripser filtration (None/inf = full filtration)
        ripser_maxdim: maximum homology dimension (default: 2, computes up to H²)
        ripser_coeff: compute homology with coefficients in Z/pZ for p=coeff (default: 47)
        distance_matrix: optional distance matrix of shape [n_samples, n_samples]
    
    Returns:
        dict with persistence_diagrams data if successful, None otherwise
    """
    if data is None and distance_matrix is None:
        print("Error: Both data and distance_matrix cannot be None")
        return None
    
    if data is not None:
        print(f"  Data shape: {data.shape}")
    
    if distance_matrix is not None:
        print(f"  Distance matrix shape: {distance_matrix.shape}")
    
    # Generate and save persistence diagram
    result = generate_persistence_diagram(
        data,
        max_dim=ripser_maxdim,
        thresh=ripser_thresh,
        coeff=ripser_coeff,
        save_diagrams_path=save_diagrams_path,
        distance_matrix=distance_matrix
    )
    
    # Save JSON data if path provided and we have data
    if result and save_json_path and 'persistence_diagrams' in result:
        try:
            with open(save_json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved persistence diagram data to {Path(save_json_path).name}")
        except Exception as e:
            print(f"  Warning: Could not save persistence diagram data to JSON: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate and save persistence diagrams (no analysis, no Betti numbers)'
    )
    parser.add_argument('--data_dir', type=str, default=None,
                      help='Directory with data representations (PCA, UMAP, etc.)')
    parser.add_argument('--fuzzy_dir', type=str, default=None,
                      help='Directory with fuzzy neighborhood distance matrices')
    parser.add_argument('--output_dir', type=str, default='./topology_analysis',
                      help='Output directory for persistence diagrams')
    parser.add_argument('--data_type', type=str, default='auto',
                      choices=['auto', 'pca', 'umap', 'downsampled'],
                      help='Type of data representation (auto=detect, pca, umap, downsampled)')
    parser.add_argument('--keys', type=str, nargs='+', default=None,
                      help='Specific keys to process (default: all)')
    parser.add_argument('--ripser_thresh', type=float, default=None,
                      help='Threshold for ripser filtration (default: None = full filtration)')
    parser.add_argument('--ripser_maxdim', type=int, default=2,
                      help='Maximum homology dimension (default: 2, computes up to H²)')
    parser.add_argument('--ripser_coeff', type=int, default=47,
                      help='Compute homology with coefficients in Z/pZ for p=coeff (default: 47)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.data_dir is None and args.fuzzy_dir is None:
        parser.error("Either --data_dir or --fuzzy_dir must be provided")
    
    if not RIPSER_AVAILABLE:
        parser.error("ripser is required but not available. Install with: pip install ripser")
    
    print("="*60)
    print("Persistence Diagram Generation")
    print("="*60)
    
    # Determine which source to use
    use_fuzzy = args.fuzzy_dir is not None
    source_dir = Path(args.fuzzy_dir) if use_fuzzy else Path(args.data_dir)
    
    # Find available results
    if args.keys:
        keys_to_process = args.keys
    else:
        if use_fuzzy:
            # Find fuzzy distance matrix files
            npz_files = list(source_dir.glob('*_fuzzy_dist.npz'))
            keys_to_process = sorted(set(
                f.name.replace('_fuzzy_dist.npz', '') for f in npz_files
            ))
        else:
            # Find data representation files
            keys_to_process = set()
            
            if args.data_type == 'umap' or args.data_type == 'auto':
                # Find UMAP files (pattern: {key}_umap_{n}d.npz)
                umap_files = list(source_dir.glob('*_umap_*d.npz'))
                for f in umap_files:
                    # Extract key by removing '_umap_{n}d.npz' pattern
                    name = f.name
                    if '_umap_' in name:
                        key = name.split('_umap_')[0]
                        keys_to_process.add(key)
            
            if args.data_type == 'downsampled' or args.data_type == 'auto':
                # Find downsampled PCA files
                downsampled_files = list(source_dir.glob('*_pca_downsampled.npz'))
                for f in downsampled_files:
                    key = f.name.replace('_pca_downsampled.npz', '')
                    keys_to_process.add(key)
            
            if args.data_type == 'pca' or args.data_type == 'auto':
                # Find PCA files
                pca_files = list(source_dir.glob('*_pca.npz'))
                for f in pca_files:
                    # Skip if it's downsampled (already counted)
                    if '_pca_downsampled' not in f.name:
                        key = f.name.replace('_pca.npz', '')
                        keys_to_process.add(key)
            
            if args.data_type == 'auto':
                # Find legacy format files
                legacy_files = list(source_dir.glob('*_reduced_*.npz'))
                for f in legacy_files:
                    key = f.name.split('_reduced_')[0]
                    keys_to_process.add(key)
            
            keys_to_process = sorted(keys_to_process)
    
    print(f"\nProcessing {len(keys_to_process)} representations...")
    print(f"Output directory: {args.output_dir}")
    
    if use_fuzzy:
        print(f"Input type: Distance matrix")
        print(f"  Source: {args.fuzzy_dir}")
    else:
        print(f"Input type: Data representation")
        print(f"  Source: {args.data_dir}")
        print(f"  Data type: {args.data_type}")
        
        # Debug: Show what files were found
        if len(keys_to_process) == 0:
            print(f"\n  Warning: No data files found in {args.data_dir}")
            print(f"  Looking for files matching:")
            if args.data_type == 'umap' or args.data_type == 'auto':
                print(f"    - *_umap_*d.npz (UMAP files)")
            if args.data_type == 'downsampled' or args.data_type == 'auto':
                print(f"    - *_pca_downsampled.npz (downsampled PCA)")
            if args.data_type == 'pca' or args.data_type == 'auto':
                print(f"    - *_pca.npz (PCA files, excluding downsampled)")
            if args.data_type == 'auto':
                print(f"    - *_reduced_*.npz (legacy format)")
            
            # List actual files in directory
            source_path = Path(args.data_dir)
            if source_path.exists():
                all_files = list(source_path.glob('*.npz'))
                if all_files:
                    print(f"\n  Found {len(all_files)} .npz files in directory:")
                    for f in sorted(all_files)[:10]:  # Show first 10
                        print(f"    - {f.name}")
                    if len(all_files) > 10:
                        print(f"    ... and {len(all_files) - 10} more")
                else:
                    print(f"\n  No .npz files found in {source_path.absolute()}")
            else:
                print(f"\n  Directory does not exist: {source_path.absolute()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each representation
    success_count = 0
    for key in keys_to_process:
        try:
            # Load data based on source type
            if use_fuzzy:
                # Load fuzzy distance matrix
                distance_matrix, pca_data = load_fuzzy_distance_matrix(args.fuzzy_dir, key)
                data = None
                
                print(f"\n{'='*60}")
                print(f"Processing: {key}")
                print(f"{'='*60}")
                print(f"  Distance matrix shape: {distance_matrix.shape}")
            else:
                # Load data representation (PCA, UMAP, or downsampled PCA)
                data, selected_indices, source_type = load_data_representation(
                    args.data_dir, key, data_type=args.data_type
                )
                distance_matrix = None
                
                if data is None or source_type is None:
                    print(f"\nSkipping {key}: No data representation found")
                    continue
                
                print(f"\n{'='*60}")
                print(f"Processing: {key}")
                print(f"{'='*60}")
                print(f"  Source type: {source_type}")
                print(f"  Shape: {data.shape} ({data.shape[0]} points × {data.shape[1]} dims)")
            
            # Generate and save persistence diagram
            diagrams_path = output_dir / f'{key}_persistence_diagram.png'
            json_path = output_dir / f'{key}_topology.json'
            
            result = generate_persistence_diagrams_only(
                data,
                save_diagrams_path=diagrams_path,
                save_json_path=json_path,
                ripser_thresh=args.ripser_thresh,
                ripser_maxdim=args.ripser_maxdim,
                ripser_coeff=args.ripser_coeff,
                distance_matrix=distance_matrix
            )
            
            if result:
                success_count += 1
                print(f"  ✓ Successfully generated persistence diagram and data")
            else:
                print(f"  ✗ Failed to generate persistence diagram")
            
        except Exception as e:
            print(f"  Error processing {key}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Persistence Diagram Generation Complete!")
    print(f"{'='*60}")
    print(f"Successfully generated {success_count}/{len(keys_to_process)} persistence diagrams")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()

