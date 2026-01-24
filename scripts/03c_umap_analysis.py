"""
Apply UMAP and visualize dimensionality reduction results
Can work with either PCA or downsampled PCA data
Similar to manifold.ipynb visualization
"""
import argparse
import json
import numpy as np
from pathlib import Path

from data_loading_utils import load_pca_data, load_fuzzy_distance_matrix, load_representations
from visualization_utils import apply_umap, visualize_reduction_result


def main():
    parser = argparse.ArgumentParser(
        description='Apply UMAP and visualize dimensionality reduction results'
    )
    parser.add_argument('--pca_dir', type=str, default=None,
                      help='Directory with PCA results from 05a or downsampled results from 05b')
    parser.add_argument('--fuzzy_dir', type=str, default=None,
                      help='Directory with fuzzy neighborhood distance matrices from 05b_fuzzy_neighborhood')
    parser.add_argument('--representation_dir', type=str, required=True,
                      help='Directory with extracted representations (for metadata)')
    parser.add_argument('--output_dir', type=str, default='./umap_visualization',
                      help='Output directory for plots')
    parser.add_argument('--use_downsampled', action='store_true',
                      help='Use downsampled PCA data from 05b instead of regular PCA from 05a')
    parser.add_argument('--use_fuzzy', action='store_true',
                      help='Use fuzzy neighborhood distance matrices from 05b_fuzzy_neighborhood (metric will be precomputed)')
    parser.add_argument('--umap_n_components', type=int, default=3,
                      help='Target dimensions for UMAP (any positive integer; visualization only supports 2D/3D)')
    parser.add_argument('--umap_min_dist', type=float, default=0.1,
                      help='UMAP min_dist parameter (lower=more local, higher=more global)')
    parser.add_argument('--umap_n_neighbors', type=int, default=15,
                      help='UMAP n_neighbors parameter (lower=more local, higher=more global)')
    parser.add_argument('--umap_metric', type=str, default='cosine',
                      help='UMAP distance metric (default: cosine, also supports euclidean, manhattan, etc.)')
    parser.add_argument('--umap_random_state', type=int, default=None,
                      help='Random seed for UMAP (default: None, for reproducibility set to integer)')
    parser.add_argument('--save_umap_result', action='store_true',
                      help='Save UMAP embeddings as .npz files (default: False)')
    parser.add_argument('--generate_visualizations', action='store_true',
                      help='Generate visualization plots/images (default: False)')
    parser.add_argument('--keys', type=str, nargs='+', default=None,
                      help='Specific keys to visualize (default: all)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_fuzzy:
        if args.fuzzy_dir is None:
            print("Error: --fuzzy_dir is required when using --use_fuzzy")
            return
        if args.pca_dir is not None and not args.use_downsampled:
            print("Warning: --pca_dir is ignored when using --use_fuzzy")
    else:
        if args.pca_dir is None:
            print("Error: --pca_dir is required when not using --use_fuzzy")
            return
    
    # Validate dimensions for visualization
    if args.generate_visualizations and args.umap_n_components not in [2, 3]:
        print(f"Error: Visualization only supports 2D or 3D, but {args.umap_n_components}D was requested")
        print("  Set --generate_visualizations to False if you only want to save higher-dimensional data")
        return
    
    # Validate dimensions are positive
    if args.umap_n_components < 1:
        print(f"Error: --umap_n_components must be positive, got {args.umap_n_components}")
        return
    
    print("="*60)
    print("UMAP + Visualization")
    print("="*60)
    
    # Load metadata for token IDs
    _, metadata = load_representations(args.representation_dir)
    token_ids = np.array([m['token_id'] for m in metadata])
    
    # Find available results
    if args.use_fuzzy:
        fuzzy_dir = Path(args.fuzzy_dir)
        if args.keys:
            keys_to_process = args.keys
        else:
            # Find fuzzy distance matrix files
            npz_files = list(fuzzy_dir.glob('*_fuzzy_dist.npz'))
            keys_to_process = sorted(set(
                f.name.replace('_fuzzy_dist.npz', '') for f in npz_files
            ))
        
        if not keys_to_process:
            print(f"Error: No fuzzy distance matrices found in {fuzzy_dir}")
            print(f"Please run ./05b_fuzzy_neighborhood.sh first")
            return
    else:
        pca_dir = Path(args.pca_dir)
        if args.keys:
            keys_to_process = args.keys
        else:
            if args.use_downsampled:
                # Find downsampled files
                npz_files = list(pca_dir.glob('*_pca_downsampled.npz'))
                keys_to_process = sorted(set(
                    f.name.replace('_pca_downsampled.npz', '') for f in npz_files
                ))
            else:
                # Find regular PCA files
                npz_files = list(pca_dir.glob('*_pca.npz'))
                keys_to_process = sorted(set(
                    f.name.replace('_pca.npz', '') for f in npz_files
                ))
        
        if not keys_to_process:
            data_type = "downsampled" if args.use_downsampled else "PCA"
            print(f"Error: No {data_type} results found in {pca_dir}")
            print(f"Please run ./05a_pca_analysis.sh first" + 
                  (" or ./05b_density_downsampling.sh if using downsampled" if args.use_downsampled else ""))
            return
    
    print(f"\nProcessing {len(keys_to_process)} representations...")
    print(f"Output directory: {args.output_dir}")
    if args.use_fuzzy:
        print(f"Using fuzzy neighborhood distance matrices (precomputed)")
    else:
        print(f"Using {'downsampled' if args.use_downsampled else 'PCA'} data")
    print(f"UMAP target dimensions: {args.umap_n_components}D")
    print(f"UMAP min_dist: {args.umap_min_dist}")
    print(f"UMAP n_neighbors: {args.umap_n_neighbors}")
    if args.use_fuzzy:
        print(f"UMAP metric: precomputed (from fuzzy neighborhood)")
    else:
        print(f"UMAP metric: {args.umap_metric}")
    if args.umap_random_state is not None:
        print(f"UMAP random_state: {args.umap_random_state}")
    else:
        print(f"UMAP random_state: (random)")
    print(f"Save UMAP result: {args.save_umap_result}")
    print(f"Generate visualizations: {args.generate_visualizations}")
    if args.save_umap_result and args.umap_n_components > 3:
        print(f"  Note: Saving {args.umap_n_components}D data (visualization only supports 2D/3D)")
    
    # Process each representation
    for key in keys_to_process:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {key}")
            print(f"{'='*60}")
            
            if args.use_fuzzy:
                # Load fuzzy neighborhood distance matrix
                distance_matrix, pca_data = load_fuzzy_distance_matrix(args.fuzzy_dir, key)
                print(f"Loaded fuzzy distance matrix: {distance_matrix.shape}")
                if pca_data is not None:
                    print(f"  (Original PCA data shape: {pca_data.shape})")
                
                # Use all token_ids (no downsampling with fuzzy neighborhood)
                token_ids_subset = token_ids
                
                # Apply UMAP with precomputed distance matrix
                print("\nApplying UMAP with precomputed distance matrix...")
                umap_data, umap_info = apply_umap(
                    distance_matrix=distance_matrix,
                    n_components=args.umap_n_components,
                    min_dist=args.umap_min_dist,
                    n_neighbors=args.umap_n_neighbors,
                    metric='precomputed',
                    random_state=args.umap_random_state
                )
            else:
                # Load PCA or downsampled PCA data
                pca_data, selected_indices = load_pca_data(
                    args.pca_dir, key, use_downsampled=args.use_downsampled
                )
                print(f"Loaded data: {pca_data.shape}")
                
                # If downsampling was applied, subset token_ids accordingly
                if selected_indices is not None:
                    print(f"  (Downsampled from original, {len(selected_indices)} samples selected)")
                    token_ids_subset = token_ids[selected_indices]
                else:
                    token_ids_subset = token_ids
                
                # Apply UMAP
                print("\nApplying UMAP...")
                umap_data, umap_info = apply_umap(
                    data=pca_data,
                    n_components=args.umap_n_components,
                    min_dist=args.umap_min_dist,
                    n_neighbors=args.umap_n_neighbors,
                    metric=args.umap_metric,
                    random_state=args.umap_random_state
                )
            
            # Save UMAP result if requested
            if args.save_umap_result:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                umap_file = output_dir / f'{key}_umap_{args.umap_n_components}d.npz'
                info_file = output_dir / f'{key}_umap_{args.umap_n_components}d_info.json'
                
                # Save numpy array
                np.savez(umap_file,
                        umap_reduced=umap_data,
                        n_points=umap_data.shape[0],
                        n_dims=umap_data.shape[1])
                
                # Save info as JSON (convert numpy types to Python types)
                info_dict = {
                    'n_components': int(umap_info['n_components']),
                    'min_dist': float(umap_info['min_dist']),
                    'n_neighbors': int(umap_info['n_neighbors']),
                    'metric': str(umap_info['metric']),
                    'use_precomputed': bool(umap_info['use_precomputed']),
                    'random_state': umap_info.get('random_state', None),
                    'n_points': int(umap_data.shape[0]),
                    'n_dims': int(umap_data.shape[1])
                }
                with open(info_file, 'w') as f:
                    json.dump(info_dict, f, indent=2)
                
                print(f"\nSaved UMAP result: {umap_file.name}")
                print(f"  Shape: {umap_data.shape[0]} points × {umap_data.shape[1]} dimensions")
                print(f"  Info saved to: {info_file.name}")
            
            # Visualize if requested
            if args.generate_visualizations:
                print("\nGenerating visualizations...")
                visualize_reduction_result(
                    key=key,
                    reduced_data=umap_data,
                    token_ids=token_ids_subset,
                    output_dir=args.output_dir
                )
            
        except FileNotFoundError as e:
            print(f"  Skipping {key}: {e}")
    
    print(f"\n{'='*60}")
    print("UMAP + Visualization complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_dir}")
    if args.save_umap_result:
        print(f"\nUMAP embeddings saved as: {{key}}_umap_{args.umap_n_components}d.npz")
        print(f"  Each file contains: umap_reduced (shape: [n_points, {args.umap_n_components}])")
        if args.umap_n_components > 3:
            print(f"  Note: {args.umap_n_components}D data saved (visualization only supports 2D/3D)")
    if args.generate_visualizations:
        print(f"\nVisualizations saved as: {{key}}_umap_{args.umap_n_components}d.{{png,html}}")


if __name__ == '__main__':
    main()

