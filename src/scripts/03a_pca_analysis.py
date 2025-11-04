"""
PCA-only dimensionality reduction analysis
Pure PCA step without downsampling
"""
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

from utils import load_representations


def standardize_data(data):
    """
    Standardize data using StandardScaler
    
    Args:
        data: numpy array of shape [n_samples, n_features]
    
    Returns:
        standardized_data: standardized array
        scaler: fitted StandardScaler object
    """
    scaler = StandardScaler()
    standardized = scaler.fit_transform(data)
    return standardized, scaler


def pca_analysis(data, n_components=None, variance_threshold=None):
    """
    Perform PCA analysis with either target dimensionality or variance threshold
    
    Args:
        data: numpy array of shape [n_samples, n_features]
        n_components: number of components to keep (None = use variance_threshold)
        variance_threshold: variance to retain (None = use n_components)
    
    Returns:
        reduced_data: PCA-transformed data
        info: PCA information dictionary
        pca: fitted PCA object
        scaler: fitted StandardScaler object
    """
    print(f"  Original shape: {data.shape}")
    
    # Step 1: Standardize
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    
    # Step 2: Fit PCA
    pca = PCA()
    pca.fit(data_std)
    
    # Calculate cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    if n_components is not None:
        # Use target dimensionality
        num_components = int(min(n_components, data_std.shape[1]))
        print(f"  Using {num_components} components (target dimensionality)")
        variance_threshold_used = float(cumulative_variance[num_components - 1])
    elif variance_threshold is not None:
        # Auto-determine components based on variance threshold
        num_components = int(np.where(cumulative_variance >= variance_threshold)[0][0] + 1)
        print(f"  Number of components to retain {variance_threshold*100:.0f}% variance: {num_components}")
        variance_threshold_used = float(variance_threshold)
    else:
        raise ValueError("Either n_components or variance_threshold must be provided")
    
    # Transform
    reduced = pca.transform(data_std)[:, :num_components]
    explained_var = pca.explained_variance_ratio_[:num_components].sum()
    
    print(f"  Reduced shape: {reduced.shape}")
    print(f"  Explained variance: {explained_var:.3f}")
    
    info = {
        'num_components': int(num_components),
        'explained_variance_ratio': float(explained_var),
        'cumulative_variance': [float(v) for v in cumulative_variance[:num_components]],
        'variance_threshold': float(variance_threshold_used) if variance_threshold else None,
        'target_components': int(n_components) if n_components else None
    }
    
    return reduced, info, pca, scaler


def analyze_representation(key, data, output_dir, 
                          n_components=None, variance_threshold=None):
    """
    Analyze a representation using PCA-only pipeline
    
    Args:
        key: representation name
        data: numpy array [vocab_size, dim]
        output_dir: output directory
        n_components: target dimensionality (None = use variance_threshold)
        variance_threshold: variance to retain (None = use n_components)
    
    Returns:
        results: dict with reduced data and info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {key}")
    print(f"{'='*60}")
    
    # Perform PCA
    print("\nPerforming PCA...")
    data_pca, pca_info, pca_model, scaler = pca_analysis(
        data, 
        n_components=n_components,
        variance_threshold=variance_threshold
    )
    
    # Save results
    print(f"\nSaving results...")
    
    results = {
        'key': key,
        'original_shape': data.shape,
        'pca_shape': data_pca.shape,
        'pca_info': pca_info
    }
    
    # Save numpy array with PCA-reduced data
    npz_file = output_dir / f'{key}_pca.npz'
    np.savez(npz_file,
             pca_reduced=data_pca,
             original_shape=results['original_shape'],
             pca_shape=results['pca_shape'])
    print(f"  Saved PCA data to {npz_file}")
    
    # Save PCA model and scaler for later use
    pkl_file = output_dir / f'{key}_pca_model.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'pca': pca_model,
            'scaler': scaler,
            'pca_info': pca_info
        }, f)
    print(f"  Saved PCA model to {pkl_file}")
    
    # Save info as JSON for easy inspection
    info_file = output_dir / f'{key}_pca_info.json'
    with open(info_file, 'w') as f:
        json.dump(pca_info, f, indent=2)
    print(f"  Saved PCA info to {info_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Perform PCA dimensionality reduction analysis'
    )
    parser.add_argument('--representation_dir', type=str, required=True,
                      help='Directory with extracted representations')
    parser.add_argument('--output_dir', type=str, default='./pca_analysis',
                      help='Output directory for PCA results')
    parser.add_argument('--n_components', type=int, default=None,
                      help='Target dimensionality (use either this or --variance_threshold)')
    parser.add_argument('--variance_threshold', type=float, default=None,
                      help='Variance threshold for PCA (0-1, use either this or --n_components)')
    parser.add_argument('--keys', type=str, nargs='+', default=None,
                      help='Specific keys to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Validate that exactly one of n_components or variance_threshold is provided
    if (args.n_components is None) == (args.variance_threshold is None):
        parser.error("Exactly one of --n_components or --variance_threshold must be provided")
    
    print("="*60)
    print("PCA Dimensionality Reduction Analysis")
    print("="*60)
    
    # Load representations
    representations, metadata = load_representations(args.representation_dir)
    
    # Determine which keys to process
    if args.keys:
        keys_to_process = [k for k in args.keys if k in representations]
        if not keys_to_process:
            print(f"Error: None of the specified keys found")
            return
    else:
        keys_to_process = sorted(representations.keys())
    
    print(f"\nProcessing {len(keys_to_process)} representations...")
    print(f"Output directory: {args.output_dir}")
    if args.n_components:
        print(f"Target dimensionality: {args.n_components}")
    else:
        print(f"Variance threshold: {args.variance_threshold}")
    
    # Process each representation
    all_results = {}
    for key in keys_to_process:
        data = representations[key]
        results = analyze_representation(
            key=key,
            data=data,
            output_dir=args.output_dir,
            n_components=args.n_components,
            variance_threshold=args.variance_threshold
        )
        all_results[key] = results
    
    print(f"\n{'='*60}")
    print("PCA analysis complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"\nGenerated files:")
    for key in keys_to_process:
        print(f"  - {key}_pca.npz (PCA-reduced data)")
        print(f"  - {key}_pca_model.pkl (PCA model + scaler)")
        print(f"  - {key}_pca_info.json (PCA information)")


if __name__ == '__main__':
    main()

