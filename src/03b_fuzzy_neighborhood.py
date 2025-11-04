"""
Fuzzy Neighborhood Distance Matrix for Persistent Homology
Generalized implementation for computing topologically-aware distance matrices

Purpose: Convert high-dimensional point cloud into distance matrix that:
- Adapts to local density variations
- Provides topological denoising
- Respects manifold structure
- Is suitable for persistent homology analysis
"""
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import brentq
from tqdm import tqdm
import warnings


def compute_base_distances(X, metric='euclidean'):
    """
    Step 1: Compute pairwise distances
    
    Input: X of shape (N, D)
    Output: distances of shape (N, N)
    
    Args:
        X: numpy array of shape [n_samples, n_features]
        metric: distance metric ('euclidean', 'cosine', 'manhattan')
    
    Returns:
        distances: symmetric distance matrix of shape [n_samples, n_samples]
    """
    if metric == 'cosine':
        # Cosine distance = 1 - cosine_similarity
        distances = squareform(pdist(X, metric='cosine'))
    elif metric == 'euclidean':
        distances = squareform(pdist(X, metric='euclidean'))
    elif metric == 'manhattan':
        distances = squareform(pdist(X, metric='cityblock'))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def find_k_nearest_neighbors(distances, k):
    """
    Step 2: Find k-nearest neighbors
    
    Input: distances of shape (N, N), k (int)
    Output: neighbors of shape (N, k) - indices of k-NN for each point
            neighbor_dists of shape (N, k) - distances to k-NN
    
    Args:
        distances: distance matrix of shape [n_samples, n_samples]
        k: number of nearest neighbors to find
    
    Returns:
        neighbors: array of shape [n_samples, k] with neighbor indices
        neighbor_dists: array of shape [n_samples, k] with neighbor distances
    """
    N = distances.shape[0]
    k = min(k, N - 1)  # Can't have more neighbors than points
    
    # For each point, find k nearest neighbors (excluding self)
    neighbors = np.zeros((N, k), dtype=int)
    neighbor_dists = np.zeros((N, k))
    
    for i in range(N):
        # Sort all points by distance, skip first (self)
        sorted_indices = np.argsort(distances[i, :])[1:k+1]
        neighbors[i] = sorted_indices
        neighbor_dists[i] = distances[i, sorted_indices]
    
    return neighbors, neighbor_dists


def compute_local_scales(neighbor_dists, k, target_entropy='auto'):
    """
    Step 3: Compute local scale parameters (σ_i)
    
    Input: neighbor_dists of shape (N, k)
           k (int)
           target_entropy: 'auto' uses log2(k), or provide float
    Output: sigmas of shape (N,) - local scale for each point
    
    Args:
        neighbor_dists: array of shape [n_samples, k] with neighbor distances
        k: number of neighbors
        target_entropy: target entropy value ('auto' or float)
    
    Returns:
        sigmas: array of shape [n_samples] with local scale parameters
    """
    N = neighbor_dists.shape[0]
    sigmas = np.zeros(N)
    
    if target_entropy == 'auto':
        target = np.log2(k)
    else:
        target = target_entropy
    
    print(f"  Computing local scales with target entropy: {target:.3f}...")
    for i in tqdm(range(N), desc="  Computing sigmas"):
        def objective(sigma):
            # Sum of membership strengths should equal target
            memberships = np.exp(-neighbor_dists[i] / (sigma + 1e-10))
            return np.sum(memberships) - target
        
        try:
            # Binary search for sigma
            sigmas[i] = brentq(objective, 1e-10, 1000, maxiter=100)
        except:
            # Fallback: use median distance
            sigmas[i] = np.median(neighbor_dists[i])
    
    return sigmas


def compute_raw_memberships(distances, neighbors, sigmas, k, epsilon=1e-10):
    """
    Step 4: Compute raw membership matrix
    
    Input: distances of shape (N, N)
           neighbors of shape (N, k)
           sigmas of shape (N,)
           k (int)
           epsilon (float) - for numerical stability
    Output: m_prime of shape (N, N) - asymmetric membership matrix (sparse)
    
    Args:
        distances: distance matrix of shape [n_samples, n_samples]
        neighbors: neighbor indices of shape [n_samples, k]
        sigmas: local scales of shape [n_samples]
        k: number of neighbors
        epsilon: numerical stability constant
    
    Returns:
        m_prime: asymmetric membership matrix of shape [n_samples, n_samples]
    """
    N = distances.shape[0]
    m_prime = np.zeros((N, N))
    
    print(f"  Computing raw memberships...")
    for i in tqdm(range(N), desc="  Raw memberships"):
        for j in neighbors[i]:
            # Membership strength from i's perspective
            m_prime[i, j] = np.exp(-distances[i, j] / (sigmas[i] + epsilon))
    
    # Clip very small values for numerical stability
    m_prime[m_prime < epsilon] = 0
    
    return m_prime


def symmetrize_memberships(m_prime, method='fuzzy_union'):
    """
    Step 5: Symmetrize membership matrix
    
    Input: m_prime of shape (N, N) - asymmetric membership matrix
           method: 'fuzzy_union', 'average', 'max', 'min'
    Output: m of shape (N, N) - symmetric membership matrix
    
    Args:
        m_prime: asymmetric membership matrix
        method: symmetrization method
    
    Returns:
        m: symmetric membership matrix
    """
    print(f"  Symmetrizing with method: {method}...")
    
    if method == 'fuzzy_union':
        # Probabilistic union: A ∪ B = A + B - A·B
        m = m_prime + m_prime.T - m_prime * m_prime.T
    elif method == 'average':
        # Simple average
        m = (m_prime + m_prime.T) / 2
    elif method == 'max':
        # Take maximum of two perspectives
        m = np.maximum(m_prime, m_prime.T)
    elif method == 'min':
        # Take minimum of two perspectives
        m = np.minimum(m_prime, m_prime.T)
    else:
        raise ValueError(f"Unknown symmetrization method: {method}")
    
    return m


def memberships_to_distances(m, epsilon=1e-10, sparsity_threshold=0.0):
    """
    Step 6: Convert to distance matrix
    
    Input: m of shape (N, N) - symmetric membership matrix
           epsilon (float) - avoid log(0)
           sparsity_threshold (float) - set weak connections to infinity
    Output: distance_matrix of shape (N, N)
    
    Args:
        m: symmetric membership matrix
        epsilon: numerical stability constant
        sparsity_threshold: threshold below which distances are set to infinity
    
    Returns:
        distance_matrix: symmetric distance matrix suitable for TDA
    """
    print(f"  Converting memberships to distances...")
    print(f"    Sparsity threshold: {sparsity_threshold}")
    
    N = m.shape[0]
    distance_matrix = np.zeros((N, N))
    
    # Apply sparsity threshold
    m_thresholded = m.copy()
    m_thresholded[m < sparsity_threshold] = 0
    
    # Convert membership to distance via -log
    # High membership → low distance
    # Low membership → high distance
    
    for i in tqdm(range(N), desc="  Converting to distances"):
        for j in range(N):
            if m_thresholded[i, j] > epsilon:
                distance_matrix[i, j] = -np.log(m_thresholded[i, j] + epsilon)
            else:
                # No connection: infinite distance
                distance_matrix[i, j] = 1e10  # Large finite number
    
    # Ensure symmetry (should already be symmetric)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Zero diagonal
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix


def compute_fuzzy_neighborhood_distance(X, n_neighbors=200, metric='euclidean',
                                      sym_method='fuzzy_union', epsilon=1e-10,
                                      sparsity_threshold=0.0, target_entropy='auto'):
    """
    Complete pipeline for fuzzy neighborhood distance computation
    
    Args:
        X: point cloud of shape [n_samples, n_features]
        n_neighbors: number of nearest neighbors
        metric: distance metric for ambient space
        sym_method: symmetrization method
        epsilon: numerical stability constant
        sparsity_threshold: threshold for sparsification
        target_entropy: target entropy for local scales
    
    Returns:
        dist_matrix: symmetric distance matrix of shape [n_samples, n_samples]
        info: dict with computation information
    """
    print(f"\nComputing fuzzy neighborhood distance matrix...")
    print(f"  Input shape: {X.shape}")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  metric: {metric}")
    print(f"  sym_method: {sym_method}")
    
    # Step 1: Compute pairwise distances
    distances = compute_base_distances(X, metric=metric)
    
    # Step 2: Find k-nearest neighbors
    neighbors, neighbor_dists = find_k_nearest_neighbors(distances, n_neighbors)
    
    # Step 3: Compute local scales
    sigmas = compute_local_scales(neighbor_dists, n_neighbors, target_entropy)
    
    # Step 4: Compute raw memberships
    m_prime = compute_raw_memberships(distances, neighbors, sigmas, n_neighbors, epsilon)
    
    # Step 5: Symmetrize memberships
    m = symmetrize_memberships(m_prime, method=sym_method)
    
    # Step 6: Convert to distance matrix
    dist_matrix = memberships_to_distances(m, epsilon, sparsity_threshold)
    
    # Collect information
    info = {
        'input_shape': X.shape,
        'n_neighbors': n_neighbors,
        'metric': metric,
        'sym_method': sym_method,
        'epsilon': epsilon,
        'sparsity_threshold': sparsity_threshold,
        'target_entropy': float(target_entropy) if target_entropy != 'auto' else 'auto',
        'sigma_mean': float(np.mean(sigmas)),
        'sigma_std': float(np.std(sigmas)),
        'sigma_min': float(np.min(sigmas)),
        'sigma_max': float(np.max(sigmas)),
        'membership_density': float(np.mean(m > epsilon)),
        'distance_mean': float(np.mean(dist_matrix[dist_matrix < 1e9])),
        'distance_std': float(np.std(dist_matrix[dist_matrix < 1e9]))
    }
    
    print(f"\nComputation complete!")
    print(f"  Sigma range: [{info['sigma_min']:.6f}, {info['sigma_max']:.6f}]")
    print(f"  Membership density: {info['membership_density']:.4f}")
    print(f"  Mean distance: {info['distance_mean']:.4f}")
    
    return dist_matrix, info


def load_pca_results(pca_dir, key):
    """Load PCA results"""
    pca_dir = Path(pca_dir)
    npz_file = pca_dir / f'{key}_pca.npz'
    
    if not npz_file.exists():
        raise FileNotFoundError(f"PCA results not found for {key} in {pca_dir}")
    
    data = np.load(npz_file)
    pca_reduced = data['pca_reduced']
    
    return pca_reduced


def process_representation(key, pca_dir, output_dir, n_neighbors=200,
                          metric='euclidean', sym_method='fuzzy_union',
                          epsilon=1e-10, sparsity_threshold=0.0,
                          target_entropy='auto'):
    """
    Process a single representation to compute fuzzy neighborhood distance
    
    Args:
        key: representation name
        pca_dir: directory with PCA results
        output_dir: output directory
        n_neighbors: number of nearest neighbors
        metric: distance metric
        sym_method: symmetrization method
        epsilon: numerical stability constant
        sparsity_threshold: sparsity threshold
        target_entropy: target entropy for local scales
    
    Returns:
        results: dict with distance matrix and info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {key}")
    print(f"{'='*60}")
    
    # Load PCA data
    print("\nLoading PCA data...")
    pca_data = load_pca_results(pca_dir, key)
    print(f"  Loaded PCA data: {pca_data.shape}")
    
    # Compute fuzzy neighborhood distance
    print("\nComputing fuzzy neighborhood distance matrix...")
    dist_matrix, info = compute_fuzzy_neighborhood_distance(
        pca_data,
        n_neighbors=n_neighbors,
        metric=metric,
        sym_method=sym_method,
        epsilon=epsilon,
        sparsity_threshold=sparsity_threshold,
        target_entropy=target_entropy
    )
    
    # Save results
    print(f"\nSaving results...")
    
    results = {
        'key': key,
        'pca_shape': pca_data.shape,
        'distance_shape': dist_matrix.shape,
        **info
    }
    
    # Save distance matrix
    npz_file = output_dir / f'{key}_fuzzy_dist.npz'
    np.savez(npz_file,
             distance_matrix=dist_matrix,
             pca_reduced=pca_data,  # Save original PCA data for reference
             pca_shape=pca_data.shape,
             distance_shape=dist_matrix.shape)
    print(f"  Saved distance matrix to {npz_file}")
    
    # Save info as JSON
    info_file = output_dir / f'{key}_fuzzy_info.json'
    with open(info_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved info to {info_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute fuzzy neighborhood distance matrices for persistent homology'
    )
    parser.add_argument('--pca_dir', type=str, required=True,
                      help='Directory with PCA results from 05a_pca_analysis.sh')
    parser.add_argument('--output_dir', type=str, default='./fuzzy_neighborhood',
                      help='Output directory for distance matrices')
    parser.add_argument('--n_neighbors', type=int, default=200,
                      help='Number of nearest neighbors (default: 200)')
    parser.add_argument('--metric', type=str, default='euclidean',
                      choices=['euclidean', 'cosine', 'manhattan'],
                      help='Distance metric for ambient space (default: euclidean)')
    parser.add_argument('--sym_method', type=str, default='fuzzy_union',
                      choices=['fuzzy_union', 'average', 'max', 'min'],
                      help='Symmetrization method (default: fuzzy_union)')
    parser.add_argument('--epsilon', type=float, default=1e-10,
                      help='Numerical stability constant (default: 1e-10)')
    parser.add_argument('--sparsity_threshold', type=float, default=0.0,
                      help='Sparsity threshold (default: 0.0)')
    parser.add_argument('--target_entropy', type=str, default='auto',
                      help='Target entropy for local scales (default: auto, uses log2(k))')
    parser.add_argument('--keys', type=str, nargs='+', default=None,
                      help='Specific keys to process (default: all)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Fuzzy Neighborhood Distance Matrix Computation")
    print("="*60)
    
    # Find available PCA results
    pca_dir = Path(args.pca_dir)
    
    if args.keys:
        keys_to_process = args.keys
    else:
        # Find all PCA .npz files
        npz_files = list(pca_dir.glob('*_pca.npz'))
        keys_to_process = sorted(set(
            f.name.replace('_pca.npz', '') for f in npz_files
        ))
    
    if not keys_to_process:
        print(f"Error: No PCA results found in {args.pca_dir}")
        print("Please run ./05a_pca_analysis.sh first")
        return
    
    print(f"\nProcessing {len(keys_to_process)} representations...")
    print(f"Output directory: {args.output_dir}")
    print(f"Configuration:")
    print(f"  n_neighbors: {args.n_neighbors}")
    print(f"  metric: {args.metric}")
    print(f"  sym_method: {args.sym_method}")
    print(f"  epsilon: {args.epsilon}")
    print(f"  sparsity_threshold: {args.sparsity_threshold}")
    print(f"  target_entropy: {args.target_entropy}")
    
    # Process each representation
    all_results = {}
    for key in keys_to_process:
        try:
            # Parse target_entropy if not 'auto'
            target_entropy = args.target_entropy
            if target_entropy != 'auto':
                try:
                    target_entropy = float(target_entropy)
                except ValueError:
                    print(f"  Warning: Could not parse target_entropy, using 'auto'")
                    target_entropy = 'auto'
            
            results = process_representation(
                key=key,
                pca_dir=args.pca_dir,
                output_dir=args.output_dir,
                n_neighbors=args.n_neighbors,
                metric=args.metric,
                sym_method=args.sym_method,
                epsilon=args.epsilon,
                sparsity_threshold=args.sparsity_threshold,
                target_entropy=target_entropy
            )
            all_results[key] = results
        except FileNotFoundError as e:
            print(f"  Skipping {key}: {e}")
    
    print(f"\n{'='*60}")
    print("Computation complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"\nGenerated files:")
    for key in keys_to_process:
        print(f"  - {key}_fuzzy_dist.npz (fuzzy distance matrix)")
        print(f"  - {key}_fuzzy_info.json (computation information)")


if __name__ == '__main__':
    main()

