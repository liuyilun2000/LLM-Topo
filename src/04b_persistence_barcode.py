"""
Plot persistence barcodes from topology analysis results
Draws barcode diagrams for H0, H1, H2 with top 30 longest bars, ordered by birth
"""
import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib import collections as mc
from scipy import stats


def load_topology_results(results_dir, key):
    """Load topology analysis results"""
    results_dir = Path(results_dir)
    json_file = results_dir / f'{key}_topology.json'
    
    if not json_file.exists():
        raise FileNotFoundError(f"Topology results not found for {key}")
    
    with open(json_file, 'r') as f:
        return json.load(f)


def select_top_k(persistence, top_k):
    """
    Select top-k longest persistence values
    
    Args:
        persistence: array of persistence values
        top_k: number of top values to select
    
    Returns:
        boolean array: True for top-k longest persistence values
    """
    if len(persistence) == 0:
        return np.array([])
    
    if len(persistence) <= top_k:
        # All values are top-k
        return np.array([True] * len(persistence))
    
    # Sort in descending order and get top-k threshold
    sorted_pers = np.sort(persistence)[::-1]
    threshold = sorted_pers[top_k - 1]  # k-th largest value (0-indexed)
    
    # Return boolean array: True for values >= threshold
    return persistence >= threshold


def plot_persistence_barcode(key, results, output_file, max_bars=30, p_value=0.05, top_k=10):
    """
    Plot persistence barcodes for H0, H1, H2
    
    Style:
    - Top 30 longest bars (by persistence = death - birth) for each dimension
    - Sorted vertically by birth time (descending)
    - Horizontal bars from birth to death
    - Shared x-axis (radius)
    - Infinite values are normalized to ceiling of largest finite value at loading
    - Top-k bars globally are selected based on difference above their dimension's average
    - Bars with top-k largest differences above mean are marked with stars
    
    Args:
        key: representation name
        results: topology analysis results dict
        output_file: output file path
        max_bars: maximum number of bars to display per dimension
        p_value: p-value threshold (kept for compatibility, not used)
        top_k: number of top bars globally (based on difference above mean) to mark as significant (default: 10)
    
    Returns:
        dict: Statistics for each dimension including mean, std, significant bars, etc.
    """
    # Use global top-k method: mark top-k bars with largest difference above their dimension's mean
    # Top-k bars globally (based on difference above mean) are marked with star (★)
    if 'persistence_diagrams' not in results:
        print(f"  Warning: No persistence diagrams found for {key}")
        return None
    
    diagrams = results['persistence_diagrams']
    
    dims = ['H0', 'H1', 'H2']
    dim_labels = ['$H^0$', '$H^1$', '$H^2$']
    
    # Find maximum death value across all dimensions (including finite values)
    all_deaths = []
    for dim_key in dims:
        if dim_key in diagrams:
            for birth, death in diagrams[dim_key]:
                if death != -1.0:  # Exclude infinite deaths for max calculation
                    all_deaths.append(death)
    
    # Determine max_radius as ceiling of largest finite value
    max_radius = 20
    if all_deaths:
        max_radius = math.ceil(max(all_deaths))
    
    # Replace all infinite values with max_radius (normalize infinity ONCE at loading)
    for dim_key in dims:
        if dim_key in diagrams:
            normalized_diagrams = []
            for birth, death in diagrams[dim_key]:
                if death == -1.0:  # Replace infinite with max_radius
                    normalized_diagrams.append((birth, max_radius))
                else:
                    normalized_diagrams.append((birth, death))
            diagrams[dim_key] = normalized_diagrams
    
    # Statistics dictionary to store per-dimension stats
    statistics = {}
    
    # Store selected pairs and statistics for each dimension (for plotting)
    dim_selected_pairs = {}
    dim_mean_persistence = {}  # Mean for grey area plotting
    dim_significance_threshold = {}  # Significance threshold value from top-k method
    
    # First pass: select max_bars for each dimension and collect all persistence values
    all_selected_pairs = []  # Store all selected pairs from all dimensions for global top-k
    dim_selected_pairs = {}
    for dim_key in dims:
        if dim_key not in diagrams or not diagrams[dim_key]:
            statistics[dim_key] = {
                'total_bars': 0,
                'selected_bars': 0,
                'mean_persistence': None,
                'std_persistence': None,
                'median_persistence': None,
                'min_persistence': None,
                'max_persistence': None,
                'num_significant': 0,
                'p_value_threshold': float(p_value),
                'top_k': int(top_k),
                'significance_threshold_value': None,
                'significant_bars': []
            }
            dim_significant_sets[dim_key] = set()
            dim_selected_pairs[dim_key] = []  # Empty list of normalized pairs
            dim_mean_persistence[dim_key] = 0.0
            dim_significance_threshold[dim_key] = 0.0
            continue
        
        pairs = diagrams[dim_key]
        
        # All values are now normalized (no infinite values)
        # Format: (birth, death, persistence)
        normalized_pairs = []
        for birth, death in pairs:
            persistence = death - birth
            normalized_pairs.append((birth, death, persistence))
        
        # Sort by persistence descending, take top max_bars
        normalized_pairs.sort(key=lambda x: x[2], reverse=True)
        selected_normalized = normalized_pairs[:max_bars]
        
        # Store selected pairs for this dimension
        dim_selected_pairs[dim_key] = selected_normalized
        
        # Calculate statistics on selected persistence values first
        persistence_values = [pers for _, _, pers in selected_normalized]
        if persistence_values:
            mean_pers = np.mean(persistence_values)
            std_pers = np.std(persistence_values)
            median_pers = np.median(persistence_values)
            min_pers = np.min(persistence_values)
            max_pers = np.max(persistence_values)
        else:
            mean_pers = None
            std_pers = None
            median_pers = None
            min_pers = None
            max_pers = None
        
        # Store mean for later use
        dim_mean_persistence[dim_key] = mean_pers if mean_pers is not None else 0.0
        
        # Collect all selected pairs from all dimensions for global top-k
        # Calculate difference above mean for each bar
        for birth, death, persistence in selected_normalized:
            # Calculate difference above this dimension's average
            if mean_pers is not None and mean_pers > 0:
                difference_above_mean = persistence - mean_pers
            else:
                difference_above_mean = persistence  # Fallback if no mean
            all_selected_pairs.append((dim_key, birth, death, persistence, difference_above_mean))
        statistics[dim_key] = {
            'total_bars': len(pairs),
            'selected_bars': len(selected_normalized),
            'mean_persistence': float(mean_pers) if mean_pers is not None else None,
            'std_persistence': float(std_pers) if std_pers is not None else None,
            'median_persistence': float(median_pers) if median_pers is not None else None,
            'min_persistence': float(min_pers) if min_pers is not None else None,
            'max_persistence': float(max_pers) if max_pers is not None else None,
            'p_value_threshold': float(p_value),
            'top_k': int(top_k),
            'significance_threshold_value': None,  # Will be updated after global top-k
            'num_significant': 0,  # Will be updated after global top-k
            'significant_bars': []  # Will be updated after global top-k
        }
    
    # Second pass: select global top-k from all dimensions based on difference above mean
    dim_significant_sets = {}
    global_significance_threshold = None
    
    if all_selected_pairs:
        # Extract all difference above mean values from all dimensions
        all_difference_values = [diff for _, _, _, _, diff in all_selected_pairs]
        
        if all_difference_values:
            # Select global top-k based on difference above mean
            all_diff_array = np.array(all_difference_values)
            global_is_significant_mask = select_top_k(all_diff_array, top_k)
            
            # Find global threshold value (will be set based on persistence values)
            
            # Create mapping from (dim_key, birth, death, persistence, diff) to significance
            for i, (dim_key, birth, death, persistence, diff_above_mean) in enumerate(all_selected_pairs):
                if global_is_significant_mask[i]:
                    if dim_key not in dim_significant_sets:
                        dim_significant_sets[dim_key] = set()
                    dim_significant_sets[dim_key].add((birth, death))
                    # Store the threshold value for statistics (use the persistence value)
                    if global_significance_threshold is None:
                        global_significance_threshold = persistence
                    else:
                        global_significance_threshold = min(global_significance_threshold, persistence)
    
    # Third pass: update statistics for each dimension
    for dim_key in dims:
        if dim_key not in diagrams or not diagrams[dim_key]:
            dim_significant_sets[dim_key] = set()
            continue
        
        # Ensure dim_key has an entry in dim_significant_sets (even if empty)
        if dim_key not in dim_significant_sets:
            dim_significant_sets[dim_key] = set()
        
        significant_set = dim_significant_sets[dim_key]
        significant_bars_list = []
        selected_normalized = dim_selected_pairs[dim_key]
        mean_pers = dim_mean_persistence[dim_key]
        
        for birth, death, persistence in selected_normalized:
            if (birth, death) in significant_set:
                significant_bars_list.append({
                    'birth': float(birth),
                    'death': float(death),
                    'persistence': float(persistence),
                    'mean_persistence': float(mean_pers) if mean_pers is not None else None,
                    'significance_threshold': float(global_significance_threshold) if global_significance_threshold is not None else None
                })
        
        # Update statistics
        statistics[dim_key]['significance_threshold_value'] = float(global_significance_threshold) if global_significance_threshold is not None else None
        statistics[dim_key]['num_significant'] = len(significant_bars_list)
        statistics[dim_key]['significant_bars'] = significant_bars_list
        dim_significance_threshold[dim_key] = global_significance_threshold if global_significance_threshold is not None else 0.0
    
    # Process each homology dimension
    fig, axes = plt.subplots(3, 1, figsize=(3, 5), sharex=True)
    
    for idx, (dim_key, dim_label) in enumerate(zip(dims, dim_labels)):
        ax = axes[idx]
        
        if dim_key not in diagrams or not diagrams[dim_key]:
            ax.text(0.5, 0.5, f'No {dim_key} features', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(dim_label, fontsize=12, rotation=0, va='center', ha='right')
            ax.set_xlim(0, max_radius)
            # Hide top and right spines
            ax.spines.top.set_visible(False)
            ax.spines.right.set_visible(False)
            continue
        
        # Use the stored selected normalized pairs (already selected in first pass)
        # Format: (birth, death, persistence)
        selected_normalized = dim_selected_pairs[dim_key]
        mean_pers = dim_mean_persistence[dim_key]
        significance_threshold_value = dim_significance_threshold[dim_key]
        
        if not selected_normalized:
            continue
        
        # Sort by birth ascending for vertical ordering (earlier first)
        # For bars with same birth time, sort by death time descending (latest first)
        top_normalized = sorted(selected_normalized, key=lambda x: (-x[0], x[1]))
        
        # Prepare data for plotting
        bars = []
        bar_persistences = []  # Store persistence values
        
        for birth, death, persistence in top_normalized:
            # Check if this is statistically significant for this dimension
            # Use .get() to handle case where dim_key might not be in dim_significant_sets
            is_significant = (birth, death) in dim_significant_sets.get(dim_key, set())
            
            # All values are now normalized (no infinite handling needed)
            bars.append((birth, death, is_significant))
            bar_persistences.append(persistence)
        
        # Draw bars
        if bars:
            bar_width = 0.6
            y = np.arange(len(bars))
            
            # First, draw continuous grey area showing mean persistence length
            # Only draw if we have a valid mean persistence
            if mean_pers > 0:
                # Prepare arrays for continuous area plot
                x_bottoms = [birth for birth, _, _ in bars]
                x_tops_mean = [birth + mean_pers for birth, _, _ in bars]
                
                ax.fill_betweenx(y, x_bottoms, x_tops_mean,
                                color='gray', alpha=0.25, zorder=0,
                                edgecolor='none')
            
            # Then draw horizontal bars (green) on top
            for i, (birth, death, is_significant) in enumerate(bars):
                ax.barh(i, death - birth, left=birth, height=bar_width, 
                       color='green', alpha=0.8, edgecolor='darkgreen', linewidth=0.5,
                       zorder=1)
                
                # Mark significant bars with star
                if is_significant:
                    # Place star slightly to the right of the bar end
                    star_offset = 0.05
                    ax.text(death + star_offset, i, '★',
                           fontsize=6, color='black', rotation=0, ha='left',
                           verticalalignment='center', zorder=2)
            
            # Hide all y-ticks
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylim(-0.5, len(bars) - 0.5)
        
        ax.set_ylabel(dim_label, fontsize=12, rotation=0, va='center', ha='right')
        ax.set_xlim(0, max_radius)
        # Set x-axis ticks: only show start (0) and end (max_radius)
        ax.set_xticks([0, max_radius])
        if idx == 2:  # Only bottom subplot shows labels
            ax.set_xticklabels(['0', f'{max_radius:.0f}'])
        else:
            ax.set_xticklabels([])
        # Hide tick marks but keep labels
        ax.tick_params(left=False, labelleft=False, right=False, labelright=False, 
                      top=False, labeltop=False, bottom=(idx == 2), labelbottom=(idx == 2),
                      length=0, width=0)  # Hide tick marks but keep labels
        # Hide top and right spines
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        # Only show x-axis label on bottom subplot (H2)
        if idx == 2:
            ax.set_xlabel('Radius', fontsize=11)
        else:
            ax.set_xlabel('')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    
    # Also save as PDF
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"  Saved barcode to {Path(output_file).name}")
    print(f"  Saved barcode to {Path(pdf_file).name}")
    
    # Return statistics for saving
    return statistics


def main():
    parser = argparse.ArgumentParser(
        description='Plot persistence barcodes from topology analysis results'
    )
    parser.add_argument('--topology_dir', type=str, required=True,
                      help='Directory with topology analysis JSON files')
    parser.add_argument('--output_dir', type=str, default='./persistence_barcode',
                      help='Output directory for barcode plots')
    parser.add_argument('--max_bars', type=int, default=30,
                      help='Maximum number of bars to display per dimension (default: 30)')
    parser.add_argument('--p_value', type=float, default=0.05,
                      help='P-value threshold (kept for compatibility, not used)')
    parser.add_argument('--top_k', type=int, default=10,
                      help='Number of top bars (based on difference above their dimension\'s mean) to mark as significant (default: 10)')
    parser.add_argument('--keys', type=str, nargs='+', default=None,
                      help='Specific keys to visualize (default: all)')
    
    args = parser.parse_args()
    
    topology_dir = Path(args.topology_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Persistence Barcode Visualization")
    print("="*60)
    
    # Find topology JSON files
    if args.keys:
        keys_to_process = args.keys
    else:
        json_files = list(topology_dir.glob('*_topology.json'))
        keys_to_process = sorted(set(
            f.name.replace('_topology.json', '') for f in json_files
        ))
    
    if not keys_to_process:
        print(f"Error: No topology JSON files found in {topology_dir}")
        return
    
    print(f"\nProcessing {len(keys_to_process)} representations...")
    print(f"Output directory: {output_dir}")
    print(f"Max bars per dimension: {args.max_bars}")
    print(f"Top-k: {args.top_k} (global across all dimensions, based on difference above mean)")
    print(f"Using global top-k method approach (difference above dimension's average)")
    print(f"Stars mark top-k bars with largest difference above their dimension's mean")
    
    # Process each representation
    for key in keys_to_process:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {key}")
            print(f"{'='*60}")
            
            # Load results
            results = load_topology_results(topology_dir, key)
            
            # Plot barcode and get statistics
            output_file = output_dir / f'{key}_barcode.png'
            statistics = plot_persistence_barcode(key, results, output_file, 
                                                   max_bars=args.max_bars, 
                                                   p_value=args.p_value,
                                                   top_k=args.top_k)
            
            # Save statistics to JSON
            if statistics:
                stats_file = output_dir / f'{key}_statistics.json'
                # Convert numpy types and inf to JSON-serializable types
                def convert_for_json(obj):
                    if isinstance(obj, dict):
                        return {k: convert_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_for_json(item) for item in obj]
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
                        return 'inf' if np.isinf(obj) else 'nan'
                    return obj
                
                statistics_json = convert_for_json(statistics)
                with open(stats_file, 'w') as f:
                    json.dump(statistics_json, f, indent=2)
                print(f"  Saved statistics to {stats_file.name}")
            
        except FileNotFoundError as e:
            print(f"  Skipping {key}: {e}")
        except Exception as e:
            print(f"  Error processing {key}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Barcode visualization complete!")
    print(f"{'='*60}")
    print(f"\nPlots saved to: {output_dir}")


if __name__ == '__main__':
    main()
