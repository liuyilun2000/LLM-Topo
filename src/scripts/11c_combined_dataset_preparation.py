"""
Script to combine target natural language dataset with source graph walk sequences

This script:
1. Loads a target dataset (e.g., TinyStories) and its tokenizer
2. Loads source graph walk sequences
3. Extends the tokenizer vocabulary with source tokens
4. Inserts source tokens into target sequences (ensuring minimum count per sequence)
5. Combines with pure source sequences
6. Shuffles and saves the combined dataset
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import tempfile


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_source_walks(csv_path):
    """Load source graph walk sequences from CSV and extract unique node IDs"""
    print(f"Loading source walks from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} source walks")
    
    # Extract all unique node IDs
    all_node_ids = set()
    for seq in df['sequence_labels']:
        node_ids = [str(x) for x in str(seq).split()]
        all_node_ids.update(node_ids)
    
    node_ids = sorted([int(x) for x in all_node_ids])
    print(f"Found {len(node_ids)} unique source tokens (node IDs: {min(node_ids)} to {max(node_ids)})")
    
    return df, node_ids


def load_target_dataset(dataset_name, dataset_config=None, split='train', streaming=False):
    """Load a target dataset from HuggingFace"""
    print(f"Loading target dataset: {dataset_name}" + (f" (config: {dataset_config})" if dataset_config else ""))
    
    if streaming:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        print("Streaming mode: will process in batches")
        return dataset, None
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        print(f"Loaded {len(dataset)} samples")
        return dataset, len(dataset)


# ============================================================================
# Tokenizer Extension Functions
# ============================================================================

def extend_tokenizer_vocab(tokenizer, source_node_ids, source_token_prefix="<GRAPH_", source_token_start="<SOURCE_START>"):
    """
    Extend tokenizer vocabulary with source tokens and special start token
    
    Returns:
        Extended tokenizer, mapping from node_id to token_id, source_start_token_id
    """
    print(f"\nExtending tokenizer vocabulary with {len(source_node_ids)} source tokens...")
    
    original_vocab_size = len(tokenizer)
    print(f"Original vocab size: {original_vocab_size}")
    
    # Create source tokens
    source_tokens = [f"{source_token_prefix}{node_id}>" for node_id in source_node_ids]
    
    # Add source tokens to tokenizer (not as special tokens)
    tokenizer.add_tokens(source_tokens, special_tokens=False)
    
    # Add source start token separately as a special token if provided
    source_start_token_id = None
    if source_token_start:
        tokenizer.add_tokens([source_token_start], special_tokens=True)
        source_start_token_id = tokenizer.convert_tokens_to_ids(source_token_start)
        print(f"Added source start token: {source_token_start} (ID: {source_start_token_id})")
    
    # Build mapping from node_id to token_id
    node_to_token_id = {}
    for node_id in source_node_ids:
        token_str = f"{source_token_prefix}{node_id}>"
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        node_to_token_id[node_id] = token_id
    
    new_vocab_size = len(tokenizer)
    print(f"New vocab size: {new_vocab_size} (+{new_vocab_size - original_vocab_size})")
    
    return tokenizer, node_to_token_id, source_start_token_id


# ============================================================================
# Source Token Distribution Functions
# ============================================================================

def sample_source_token_count(min_count, max_count, power_law_alpha=2.0, rng=None):
    """
    Sample number of source tokens to insert using a truncated power law distribution.
    
    This simulates natural language patterns where most sequences have few source tokens
    (like occasional mentions of days/months) and few sequences have many source tokens.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if max_count is None or max_count <= min_count:
        return min_count
    
    # Power law distribution: P(k) ∝ k^(-alpha)
    # For truncated power law on [a, b]: 
    # x = [a^(1-alpha) + u * (b^(1-alpha) - a^(1-alpha))]^(1/(1-alpha))
    u = rng.random()
    
    if power_law_alpha == 1.0:
        # Special case: uniform distribution
        count = int(rng.integers(min_count, max_count + 1))
    else:
        a, b = float(min_count), float(max_count)
        x = (a**(1 - power_law_alpha) + u * (b**(1 - power_law_alpha) - a**(1 - power_law_alpha)))**(1 / (1 - power_law_alpha))
        count = int(np.clip(np.round(x), min_count, max_count))
    
    return count


# ============================================================================
# Source Sequence Iterator Class
# ============================================================================

class SourceSequenceIterator:
    """
    Iterator that sequentially extracts tokens from source sequences.
    Cycles through sequences when exhausted to ensure tokens are always available.
    """
    
    def __init__(self, source_sequences, rng=None, cycle=True):
        self.source_sequences = source_sequences
        self.rng = rng if rng is not None else np.random.default_rng()
        self.current_seq_idx = 0
        self.current_pos = 0
        self.total_extracted = 0
        self.sequences_used = 0
        self.cycle = cycle
        self.cycles_completed = 0
    
    def _reset_to_beginning(self):
        """Reset iterator to beginning of sequences (for cycling)"""
        self.current_seq_idx = 0
        self.current_pos = 0
        self.cycles_completed += 1
    
    def extract_fragment(self, num_tokens):
        """
        Extract a contiguous fragment from current source sequence.
        Moves to next sequence if current one is exhausted.
        If cycle=True and all sequences are exhausted, restarts from the beginning.
        """
        # If exhausted and cycling enabled, restart
        if self.current_seq_idx >= len(self.source_sequences):
            if self.cycle and len(self.source_sequences) > 0:
                self._reset_to_beginning()
            else:
                return []
        
        while self.current_seq_idx < len(self.source_sequences):
            current_seq = self.source_sequences[self.current_seq_idx]
            available = len(current_seq) - self.current_pos
            
            if available == 0:
                # Current sequence exhausted, move to next
                self.current_seq_idx += 1
                self.current_pos = 0
                
                # If exhausted and cycling enabled, restart
                if self.current_seq_idx >= len(self.source_sequences):
                    if self.cycle and len(self.source_sequences) > 0:
                        self._reset_to_beginning()
                    else:
                        return []
                continue
            
            # Extract contiguous fragment
            fragment_length = min(num_tokens, available)
            fragment = current_seq[self.current_pos:self.current_pos + fragment_length]
            
            # Update position
            self.current_pos += fragment_length
            self.total_extracted += fragment_length
            
            # Track if we've fully used this sequence
            if self.current_pos == len(current_seq):
                self.sequences_used += 1
            
            return fragment
        
        return []
    
    def has_more(self):
        """Check if there are more tokens available"""
        if self.current_seq_idx >= len(self.source_sequences):
            # If cycling is enabled and we have sequences, we always have more
            return self.cycle and len(self.source_sequences) > 0
        
        # Check if current sequence has more tokens
        if self.current_pos < len(self.source_sequences[self.current_seq_idx]):
            return True
        
        # Current sequence exhausted, but we might have more sequences or can cycle
        return (self.current_seq_idx + 1 < len(self.source_sequences)) or (self.cycle and len(self.source_sequences) > 0)


# ============================================================================
# Token Insertion Functions
# ============================================================================

def insert_source_tokens_scattered(target_token_ids, tokens_to_insert, source_start_token_id=None, rng=None):
    """
    Insert source tokens into a target sequence at random positions (scattered).
    
    Each source token is inserted at a random position, with optional SOURCE_START token
    prepended before it.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if len(tokens_to_insert) == 0:
        return target_token_ids
    
    # Insert at random positions (scattered)
    combined = target_token_ids.copy()
    for token in tokens_to_insert:
        # Insert at random position (including at the end)
        if isinstance(rng, np.random.Generator):
            insert_pos = rng.integers(0, len(combined) + 1)
        else:
            insert_pos = rng.randint(0, len(combined) + 1)
        
        # Insert source start token before source token if provided
        if source_start_token_id is not None:
            combined.insert(insert_pos, source_start_token_id)
            insert_pos += 1
        
        combined.insert(insert_pos, token)
    
    return combined


# ============================================================================
# Statistics Estimation Functions
# ============================================================================

def estimate_source_token_counts(
    num_target_sequences,
    node_to_token_id,
    min_source_per_seq,
    max_source_per_seq,
    power_law_alpha,
    source_ratio,
    source_start_token_id=None,
    source_df=None
):
    """
    Estimate expected counts for each source token before processing.
    
    This helps validate that actual counts match expectations after processing.
    """
    # Estimate average tokens per target sequence from power law distribution
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    samples = [sample_source_token_count(min_source_per_seq, max_source_per_seq, power_law_alpha, rng) 
               for _ in range(10000)]
    avg_tokens_per_seq = np.mean(samples)
    
    # Total tokens inserted into target sequences
    total_tokens_inserted = num_target_sequences * avg_tokens_per_seq
    
    # Estimate pure source sequences
    num_pure_source = int(num_target_sequences * source_ratio / (1 - source_ratio)) if source_ratio > 0 else 0
    
    # Calculate average source sequence length
    if source_df is not None:
        lengths = [len(str(seq).split()) for seq in source_df['sequence_labels']]
        avg_source_seq_length = np.mean(lengths) if lengths else 50
    else:
        avg_source_seq_length = 50  # Rough estimate
    
    # Total tokens from pure source sequences
    total_pure_source_tokens = num_pure_source * avg_source_seq_length
    
    # Estimate SOURCE_START token counts
    # Each inserted token has one SOURCE_START before it
    # Each pure source sequence has one SOURCE_START at the beginning
    source_start_count = total_tokens_inserted + num_pure_source
    
    # Distribute tokens across source tokens
    estimated_counts = {}
    source_token_ids = list(node_to_token_id.values())
    num_source_tokens = len(source_token_ids)
    
    if source_df is not None:
        # Use actual distribution from source sequences for pure source sequences
        node_counts = {}
        for seq in source_df['sequence_labels']:
            node_ids = [int(x) for x in str(seq).split()]
            for node_id in node_ids:
                node_counts[node_id] = node_counts.get(node_id, 0) + 1
        
        # Scale to match expected pure source tokens
        total_node_occurrences = sum(node_counts.values())
        if total_node_occurrences > 0:
            scale_factor = total_pure_source_tokens / total_node_occurrences
            for node_id, count in node_counts.items():
                token_id = node_to_token_id.get(node_id)
                if token_id is not None:
                    estimated_counts[token_id] = count * scale_factor
        
        # Add uniform distribution for inserted tokens
        tokens_per_source_inserted = total_tokens_inserted / num_source_tokens if num_source_tokens > 0 else 0
        for token_id in source_token_ids:
            estimated_counts[token_id] = estimated_counts.get(token_id, 0) + tokens_per_source_inserted
    else:
        # Uniform distribution (simplified)
        tokens_per_source = (total_tokens_inserted + total_pure_source_tokens) / num_source_tokens if num_source_tokens > 0 else 0
        for token_id in source_token_ids:
            estimated_counts[token_id] = tokens_per_source
    
    # SOURCE_START token
    if source_start_token_id is not None:
        estimated_counts[source_start_token_id] = source_start_count
    
    return estimated_counts, {
        'avg_tokens_per_seq': avg_tokens_per_seq,
        'total_tokens_inserted': total_tokens_inserted,
        'num_pure_source': num_pure_source,
        'avg_source_seq_length': avg_source_seq_length,
        'total_pure_source_tokens': total_pure_source_tokens,
        'source_start_count': source_start_count
    }


# ============================================================================
# Dataset Processing Helper Functions
# ============================================================================

def _tokenize_example(example, tokenizer):
    """Extract and tokenize text from a dataset example"""
    if 'input_ids' in example:
        return example['input_ids']
    elif 'text' in example:
        tokenized = tokenizer(example['text'], add_special_tokens=False, return_attention_mask=False)
        return tokenized['input_ids']
    else:
        # Try to find text field (use first string field)
        text = None
        for key in example.keys():
            if isinstance(example[key], str):
                text = example[key]
                break
        if text is None:
            # Fallback: convert first field to string
            text = str(example.get(list(example.keys())[0]))
        tokenized = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        return tokenized['input_ids']


def _process_target_sequences(
    target_dataset,
    source_iterator,
    tokenizer,
    node_to_token_id,
    source_token_ids_set,
    actual_token_counts,
    min_source_per_seq,
    max_source_per_seq,
    power_law_alpha,
    max_length,
    source_start_token_id,
    rng,
    batch_size
):
    """
    Process target sequences: insert source tokens and collect statistics.
    
    Returns:
        List of batches, insertion_counts array
    """
    insertion_counts = []
    target_batches = []
    batch = []
    
    for example in tqdm(target_dataset, desc="Processing target sequences"):
        # Tokenize target text
        target_token_ids = _tokenize_example(example, tokenizer)
        
        # Sample number of source tokens to insert
        num_insertions = sample_source_token_count(
            min_source_per_seq, max_source_per_seq, power_law_alpha, rng
        )
        insertion_counts.append(num_insertions)
        
        # Extract fragment from source sequences and insert into target
        if source_iterator.has_more():
            fragment = source_iterator.extract_fragment(num_insertions)
            combined_ids = insert_source_tokens_scattered(
                target_token_ids, fragment, source_start_token_id, rng
            )
        else:
            combined_ids = target_token_ids
        
        # Truncate if needed
        if max_length and len(combined_ids) > max_length:
            combined_ids = combined_ids[:max_length]
        
        # Count source tokens in final sequence (after truncation) for accurate statistics
        for token_id in combined_ids:
            if token_id in source_token_ids_set:
                actual_token_counts[token_id] += 1
        
        batch.append({
            'input_ids': combined_ids,
            'length': len(combined_ids)
        })
        
        # Save batch when it reaches batch_size
        if len(batch) >= batch_size:
            target_batches.append(batch)
            batch = []
    
    # Add remaining batch
    if batch:
        target_batches.append(batch)
    
    return target_batches, np.array(insertion_counts)


def _process_pure_source_sequences(
    source_df,
    node_to_token_id,
    source_token_ids_set,
    actual_token_counts,
    max_length,
    source_start_token_id,
    source_ratio,
    num_target_sequences,
    rng
):
    """
    Process pure source sequences: convert to token IDs and select based on ratio.
    
    Returns:
        List of selected source sequences
    """
    source_sequences = []
    
    for idx, row in tqdm(source_df.iterrows(), total=len(source_df), desc="Processing source sequences"):
        # Convert node IDs to token IDs
        sequence = str(row['sequence_labels'])
        node_ids = [int(x) for x in sequence.split()]
        token_ids = [node_to_token_id[node_id] for node_id in node_ids]
        
        # Prepend source start token at the beginning if provided
        if source_start_token_id is not None:
            token_ids = [source_start_token_id] + token_ids
        
        # Truncate if needed
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        source_sequences.append({
            'input_ids': token_ids,
            'length': len(token_ids)
        })
    
    # Apply source ratio: determine how many source sequences to include
    if source_ratio > 0:
        target_source = int(num_target_sequences * source_ratio / (1 - source_ratio))
        if len(source_sequences) > target_source:
            # Sample source sequences
            indices = rng.choice(len(source_sequences), size=target_source, replace=False)
            selected_source_sequences = [source_sequences[i] for i in indices]
            
            # Count tokens in selected pure source sequences
            for seq_data in selected_source_sequences:
                for token_id in seq_data['input_ids']:
                    if token_id in source_token_ids_set:
                        actual_token_counts[token_id] += 1
            
            print(f"\nSampling {target_source} source sequences (ratio: {source_ratio})")
            return selected_source_sequences
        else:
            # Count tokens in all pure source sequences
            for seq_data in source_sequences:
                for token_id in seq_data['input_ids']:
                    if token_id in source_token_ids_set:
                        actual_token_counts[token_id] += 1
            
            print(f"\nUsing all {len(source_sequences)} source sequences (requested ratio would need {target_source})")
            return source_sequences
    else:
        return []


def _create_combined_dataset(target_batches, source_sequences, seed):
    """Combine target and source batches into a single shuffled dataset"""
    all_batches = target_batches + ([source_sequences] if source_sequences else [])
    
    # Create dataset from batches
    all_datasets = []
    for batch in tqdm(all_batches, desc="Creating dataset batches"):
        if batch:  # Skip empty batches
            batch_dataset = Dataset.from_list(batch)
            all_datasets.append(batch_dataset)
    
    # Concatenate all batches
    if len(all_datasets) > 1:
        dataset = concatenate_datasets(all_datasets)
    elif all_datasets:
        dataset = all_datasets[0]
    else:
        dataset = Dataset.from_list([])
    
    # Shuffle the entire dataset
    dataset = dataset.shuffle(seed=seed)
    
    return DatasetDict({'train': dataset})


def _print_statistics(actual_token_counts, estimated_counts, insertion_counts, source_iterator, source_token_sequences):
    """Print statistics about token insertion and distribution"""
    # Report insertion distribution
    if len(insertion_counts) > 0:
        print(f"\nSource token insertion statistics:")
        print(f"  Mean: {insertion_counts.mean():.2f}")
        print(f"  Median: {np.median(insertion_counts):.2f}")
        print(f"  Std: {insertion_counts.std():.2f}")
        print(f"  Min: {insertion_counts.min()}")
        print(f"  Max: {insertion_counts.max()}")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"  Percentiles:")
        for p in percentiles:
            val = np.percentile(insertion_counts, p)
            print(f"    {p}th: {val:.1f}")
    
    # Report source sequence usage
    print(f"\nSource sequence usage:")
    print(f"  Total source sequences: {len(source_token_sequences)}")
    print(f"  Sequences fully used: {source_iterator.sequences_used}")
    print(f"  Cycles completed: {source_iterator.cycles_completed}")
    print(f"  Current sequence index: {source_iterator.current_seq_idx}")
    print(f"  Current position in sequence: {source_iterator.current_pos}")
    print(f"  Total tokens extracted: {source_iterator.total_extracted}")
    
    # Compare actual vs estimated
    print(f"\n{'='*60}")
    print("Source Token Count Statistics (Actual vs Estimated)")
    print(f"{'='*60}")
    
    comparison = []
    for token_id in sorted(actual_token_counts.keys()):
        actual = actual_token_counts[token_id]
        estimated = estimated_counts.get(token_id, 0)
        diff_pct = ((actual - estimated) / estimated * 100) if estimated > 0 else 0
        comparison.append({
            'token_id': token_id,
            'actual': actual,
            'estimated': estimated,
            'difference_pct': diff_pct
        })
    
    # Print summary
    total_actual = sum(actual_token_counts.values())
    total_estimated = sum(estimated_counts.values())
    print(f"  Total actual tokens: {total_actual}")
    print(f"  Total estimated tokens: {total_estimated:.0f}")
    print(f"  Difference: {total_actual - total_estimated:.0f} ({((total_actual - total_estimated) / total_estimated * 100) if total_estimated > 0 else 0:.1f}%)")
    
    # Top 10 most frequent tokens
    sorted_tokens = sorted(comparison, key=lambda x: x['actual'], reverse=True)[:10]
    print(f"\n  Top 10 most frequent source tokens:")
    for item in sorted_tokens:
        diff = item['actual'] - item['estimated']
        print(f"    Token {item['token_id']}: actual={item['actual']}, estimated={item['estimated']:.1f}, diff={diff:.1f} ({item['difference_pct']:.1f}%)")
    
    return comparison


# ============================================================================
# Main Dataset Preparation Function
# ============================================================================

def prepare_combined_dataset(
    target_dataset,
    source_df,
    tokenizer,
    node_to_token_id,
    min_source_per_seq,
    max_source_per_seq=None,
    source_ratio=0.5,
    max_length=None,
    source_start_token_id=None,
    power_law_alpha=2.0,
    seed=42,
    batch_size=10000
):
    """
    Prepare combined dataset with target and source sequences.
    
    This is the main function that orchestrates the entire process:
    1. Estimates expected token counts
    2. Processes target sequences (inserts source tokens)
    3. Processes pure source sequences
    4. Combines and shuffles everything
    5. Returns dataset and statistics
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    
    print(f"\nPreparing combined dataset...")
    print(f"  Source token distribution: Power law (alpha={power_law_alpha})")
    print(f"  Min source tokens per target seq: {min_source_per_seq}")
    
    # Set reasonable default for max_source_per_seq if not provided
    if max_source_per_seq is None:
        if max_length is not None:
            max_source_per_seq = max(int(max_length * 0.2), min_source_per_seq * 10)
        else:
            max_source_per_seq = max(50, min_source_per_seq * 10)
        print(f"  Max source tokens per target seq: {max_source_per_seq} (auto-set)")
    else:
        print(f"  Max source tokens per target seq: {max_source_per_seq}")
    
    print(f"  Source sequence ratio: {source_ratio}")
    print(f"  Max length: {max_length or 'unlimited'}")
    
    # Estimate source token counts before processing
    num_target_sequences = len(target_dataset)
    print(f"\nEstimating source token counts...")
    estimated_counts, estimation_details = estimate_source_token_counts(
        num_target_sequences, node_to_token_id, min_source_per_seq, max_source_per_seq,
        power_law_alpha, source_ratio, source_start_token_id, source_df
    )
    
    print(f"  Estimated average tokens per target seq: {estimation_details['avg_tokens_per_seq']:.2f}")
    print(f"  Estimated total tokens inserted: {estimation_details['total_tokens_inserted']:.0f}")
    print(f"  Estimated pure source sequences: {estimation_details['num_pure_source']}")
    if source_start_token_id is not None:
        print(f"  Estimated SOURCE_START count: {estimation_details['source_start_count']:.0f}")
    
    # Initialize actual token counters
    actual_token_counts = {}
    source_token_ids_set = set()
    for node_id, token_id in node_to_token_id.items():
        actual_token_counts[token_id] = 0
        source_token_ids_set.add(token_id)
    if source_start_token_id is not None:
        actual_token_counts[source_start_token_id] = 0
        source_token_ids_set.add(source_start_token_id)
    
    # Prepare source sequences for sequential extraction
    print(f"\nPreparing source sequences for sequential extraction...")
    source_token_sequences = []
    for idx, row in tqdm(source_df.iterrows(), total=len(source_df), desc="Converting source sequences"):
        sequence = str(row['sequence_labels'])
        node_ids = [int(x) for x in sequence.split()]
        token_ids = [node_to_token_id[node_id] for node_id in node_ids]
        source_token_sequences.append(token_ids)
    
    print(f"  Prepared {len(source_token_sequences)} source sequences")
    
    # Create iterator for sequential extraction from source sequences
    source_iterator = SourceSequenceIterator(source_token_sequences, rng)
    
    # Process target sequences
    print(f"\nProcessing {len(target_dataset)} target sequences...")
    print(f"  Using batch size: {batch_size}")
    target_batches, insertion_counts = _process_target_sequences(
        target_dataset, source_iterator, tokenizer, node_to_token_id,
        source_token_ids_set, actual_token_counts, min_source_per_seq,
        max_source_per_seq, power_law_alpha, max_length, source_start_token_id,
        rng, batch_size
    )
    
    print(f"  Processed into {len(target_batches)} batches")
    
    # Process pure source sequences
    print(f"\nProcessing {len(source_df)} source sequences...")
    source_sequences = _process_pure_source_sequences(
        source_df, node_to_token_id, source_token_ids_set, actual_token_counts,
        max_length, source_start_token_id, source_ratio, num_target_sequences, rng
    )
    
    # Combine sequences
    print(f"\nCombining sequences in batches...")
    print(f"  Total target sequences: {sum(len(batch) for batch in target_batches)}")
    print(f"  Pure source sequences: {len(source_sequences)}")
    
    dataset_dict = _create_combined_dataset(target_batches, source_sequences, seed)
    
    print(f"\nDataset size:")
    print(f"  Total: {len(dataset_dict['train'])} samples")
    
    # Print statistics
    comparison = _print_statistics(
        actual_token_counts, estimated_counts, insertion_counts,
        source_iterator, source_token_sequences
    )
    
    # Check if all estimated counts are the same (excluding SOURCE_START which might differ)
    source_token_estimates = [v for k, v in estimated_counts.items() 
                             if k != source_start_token_id] if source_start_token_id is not None else list(estimated_counts.values())
    
    if len(set(source_token_estimates)) == 1 and len(source_token_estimates) > 0:
        # All source tokens have the same estimated count, save just one value
        estimated_count_value = source_token_estimates[0] if source_token_estimates else 0
        estimated_counts_saved = {'uniform_estimate': float(estimated_count_value)}
        if source_start_token_id is not None:
            estimated_counts_saved['source_start_token'] = {
                'token_id': int(source_start_token_id),
                'estimated_count': float(estimated_counts.get(source_start_token_id, 0))
            }
    else:
        # Different estimates, save per token
        estimated_counts_saved = {str(k): float(v) for k, v in estimated_counts.items()}
    
    # Store statistics in return value
    statistics = {
        'estimated_counts': estimated_counts_saved,
        'actual_counts': {str(k): int(v) for k, v in actual_token_counts.items()},
        'estimation_details': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in estimation_details.items()},
        'comparison': comparison
    }
    
    return dataset_dict, statistics


# ============================================================================
# Utility Functions
# ============================================================================

def save_tokenizer_and_vocab_info(tokenizer, node_to_token_id, output_dir):
    """Save extended tokenizer and vocabulary information"""
    print(f"\nSaving tokenizer to {output_dir}")
    tokenizer.save_pretrained(output_dir)
    
    vocab_info = {
        'vocab_size': len(tokenizer),
        'original_vocab_size': len(tokenizer) - len(node_to_token_id),
        'source_token_count': len(node_to_token_id),
        'node_to_token_id': {str(k): v for k, v in node_to_token_id.items()}
    }
    
    vocab_path = Path(output_dir) / 'vocab_info.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab_info, f, indent=2)
    print(f"Vocabulary info saved to {vocab_path}")


def load_target_dataset_with_splits(dataset_name, dataset_config, dataset_split, max_target_samples):
    """
    Load target dataset, handling multiple splits if available.
    
    Returns:
        target_train, target_val, use_original_splits
    """
    print(f"\nChecking available splits in target dataset...")
    try:
        full_dataset = load_dataset(dataset_name, dataset_config, streaming=False)
        
        if isinstance(full_dataset, dict):
            available_splits = list(full_dataset.keys())
            print(f"  Available splits: {available_splits}")
            
            has_train = 'train' in available_splits
            has_val = 'validation' in available_splits or 'val' in available_splits or 'test' in available_splits
            
            if has_train and has_val:
                val_split_name = 'validation' if 'validation' in available_splits else ('val' if 'val' in available_splits else 'test')
                target_train = full_dataset['train']
                target_val = full_dataset[val_split_name]
                
                print(f"  Using original dataset splits:")
                print(f"    Train: {len(target_train)} samples")
                print(f"    {val_split_name.capitalize()}: {len(target_val)} samples")
                
                # Limit samples if specified
                if max_target_samples:
                    if len(target_train) > max_target_samples:
                        print(f"  Limiting train to {max_target_samples} samples")
                        target_train = target_train.select(range(max_target_samples))
                    val_ratio = len(target_val) / (len(full_dataset['train']) + len(target_val))
                    max_val_samples = int(max_target_samples * val_ratio / (1 - val_ratio))
                    if len(target_val) > max_val_samples:
                        print(f"  Limiting {val_split_name} to {max_val_samples} samples")
                        target_val = target_val.select(range(max_val_samples))
                
                return target_train, target_val, True
            else:
                print(f"  Only one split available, using: {dataset_split}")
                target_train = full_dataset.get(dataset_split, full_dataset[available_splits[0]])
                if max_target_samples and len(target_train) > max_target_samples:
                    target_train = target_train.select(range(max_target_samples))
                return target_train, None, False
        else:
            target_train = full_dataset
            if max_target_samples and len(target_train) > max_target_samples:
                target_train = target_train.select(range(max_target_samples))
            return target_train, None, False
    except Exception as e:
        print(f"  Could not load full dataset, using specified split: {dataset_split}")
        print(f"  Error: {e}")
        target_train = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=False)
        if max_target_samples and len(target_train) > max_target_samples:
            target_train = target_train.select(range(max_target_samples))
        return target_train, None, False


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Combine target and source datasets')
    
    # Target dataset arguments
    parser.add_argument('--target_dataset_name', type=str, required=True,
                      help='HuggingFace dataset name (e.g., "roneneldan/TinyStories")')
    parser.add_argument('--target_dataset_config', type=str, default=None,
                      help='Dataset config name (optional)')
    parser.add_argument('--target_dataset_split', type=str, default='train',
                      help='Dataset split to use')
    parser.add_argument('--target_dataset_text_field', type=str, default='text',
                      help='Field name containing text in dataset')
    parser.add_argument('--target_tokenizer_name', type=str, default=None,
                      help='Tokenizer name (default: same as dataset or "gpt2")')
    parser.add_argument('--max_target_samples', type=int, default=None,
                      help='Maximum number of target samples to use (None = all)')
    
    # Source dataset arguments
    parser.add_argument('--source_csv', type=str, required=True,
                      help='Path to source walks CSV file')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for combined dataset')
    
    # Configuration arguments
    parser.add_argument('--min_source_per_seq', type=int, default=1,
                      help='Minimum source tokens per target sequence')
    parser.add_argument('--max_source_per_seq', type=int, default=None,
                      help='Maximum source tokens per target sequence (None = unlimited)')
    parser.add_argument('--source_ratio', type=float, default=0.5,
                      help='Ratio of pure source sequences in final dataset')
    parser.add_argument('--max_length', type=int, default=None,
                      help='Maximum sequence length (truncate if longer)')
    parser.add_argument('--power_law_alpha', type=float, default=2.0,
                      help='Power law exponent for source token distribution (higher = more long-tailed)')
    parser.add_argument('--source_token_prefix', type=str, default='<GRAPH_',
                      help='Prefix for source tokens (e.g., "<GRAPH_0>")')
    parser.add_argument('--source_token_start', type=str, default='<SOURCE_START>',
                      help='Special token to prepend before source tokens')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--batch_size', type=int, default=10000,
                      help='Batch size for processing sequences (smaller = less memory)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Combined Dataset Preparation")
    print("="*60)
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load source walks
    source_df, source_node_ids = load_source_walks(args.source_csv)
    
    # Load target dataset
    target_train, target_val, use_original_splits = load_target_dataset_with_splits(
        args.target_dataset_name, args.target_dataset_config,
        args.target_dataset_split, args.max_target_samples
    )
    
    # Load tokenizer
    tokenizer_name = args.target_tokenizer_name or args.target_dataset_name
    print(f"\nLoading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except:
        print(f"Failed to load tokenizer from {tokenizer_name}, using GPT-2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Extend tokenizer with source tokens
    tokenizer, node_to_token_id, source_start_token_id = extend_tokenizer_vocab(
        tokenizer, source_node_ids, args.source_token_prefix, args.source_token_start
    )
    
    # Process train split
    print(f"\n{'='*60}")
    print("Processing TRAIN split")
    print(f"{'='*60}")
    train_combined, train_statistics = prepare_combined_dataset(
        target_train, source_df, tokenizer, node_to_token_id,
        args.min_source_per_seq, args.max_source_per_seq, args.source_ratio,
        args.max_length, source_start_token_id, args.power_law_alpha,
        args.seed, args.batch_size
    )
    
    # Process validation split if available
    if use_original_splits and target_val is not None:
        print(f"\n{'='*60}")
        print("Processing VALIDATION split")
        print(f"{'='*60}")
        val_combined, val_statistics = prepare_combined_dataset(
            target_val, source_df, tokenizer, node_to_token_id,
            args.min_source_per_seq, args.max_source_per_seq, args.source_ratio,
            args.max_length, source_start_token_id, args.power_law_alpha,
            args.seed + 1, args.batch_size  # Different seed for validation
        )
        dataset_dict = DatasetDict({
            'train': train_combined['train'],
            'validation': val_combined['train']
        })
        print(f"\n{'='*60}")
        print("Final Dataset Summary")
        print(f"{'='*60}")
        print(f"  Train: {len(dataset_dict['train'])} samples")
        print(f"  Validation: {len(dataset_dict['validation'])} samples")
        print(f"  Using original dataset split ratio")
    else:
        print(f"\n  No original validation split found, creating 90/10 split...")
        dataset = train_combined['train']
        split_dataset = dataset.train_test_split(train_size=0.9, seed=args.seed)
        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })
        val_statistics = {
            'estimated_counts': {},
            'actual_counts': {},
            'estimation_details': {},
            'comparison': []
        }
        print(f"\n{'='*60}")
        print("Final Dataset Summary")
        print(f"{'='*60}")
        print(f"  Train: {len(dataset_dict['train'])} samples")
        print(f"  Validation: {len(dataset_dict['validation'])} samples")
        print(f"  Created new 90/10 split")
    
    # Save dataset
    print(f"\nSaving dataset to {args.output_dir}")
    dataset_dict.save_to_disk(args.output_dir)
    print(f"Dataset saved successfully!")
    
    # Save tokenizer and vocab info
    save_tokenizer_and_vocab_info(tokenizer, node_to_token_id, args.output_dir)
    
    # Save source token statistics
    statistics_path = Path(args.output_dir) / 'source_token_statistics.json'
    statistics_data = {
        'train': train_statistics,
        'validation': val_statistics if use_original_splits and target_val is not None else None
    }
    with open(statistics_path, 'w') as f:
        json.dump(statistics_data, f, indent=2)
    print(f"\nSource token statistics saved to {statistics_path}")
    
    # Show sample
    print(f"\n{'='*60}")
    print("Sample from combined dataset:")
    print(f"{'='*60}")
    sample = dataset_dict['train'][0]
    print(f"Length: {sample['length']}")
    print(f"Input IDs (first 30): {sample['input_ids'][:30]}")
    
    try:
        decoded = tokenizer.decode(sample['input_ids'][:30], skip_special_tokens=False)
        print(f"Decoded (first 30 tokens): {decoded[:200]}...")
    except:
        print("(Could not decode sample)")
    
    print(f"\n{'='*60}")
    print("Combined dataset preparation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
