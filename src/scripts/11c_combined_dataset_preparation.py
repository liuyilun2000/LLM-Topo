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
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random


def load_source_walks(csv_path):
    """Load source graph walk sequences from CSV"""
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
        # For streaming, we'll need to handle it differently
        print("Streaming mode: will process in batches")
        return dataset, None
    else:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        print(f"Loaded {len(dataset)} samples")
        return dataset, len(dataset)


def extend_tokenizer_vocab(tokenizer, source_node_ids, source_token_prefix="<GRAPH_"):
    """
    Extend tokenizer vocabulary with source tokens
    
    Args:
        tokenizer: Base tokenizer to extend
        source_node_ids: List of node IDs to add as tokens
        source_token_prefix: Prefix for source tokens (e.g., "<GRAPH_0>", "<GRAPH_1>")
    
    Returns:
        Extended tokenizer, mapping from node_id to token_id
    """
    print(f"\nExtending tokenizer vocabulary with {len(source_node_ids)} source tokens...")
    
    # Get current vocab size
    original_vocab_size = len(tokenizer)
    print(f"Original vocab size: {original_vocab_size}")
    
    # Create source tokens
    source_tokens = []
    node_to_token_id = {}
    
    for node_id in source_node_ids:
        token_str = f"{source_token_prefix}{node_id}>"
        source_tokens.append(token_str)
        # We'll map after adding to tokenizer
    
    # Add tokens to tokenizer
    tokenizer.add_tokens(source_tokens, special_tokens=False)
    
    # Build mapping from node_id to token_id
    for node_id in source_node_ids:
        token_str = f"{source_token_prefix}{node_id}>"
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        node_to_token_id[node_id] = token_id
    
    new_vocab_size = len(tokenizer)
    print(f"New vocab size: {new_vocab_size} (+{new_vocab_size - original_vocab_size})")
    
    return tokenizer, node_to_token_id


def insert_source_tokens(target_token_ids, source_token_ids, min_source_count, max_insertions_per_seq=None, rng=None):
    """
    Insert source tokens into a target sequence
    
    Args:
        target_token_ids: List of token IDs from target text
        source_token_ids: List of source token IDs to choose from
        min_source_count: Minimum number of source tokens to insert
        max_insertions_per_seq: Maximum source tokens per sequence (None = no limit)
        rng: Random number generator
    
    Returns:
        Combined sequence with source tokens inserted
    """
    if rng is None:
        rng = random
    
    # Determine how many source tokens to insert
    num_insertions = min_source_count
    if max_insertions_per_seq is not None:
        num_insertions = min(num_insertions, max_insertions_per_seq)
    
    # If we need more than available, use all available
    num_insertions = min(num_insertions, len(source_token_ids))
    
    if num_insertions == 0:
        return target_token_ids
    
    # Sample source tokens to insert
    tokens_to_insert = rng.choices(source_token_ids, k=num_insertions)
    
    # Insert at random positions
    combined = target_token_ids.copy()
    for token in tokens_to_insert:
        # Insert at random position (including at the end)
        insert_pos = rng.randint(0, len(combined))
        combined.insert(insert_pos, token)
    
    return combined


def prepare_combined_dataset(
    target_dataset,
    source_df,
    tokenizer,
    node_to_token_id,
    min_source_per_seq,
    max_source_per_seq=None,
    source_ratio=0.5,
    max_length=None,
    seed=42
):
    """
    Prepare combined dataset with target and source sequences
    
    Args:
        target_dataset: HuggingFace dataset with target text
        source_df: DataFrame with source walks
        tokenizer: Extended tokenizer
        node_to_token_id: Mapping from node_id to token_id
        min_source_per_seq: Minimum source tokens per target sequence
        max_source_per_seq: Maximum source tokens per target sequence
        source_ratio: Ratio of pure source sequences in final dataset
        max_length: Maximum sequence length (truncate if needed)
        seed: Random seed
    
    Returns:
        Combined DatasetDict
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    
    print(f"\nPreparing combined dataset...")
    print(f"  Min source tokens per target seq: {min_source_per_seq}")
    print(f"  Max source tokens per target seq: {max_source_per_seq or 'unlimited'}")
    print(f"  Source sequence ratio: {source_ratio}")
    print(f"  Max length: {max_length or 'unlimited'}")
    
    # Get all source token IDs
    source_token_ids = list(node_to_token_id.values())
    
    # Process target sequences
    print(f"\nProcessing {len(target_dataset)} target sequences...")
    target_sequences = []
    
    for example in tqdm(target_dataset, desc="Processing target sequences"):
        # Get text and tokenize
        if 'input_ids' in example:
            # Already tokenized
            target_token_ids = example['input_ids']
        elif 'text' in example:
            # Need to tokenize
            text = example['text']
            tokenized = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
            target_token_ids = tokenized['input_ids']
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
            target_token_ids = tokenized['input_ids']
        
        # Insert source tokens
        combined_ids = insert_source_tokens(
            target_token_ids,
            source_token_ids,
            min_source_per_seq,
            max_source_per_seq,
            rng
        )
        
        # Truncate if needed
        if max_length and len(combined_ids) > max_length:
            combined_ids = combined_ids[:max_length]
        
        target_sequences.append({
            'input_ids': combined_ids,
            'length': len(combined_ids)
        })
    
    # Process source sequences
    print(f"\nProcessing {len(source_df)} source sequences...")
    source_sequences = []
    
    for idx, row in tqdm(source_df.iterrows(), total=len(source_df), desc="Processing source sequences"):
        # Convert node IDs to token IDs
        sequence = str(row['sequence_labels'])
        node_ids = [int(x) for x in sequence.split()]
        token_ids = [node_to_token_id[node_id] for node_id in node_ids]
        
        # Truncate if needed
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        source_sequences.append({
            'input_ids': token_ids,
            'length': len(token_ids)
        })
    
    # Combine with ratio control
    print(f"\nCombining sequences...")
    
    # Apply source ratio: determine how many source sequences to include
    num_target = len(target_sequences)
    if source_ratio > 0:
        # Calculate target number of source sequences based on ratio
        # If ratio = 0.5, we want equal numbers of target and source
        target_source = int(num_target * source_ratio / (1 - source_ratio))
        if len(source_sequences) > target_source:
            # Sample source sequences
            indices = rng.choice(len(source_sequences), size=target_source, replace=False)
            source_sequences = [source_sequences[i] for i in indices]
            print(f"  Sampling {target_source} source sequences (ratio: {source_ratio})")
        else:
            print(f"  Using all {len(source_sequences)} source sequences (requested ratio would need {target_source})")
    
    all_sequences = target_sequences + source_sequences
    
    # Shuffle
    rng.shuffle(all_sequences)
    
    print(f"  Total sequences: {len(all_sequences)}")
    print(f"    Target (with source inserted): {len(target_sequences)}")
    print(f"    Pure source: {len(source_sequences)}")
    
    # Create dataset
    dataset = Dataset.from_list(all_sequences)
    
    # Split into train/validation (90/10)
    split_dataset = dataset.train_test_split(train_size=0.9, seed=seed)
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test']
    })
    
    print(f"\nDataset split:")
    print(f"  Train: {len(dataset_dict['train'])} samples")
    print(f"  Validation: {len(dataset_dict['validation'])} samples")
    
    return dataset_dict


def save_tokenizer_and_vocab_info(tokenizer, node_to_token_id, output_dir):
    """Save extended tokenizer and vocabulary information"""
    print(f"\nSaving tokenizer to {output_dir}")
    tokenizer.save_pretrained(output_dir)
    
    # Save vocabulary info
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


def main():
    parser = argparse.ArgumentParser(description='Combine target and source datasets')
    
    # Target dataset
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
    
    # Source dataset
    parser.add_argument('--source_csv', type=str, required=True,
                      help='Path to source walks CSV file')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for combined dataset')
    
    # Configuration
    parser.add_argument('--min_source_per_seq', type=int, default=5,
                      help='Minimum source tokens per target sequence')
    parser.add_argument('--max_source_per_seq', type=int, default=None,
                      help='Maximum source tokens per target sequence (None = unlimited)')
    parser.add_argument('--source_ratio', type=float, default=0.5,
                      help='Ratio of pure source sequences in final dataset')
    parser.add_argument('--max_length', type=int, default=None,
                      help='Maximum sequence length (truncate if longer)')
    parser.add_argument('--source_token_prefix', type=str, default='<GRAPH_',
                      help='Prefix for source tokens (e.g., "<GRAPH_0>")')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
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
    target_dataset = load_dataset(
        args.target_dataset_name,
        args.target_dataset_config,
        split=args.target_dataset_split,
        streaming=False
    )
    
    # Limit target samples if specified
    if args.max_target_samples and len(target_dataset) > args.max_target_samples:
        print(f"Limiting target dataset to {args.max_target_samples} samples")
        target_dataset = target_dataset.select(range(args.max_target_samples))
    
    # Load tokenizer
    tokenizer_name = args.target_tokenizer_name or args.target_dataset_name
    print(f"\nLoading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except:
        # Fallback to GPT-2 tokenizer
        print(f"Failed to load tokenizer from {tokenizer_name}, using GPT-2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Extend tokenizer with source tokens
    tokenizer, node_to_token_id = extend_tokenizer_vocab(
        tokenizer,
        source_node_ids,
        args.source_token_prefix
    )
    
    # Prepare combined dataset
    dataset_dict = prepare_combined_dataset(
        target_dataset,
        source_df,
        tokenizer,
        node_to_token_id,
        args.min_source_per_seq,
        args.max_source_per_seq,
        args.source_ratio,
        args.max_length,
        args.seed
    )
    
    # Save dataset
    print(f"\nSaving dataset to {args.output_dir}")
    dataset_dict.save_to_disk(args.output_dir)
    print(f"Dataset saved successfully!")
    
    # Save tokenizer and vocab info
    save_tokenizer_and_vocab_info(tokenizer, node_to_token_id, args.output_dir)
    
    # Show sample
    print(f"\n{'='*60}")
    print("Sample from combined dataset:")
    print(f"{'='*60}")
    sample = dataset_dict['train'][0]
    print(f"Length: {sample['length']}")
    print(f"Input IDs (first 30): {sample['input_ids'][:30]}")
    
    # Decode sample (may contain source tokens)
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

