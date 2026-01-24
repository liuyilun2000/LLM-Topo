"""
View examples from combined dataset

This script loads the combined dataset and displays sample sequences,
showing how source tokens are inserted into natural language text.
"""
import argparse
import json
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='View combined dataset examples')
    parser.add_argument('--dataset_dir', type=str, required=True,
                      help='Directory with combined dataset')
    parser.add_argument('--num_examples', type=int, default=10,
                      help='Number of examples to display')
    parser.add_argument('--split', type=str, default='train',
                      choices=['train', 'validation'],
                      help='Dataset split to view')
    parser.add_argument('--max_length', type=int, default=200,
                      help='Maximum length to display per example')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Combined Dataset Viewer")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)
    
    if args.split not in dataset:
        print(f"Error: Split '{args.split}' not found in dataset")
        print(f"Available splits: {list(dataset.keys())}")
        return
    
    split_dataset = dataset[args.split]
    print(f"  Split: {args.split}")
    print(f"  Total samples: {len(split_dataset)}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.dataset_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.dataset_dir)
        print(f"  Tokenizer vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"  Error loading tokenizer: {e}")
        print("  Will display token IDs only")
        tokenizer = None
    
    # Load vocab info if available
    vocab_info_path = Path(args.dataset_dir) / 'vocab_info.json'
    source_token_info = None
    if vocab_info_path.exists():
        with open(vocab_info_path, 'r') as f:
            vocab_info = json.load(f)
        source_token_info = {
            'source_token_count': vocab_info.get('source_token_count', 0),
            'node_to_token_id': vocab_info.get('node_to_token_id', {})
        }
        print(f"  Source tokens: {source_token_info['source_token_count']}")
    
    # Display examples
    print(f"\n{'='*80}")
    print(f"Displaying {args.num_examples} examples from {args.split} split")
    print(f"{'='*80}\n")
    
    for i in range(min(args.num_examples, len(split_dataset))):
        example = split_dataset[i]
        input_ids = example['input_ids']
        length = example.get('length', len(input_ids))
        
        print(f"Example {i+1}:")
        print(f"  Length: {length} tokens")
        print(f"  Token IDs (first 50): {input_ids[:50]}")
        
        # Decode if tokenizer available
        if tokenizer:
            # Decode full sequence
            try:
                decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
                # Truncate for display
                if len(decoded_full) > args.max_length:
                    decoded_display = decoded_full[:args.max_length] + "..."
                else:
                    decoded_display = decoded_full
                print(f"  Decoded text (first {args.max_length} chars):")
                print(f"    {decoded_display}")
                
                # Count source tokens
                source_token_count = 0
                source_start_count = 0
                for token_id in input_ids:
                    try:
                        token_str = tokenizer.decode([token_id])
                        if token_str.startswith('<GRAPH_'):
                            source_token_count += 1
                        elif token_str == '<SOURCE_START>':
                            source_start_count += 1
                    except:
                        pass
                
                if source_token_count > 0 or source_start_count > 0:
                    print(f"  Source tokens: {source_token_count} graph tokens, {source_start_count} start tokens")
                
            except Exception as e:
                print(f"  Error decoding: {e}")
        
        # Show token breakdown
        if source_token_info:
            graph_token_ids = set()
            for node_id, token_id in source_token_info['node_to_token_id'].items():
                graph_token_ids.add(int(token_id))
            
            graph_tokens_in_seq = [tid for tid in input_ids if tid in graph_token_ids]
            if graph_tokens_in_seq:
                print(f"  Graph token IDs in sequence: {graph_tokens_in_seq[:20]}{'...' if len(graph_tokens_in_seq) > 20 else ''}")
        
        print()
    
    # Statistics
    print(f"{'='*80}")
    print("Dataset Statistics:")
    print(f"{'='*80}")
    
    # Calculate average length
    lengths = [ex.get('length', len(ex['input_ids'])) for ex in split_dataset]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0
    
    print(f"  Average sequence length: {avg_length:.1f} tokens")
    print(f"  Min length: {min_length} tokens")
    print(f"  Max length: {max_length} tokens")
    
    # Count sequences with source tokens
    if tokenizer:
        sequences_with_source = 0
        total_source_tokens = 0
        for example in split_dataset:
            input_ids = example['input_ids']
            has_source = False
            for token_id in input_ids:
                try:
                    token_str = tokenizer.decode([token_id])
                    if token_str.startswith('<GRAPH_') or token_str == '<SOURCE_START>':
                        has_source = True
                        total_source_tokens += 1
                except:
                    pass
            if has_source:
                sequences_with_source += 1
        
        print(f"  Sequences with source tokens: {sequences_with_source} / {len(split_dataset)} ({sequences_with_source/len(split_dataset)*100:.1f}%)")
        print(f"  Total source tokens in split: {total_source_tokens}")
    
    print(f"\n{'='*80}")
    print("View complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

