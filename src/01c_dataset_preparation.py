"""
Script to convert CSV graph walk data to HuggingFace Dataset format
"""
import argparse
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from pathlib import Path


def load_csv_data(csv_path):
    """Load graph walk data from CSV"""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} walks")
    return df


def analyze_vocabulary(df):
    """Analyze the vocabulary from the dataset"""
    all_ids = []
    for seq in df['sequence_labels']:
        ids = [int(x) for x in str(seq).split()]
        all_ids.extend(ids)
    
    max_id = max(all_ids)
    min_id = min(all_ids)
    vocab_size = max_id + 1
    
    print(f"\nVocabulary statistics:")
    print(f"  Min ID: {min_id}")
    print(f"  Max ID: {max_id}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Total tokens: {len(all_ids)}")
    
    return vocab_size, min_id, max_id


def prepare_dataset(df, train_split=0.9):
    """Convert DataFrame to HuggingFace Dataset"""
    # Convert sequences to list of token IDs
    data = []
    for idx, row in df.iterrows():
        sequence = str(row['sequence_labels'])
        token_ids = [int(x) for x in sequence.split()]
        data.append({
            'walk_id': int(row['walk_id']),
            'length': int(row['length']),
            'input_ids': token_ids
        })
    
    # Create dataset
    dataset = Dataset.from_list(data)
    
    # Split into train/validation
    if train_split < 1.0:
        split_dataset = dataset.train_test_split(train_size=train_split, seed=42)
        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })
        print(f"\nDataset split:")
        print(f"  Train: {len(dataset_dict['train'])} samples")
        print(f"  Validation: {len(dataset_dict['validation'])} samples")
        return dataset_dict
    else:
        dataset_dict = DatasetDict({'train': dataset})
        print(f"\nDataset created:")
        print(f"  Train: {len(dataset_dict['train'])} samples")
        return dataset_dict


def save_vocab_info(vocab_size, min_id, max_id, output_dir):
    """Save vocabulary information to JSON"""
    vocab_info = {
        'vocab_size': vocab_size,
        'min_id': min_id,
        'max_id': max_id
    }
    
    vocab_path = Path(output_dir) / 'vocab_info.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab_info, f, indent=2)
    print(f"\nVocabulary info saved to {vocab_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare graph walk dataset')
    parser.add_argument('--input_csv', type=str, default='dataset_walks_torus.csv',
                      help='Input CSV file path')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                      help='Output directory for HuggingFace dataset')
    parser.add_argument('--train_split', type=float, default=0.9,
                      help='Fraction of data for training (default: 0.9)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Graph Walk Dataset Preparation")
    print("="*60)
    
    # Load CSV data
    df = load_csv_data(args.input_csv)
    
    # Analyze vocabulary
    vocab_size, min_id, max_id = analyze_vocabulary(df)
    
    # Prepare dataset
    dataset = prepare_dataset(df, args.train_split)
    
    # Save dataset
    print(f"\nSaving dataset to {args.output_dir}")
    dataset.save_to_disk(args.output_dir)
    print(f"Dataset saved successfully!")
    
    # Save vocabulary info
    save_vocab_info(vocab_size, min_id, max_id, args.output_dir)
    
    # Show sample
    print(f"\n{'='*60}")
    print("Sample from dataset:")
    print(f"{'='*60}")
    sample = dataset['train'][0]
    print(f"Walk ID: {sample['walk_id']}")
    print(f"Length: {sample['length']}")
    print(f"Input IDs (first 20): {sample['input_ids'][:20]}")
    
    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

