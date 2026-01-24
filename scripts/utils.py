"""
Utility functions for loading token representations and model metadata
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def load_representations(representation_dir: str) -> tuple[Dict[str, np.ndarray], List[Dict]]:
    """
    Load token representation files.
    
    Args:
        representation_dir: Path to directory containing token_representations.npz and token_metadata.json
    
    Returns:
        representations: Dict mapping representation names to numpy arrays
        metadata: List of token metadata dictionaries
    """
    representation_dir = Path(representation_dir)
    representation_file = representation_dir / 'token_representations.npz'
    metadata_file = representation_dir / 'token_metadata.json'
    
    if not representation_file.exists():
        raise FileNotFoundError(f"Representation file not found: {representation_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    print(f"Loading representations from {representation_file}")
    data = np.load(representation_file)
    
    print(f"Loading metadata from {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    representations = {key: data[key] for key in data.keys()}
    
    print(f"\nLoaded {len(representations)} representations:")
    for key, arr in representations.items():
        print(f"  {key}: {arr.shape}")
    
    return representations, metadata


def load_summary(representation_dir: str) -> Optional[Dict]:
    """
    Load extraction summary information.
    
    Args:
        representation_dir: Path to directory containing extraction_summary.json
    
    Returns:
        summary: Dict with model and extraction info, or None if not found
    """
    summary_file = Path(representation_dir) / 'extraction_summary.json'
    if not summary_file.exists():
        return None
    
    with open(summary_file, 'r') as f:
        return json.load(f)


def load_vocab_size(representation_dir: str) -> Optional[int]:
    """
    Load vocab size from extraction summary.
    
    Args:
        representation_dir: Path to directory containing extraction_summary.json
    
    Returns:
        vocab_size: Integer vocabulary size, or None if not found
    """
    summary = load_summary(representation_dir)
    return summary.get('vocab_size') if summary else None

