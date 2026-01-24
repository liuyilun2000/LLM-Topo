"""
Extract representations for GRAPH tokens from trained model

This script extracts internal representations for GRAPH tokens (e.g., <GRAPH_0>, 
<GRAPH_1>, ..., <GRAPH_1199>) that were added to the vocabulary during combined 
dataset preparation.

IMPORTANT: This script extracts GRAPH tokens from sequences of the form:
  [SOURCE_START, GRAPH_xxx]
Only the GRAPH token's hidden state (at position 1) is extracted, not SOURCE_START.

Uses token IDs directly to avoid tokenizer interpretation issues.

Supported representation types:
- 'residual_before': Residual stream state before each decoder block (input to block)
- 'after_attention': Hidden state after attention, before FFN
- 'after_block': Hidden state after entire decoder block (attention + FFN + residuals) - default
- 'ffn_gate': FFN gate projection activations
- 'ffn_up': FFN up projection activations
"""
import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from transformers import LlamaForCausalLM, AutoTokenizer


class RepresentationExtractor:
    """Extract internal representations from Llama model"""
    
    def __init__(self, model, representations=None):
        self.model = model
        self.representations = {}
        self.hooks = []
        # Default to after_block if not specified
        self.representations_to_extract = representations if representations else ['after_block']
        
    def register_hooks(self):
        """Register forward hooks to capture representations"""
        
        def get_after_block_hook(layer_idx):
            """Hook on layer output - captures state AFTER entire decoder block"""
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0].detach().cpu()
                else:
                    hidden_states = output.detach().cpu()
                self.representations[f'layer_{layer_idx}_after_block'] = hidden_states
            return hook
        
        def get_residual_before_hook(layer_idx):
            """Hook on layer input - captures residual stream BEFORE decoder block"""
            def hook(module, input, output):
                # input[0] is the hidden states coming into the layer
                if isinstance(input, tuple) and len(input) > 0:
                    hidden_states = input[0].detach().cpu()
                else:
                    hidden_states = input.detach().cpu()
                self.representations[f'layer_{layer_idx}_residual_before'] = hidden_states
            return hook
        
        def get_after_attention_hook(layer_idx):
            """Hook after attention output - captures state after attention, before FFN"""
            def hook(module, input, output):
                # output[0] is attention output
                if isinstance(output, tuple):
                    hidden_states = output[0].detach().cpu()
                else:
                    hidden_states = output.detach().cpu()
                self.representations[f'layer_{layer_idx}_after_attention'] = hidden_states
            return hook
        
        def get_ffn_gate_hook(layer_idx):
            """Hook on FFN gate projection"""
            def hook(module, input, output):
                gate_output = output.detach().cpu()
                self.representations[f'layer_{layer_idx}_ffn_gate'] = gate_output
            return hook
        
        def get_ffn_up_hook(layer_idx):
            """Hook on FFN up projection"""
            def hook(module, input, output):
                up_output = output.detach().cpu()
                self.representations[f'layer_{layer_idx}_ffn_up'] = up_output
            return hook
        
        # Register hooks for each layer based on requested representations
        for layer_idx, layer in enumerate(self.model.model.layers):
            if 'after_block' in self.representations_to_extract:
                handle = layer.register_forward_hook(get_after_block_hook(layer_idx))
                self.hooks.append(handle)
            
            if 'residual_before' in self.representations_to_extract:
                handle = layer.register_forward_hook(get_residual_before_hook(layer_idx))
                self.hooks.append(handle)
            
            if 'after_attention' in self.representations_to_extract:
                handle = layer.self_attn.register_forward_hook(get_after_attention_hook(layer_idx))
                self.hooks.append(handle)
            
            if 'ffn_gate' in self.representations_to_extract:
                handle = layer.mlp.gate_proj.register_forward_hook(get_ffn_gate_hook(layer_idx))
                self.hooks.append(handle)
            
            if 'ffn_up' in self.representations_to_extract:
                handle = layer.mlp.up_proj.register_forward_hook(get_ffn_up_hook(layer_idx))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_representations(self):
        """Clear stored representations"""
        self.representations = {}


def extract_token_representations_from_sequence(model, sequence_token_ids, target_position, representations=None):
    """
    Extract representations for a token at a specific position in a sequence
    
    Args:
        model: LlamaForCausalLM model
        sequence_token_ids: list of token IDs (e.g., [SOURCE_START_ID, GRAPH_ID])
        target_position: int, position in sequence to extract (0-indexed)
        representations: list of representation types to extract
    
    Returns:
        dict with representations for each layer
    """
    extractor = RepresentationExtractor(model, representations)
    extractor.register_hooks()
    
    # Create input tensor for sequence
    input_ids = torch.tensor([sequence_token_ids], dtype=torch.long)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True
        )
    
    reps = extractor.representations.copy()
    extractor.remove_hooks()
    
    # Extract representations at target position
    batch_size, seq_len = input_ids.shape
    token_reps = {}
    
    for key, value in reps.items():
        # Handle different tensor shapes
        if value.dim() == 3:
            # Standard shape: [batch=1, seq_len, dim]
            token_reps[key] = value[0, target_position, :].detach().cpu()  # [dim]
        elif value.dim() == 2:
            # Could be [batch*seq_len, dim] (FFN projections) or [batch, dim]
            # Check if first dimension matches batch*seq_len
            if value.shape[0] == batch_size * seq_len:
                # Reshape to [batch, seq_len, dim] and extract target position
                reshaped = value.view(batch_size, seq_len, -1)
                token_reps[key] = reshaped[0, target_position, :].detach().cpu()  # [dim]
            elif value.shape[0] == seq_len:
                # Sequence-level output [seq_len, dim]
                token_reps[key] = value[target_position, :].detach().cpu()  # [dim]
            else:
                # Single value per batch, just take it (assuming batch=1)
                token_reps[key] = value[0, :].detach().cpu() if value.shape[0] >= 1 else value.squeeze().detach().cpu()
        else:
            # Scalar or unexpected shape, store as-is
            token_reps[key] = value.detach().cpu()
    
    # Also store input embedding and final hidden state at target position
    token_reps['input_embeds'] = outputs.hidden_states[0][0, target_position, :].detach().cpu()  # [hidden_dim]
    token_reps['final_hidden'] = outputs.hidden_states[-1][0, target_position, :].detach().cpu()  # [hidden_dim]
    
    return token_reps


def identify_source_tokens_by_id(vocab_info, tokenizer):
    """
    Identify GRAPH token IDs and SOURCE_START token ID using token IDs directly
    
    Args:
        vocab_info: dict from vocab_info.json
        tokenizer: tokenizer (for verification only, but we use IDs directly)
    
    Returns:
        dict with:
            - graph_token_ids: sorted list of GRAPH token IDs (GRAPH_0 to GRAPH_1199)
            - source_start_token_id: ID of SOURCE_START token (if exists)
            - source_token_map: mapping from node_id to token_id
    """
    source_token_ids = []
    source_start_token_id = None
    source_token_map = {}
    
    # Get node_to_token_id mapping from vocab_info
    node_to_token_id = vocab_info.get('node_to_token_id', {})
    
    # Convert node IDs to token IDs (these are the GRAPH tokens)
    for node_id_str, token_id in node_to_token_id.items():
        source_token_ids.append(int(token_id))
        source_token_map[int(node_id_str)] = int(token_id)
    
    # Find SOURCE_START token ID directly using tokenizer
    # Try multiple methods to find the token ID
    try:
        source_start_token_id = tokenizer.convert_tokens_to_ids('<SOURCE_START>')
        if source_start_token_id == tokenizer.unk_token_id:
            source_start_token_id = None
    except:
        pass
    
    # Also try to find it by checking special tokens
    if source_start_token_id is None:
        try:
            # Check if it's in the added tokens
            added_tokens = tokenizer.get_added_vocab()
            if '<SOURCE_START>' in added_tokens:
                source_start_token_id = added_tokens['<SOURCE_START>']
        except:
            pass
    
    # If still not found, try to infer from vocab size
    # SOURCE_START is typically the last token ID (highest)
    if source_start_token_id is None and source_token_ids:
        max_graph_id = max(source_token_ids)
        # SOURCE_START should be max_graph_id + 1
        candidate_id = max_graph_id + 1
        # Verify by trying to decode
        try:
            decoded = tokenizer.decode([candidate_id])
            if '<SOURCE_START>' in decoded or decoded.strip() == '<SOURCE_START>':
                source_start_token_id = candidate_id
        except:
            pass
    
    # Sort GRAPH token IDs
    graph_token_ids = sorted(source_token_ids)
    
    return {
        'graph_token_ids': graph_token_ids,
        'source_start_token_id': source_start_token_id,
        'source_token_map': source_token_map
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract source token representations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Representation types:
  residual_before  - Residual stream state BEFORE each decoder block (input to block)
  after_attention  - Hidden state AFTER attention, before FFN
  after_block      - Hidden state AFTER entire decoder block (attention + FFN + residuals) [default]
  ffn_gate         - FFN gate projection activations
  ffn_up           - FFN up projection activations

Examples:
  --representations after_block
  --representations residual_before after_attention after_block
  --representations after_block ffn_gate ffn_up
        """
    )
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory with trained model')
    parser.add_argument('--dataset_dir', type=str, required=True,
                      help='Directory with combined dataset (contains vocab_info.json and tokenizer)')
    parser.add_argument('--output_dir', type=str, default='./token_representations',
                      help='Output directory for representations')
    parser.add_argument('--representations', type=str, nargs='+',
                      default=['after_block'],
                      choices=['residual_before', 'after_attention', 'after_block', 'ffn_gate', 'ffn_up'],
                      help='Representation types to extract (default: after_block)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Source Token Representation Extraction")
    print("="*60)
    
    # Load vocab info
    vocab_info_path = Path(args.dataset_dir) / 'vocab_info.json'
    print(f"\nLoading vocabulary info from {vocab_info_path}")
    with open(vocab_info_path, 'r') as f:
        vocab_info = json.load(f)
    
    print(f"  Original vocab size: {vocab_info.get('original_vocab_size', 'unknown')}")
    print(f"  Source token count: {vocab_info.get('source_token_count', 'unknown')}")
    print(f"  Total vocab size: {vocab_info.get('vocab_size', 'unknown')}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.dataset_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.dataset_dir)
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    
    # Identify source tokens using token IDs directly
    print(f"\nIdentifying source tokens by token ID...")
    source_info = identify_source_tokens_by_id(vocab_info, tokenizer)
    graph_token_ids = source_info['graph_token_ids']
    source_start_token_id = source_info['source_start_token_id']
    source_token_map = source_info['source_token_map']
    
    print(f"  Found {len(graph_token_ids)} GRAPH tokens")
    if source_start_token_id is not None:
        print(f"  Found SOURCE_START token: ID {source_start_token_id}")
    else:
        print(f"  ERROR: SOURCE_START token not found!")
        print(f"  Cannot extract GRAPH tokens in context of <SOURCE_START><GRAPH_xxx>")
        sys.exit(1)
    
    print(f"  GRAPH token ID range: {min(graph_token_ids)} to {max(graph_token_ids)}")
    print(f"  Extracting GRAPH tokens from sequences: [SOURCE_START={source_start_token_id}, GRAPH_xxx]")
    
    # Load model
    print(f"\nLoading model from {args.model_dir}")
    model = LlamaForCausalLM.from_pretrained(args.model_dir)
    model.eval()
    
    # Get model info
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    
    print(f"  Num layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    
    print(f"\nRepresentations to extract: {args.representations}")
    print("\nRepresentation locations:")
    if 'residual_before' in args.representations:
        print("  - residual_before: Residual stream BEFORE each decoder block")
    if 'after_attention' in args.representations:
        print("  - after_attention: Hidden state AFTER attention, BEFORE FFN")
    if 'after_block' in args.representations:
        print("  - after_block: Hidden state AFTER entire decoder block (attention + FFN + residuals)")
    if 'ffn_gate' in args.representations:
        print("  - ffn_gate: FFN gate projection activations")
    if 'ffn_up' in args.representations:
        print("  - ffn_up: FFN up projection activations")
    
    # Extract representations for each GRAPH token from <SOURCE_START><GRAPH_xxx> sequences
    print(f"\n{'='*60}")
    print("Extracting GRAPH token representations from sequences...")
    print(f"{'='*60}\n")
    print(f"Extracting from sequences: [SOURCE_START={source_start_token_id}, GRAPH_xxx]")
    print(f"Extracting hidden state at position 1 (GRAPH token position)\n")
    
    all_representations = defaultdict(list)
    token_metadata = []
    
    for graph_token_id in tqdm(graph_token_ids, desc="Processing GRAPH tokens"):
        # Create sequence: [SOURCE_START_ID, GRAPH_ID]
        sequence = [source_start_token_id, graph_token_id]
        
        # Extract representations for GRAPH token at position 1
        token_reps = extract_token_representations_from_sequence(
            model, 
            sequence, 
            target_position=1,  # Position 1 is the GRAPH token
            representations=args.representations
        )
        
        # Store representations
        for key, value in token_reps.items():
            all_representations[key].append(value.numpy())
        
        # Store metadata
        # Get node_id for this graph token
        node_id = None
        for nid, tid in source_token_map.items():
            if tid == graph_token_id:
                node_id = nid
                break
        
        # Try to decode token to get its string representation (for reference only)
        try:
            token_str = tokenizer.decode([graph_token_id])
        except:
            token_str = f"token_{graph_token_id}"
        
        token_metadata.append({
            'token_id': int(graph_token_id),
            'token_str': token_str,
            'node_id': node_id,
            'extraction_sequence': sequence,
            'extraction_position': 1
        })
    
    # Convert to arrays
    print(f"\nConverting to arrays...")
    save_dict = {}
    for key, value_list in all_representations.items():
        save_dict[key] = np.stack(value_list)  # [num_graph_tokens, hidden_dim]
        print(f"  {key}: {save_dict[key].shape}")
    
    # Save representations
    output_file = output_dir / 'token_representations.npz'
    print(f"\nSaving representations to {output_file}")
    np.savez_compressed(output_file, **save_dict)
    
    # Save metadata
    metadata_file = output_dir / 'token_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(token_metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    # Save source token mapping
    mapping_file = output_dir / 'source_token_mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump({
            'source_token_map': {str(k): v for k, v in source_token_map.items()},
            'source_start_token_id': source_start_token_id,
            'graph_token_ids': graph_token_ids,
            'extraction_method': 'sequence_based',
            'extraction_sequence_template': [source_start_token_id, 'GRAPH_TOKEN_ID'],
            'extraction_position': 1
        }, f, indent=2)
    print(f"Saved token mapping to {mapping_file}")
    
    # Save summary
    summary = {
        'model_dir': str(args.model_dir),
        'dataset_dir': str(args.dataset_dir),
        'num_graph_tokens': len(graph_token_ids),
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'representations_extracted': args.representations,
        'representation_keys': list(save_dict.keys()),
        'representation_shape': f'[num_graph_tokens={len(graph_token_ids)}, hidden_dim]',
        'source_start_token_id': source_start_token_id,
        'extraction_method': 'sequence_based',
        'extraction_sequence_template': [source_start_token_id, 'GRAPH_TOKEN_ID'],
        'extraction_position': 1,
        'graph_token_id_range': [min(graph_token_ids), max(graph_token_ids)]
    }
    
    summary_file = output_dir / 'extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"{'='*60}")
    print(f"\nTotal GRAPH tokens processed: {len(graph_token_ids)}")
    print(f"Extracted from sequences: [SOURCE_START={source_start_token_id}, GRAPH_xxx]")
    print(f"Output directory: {output_dir}")
    print(f"\nRepresentation shapes:")
    for key, arr in save_dict.items():
        print(f"  {key}: {arr.shape}")


if __name__ == '__main__':
    main()

