"""
Extract and save internal representations for each token
- Hidden states at different points in the architecture
- FFN activations at each layer

Supported representation types:
- 'residual_before': Residual stream state before each decoder block (input to block)
- 'after_attention': Hidden state after attention, before FFN
- 'after_block': Hidden state after entire decoder block (attention + FFN + residuals) - default
- 'ffn_gate': FFN gate projection activations
- 'ffn_up': FFN up projection activations
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from transformers import LlamaForCausalLM


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


def extract_token_representations(model, token_id, representations=None):
    """
    Extract representations for a single token
    
    Args:
        model: LlamaForCausalLM model
        token_id: int, the token ID to extract
        representations: list of representation types to extract
    
    Returns:
        dict with representations for each layer
    """
    extractor = RepresentationExtractor(model, representations)
    extractor.register_hooks()
    
    # Create input tensor for single token
    input_ids = torch.tensor([[token_id]], dtype=torch.long)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True
        )
    
    reps = extractor.representations.copy()
    extractor.remove_hooks()
    
    # Extract single token representations (squeeze batch and sequence dims)
    token_reps = {}
    for key, value in reps.items():
        # value shape: [batch=1, seq_len=1, dim]
        # We want [dim] by squeezing the first two dimensions
        if value.dim() == 3:
            token_reps[key] = value.squeeze(0).squeeze(0)  # [dim]
        elif value.dim() == 2:
            token_reps[key] = value.squeeze(0)  # [dim]
        else:
            token_reps[key] = value
    
    # Also store input embedding and final hidden state
    token_reps['input_embeds'] = outputs.hidden_states[0].squeeze(0).squeeze(0).detach().cpu()  # [hidden_dim]
    token_reps['final_hidden'] = outputs.hidden_states[-1].squeeze(0).squeeze(0).detach().cpu()  # [hidden_dim]
    
    return token_reps


def main():
    parser = argparse.ArgumentParser(
        description='Extract token representations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Representation types:
  residual_before  - Residual stream state BEFORE each decoder block (input to block)
  after_attention  - Hidden state AFTER attention, before FFN
  after_block      - Hidden state AFTER entire decoder block (attention + FFN + residuals) [default]
  ffn_gate         - FFN gate projection activations
  ffn_up           - FFN up projection activations

Note: 'after_block' is extracted by default. The current default behavior captures
the output after each decoder block (after attention + FFN + all residuals).

Examples:
  --representations after_block
  --representations residual_before after_attention after_block
  --representations after_block ffn_gate ffn_up
        """
    )
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory with trained model')
    parser.add_argument('--output_dir', type=str, default='./token_representations',
                      help='Output directory for representations')
    parser.add_argument('--vocab_file', type=str, default=None,
                      help='Path to vocab.json (default: model_dir/vocab.json)')
    parser.add_argument('--representations', type=str, nargs='+',
                      default=['after_block'],
                      choices=['residual_before', 'after_attention', 'after_block', 'ffn_gate', 'ffn_up'],
                      help='Representation types to extract (default: after_block)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Token Representation Extraction")
    print("="*60)
    
    # Load vocabulary
    if args.vocab_file is None:
        vocab_file = Path(args.model_dir) / 'vocab.json'
    else:
        vocab_file = Path(args.vocab_file)
    
    print(f"\nLoading vocabulary from {vocab_file}")
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    
    token_ids = sorted([int(v) for v in vocab.values()])
    vocab_size = len(token_ids)
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Token IDs: {min(token_ids)} to {max(token_ids)}")
    
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
    
    # Extract representations for each token
    print(f"\n{'='*60}")
    print("Extracting token representations...")
    print(f"{'='*60}\n")
    
    all_representations = defaultdict(list)
    token_metadata = []
    
    for token_id in tqdm(token_ids, desc="Processing tokens"):
        # Extract representations for this token
        token_reps = extract_token_representations(model, token_id, args.representations)
        
        # Store representations
        for key, value in token_reps.items():
            all_representations[key].append(value.numpy())
        
        # Store metadata
        token_metadata.append({
            'token_id': int(token_id),
            'token_str': str(token_id)
        })
    
    # Convert to arrays
    print(f"\nConverting to arrays...")
    save_dict = {}
    for key, value_list in all_representations.items():
        save_dict[key] = np.stack(value_list)  # [vocab_size, hidden_dim]
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
    
    # Save summary
    summary = {
        'model_dir': str(args.model_dir),
        'vocab_file': str(vocab_file),
        'vocab_size': vocab_size,
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'representations_extracted': args.representations,
        'representation_keys': list(save_dict.keys()),
        'representation_shape': f'[vocab_size={vocab_size}, hidden_dim]'
    }
    
    summary_file = output_dir / 'extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"{'='*60}")
    print(f"\nTotal tokens processed: {vocab_size}")
    print(f"Output directory: {output_dir}")
    print(f"\nRepresentation shapes:")
    for key, arr in save_dict.items():
        print(f"  {key}: {arr.shape}")
    
    print(f"\nTo analyze:")
    print(f"  python visualize_representations.py --representation_dir {output_dir}")


if __name__ == '__main__':
    main()
