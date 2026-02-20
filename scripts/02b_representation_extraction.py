"""
Extract and save internal representations for each token
- Hidden states at different points in the architecture
- FFN activations at each layer

When upsampling is enabled: for each token, sample N sequences from the dataset
that contain that token, extract the representation at the token's position in
each sequence; save the average-pooled vector as the main representation (same
naming as non-upsampled) and save the N raw vectors for visualization.

Supported representation types:
- 'residual_before': Residual stream state before each decoder block (input to block)
- 'after_attention': Hidden state after attention, before FFN
- 'after_block': Hidden state after entire decoder block (attention + FFN + residuals) - default
- 'ffn_gate': FFN gate projection activations
- 'ffn_up': FFN up projection activations
"""
import argparse
import json
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM
from datasets import load_from_disk


def _get_architecture(model):
    """Return architecture name from model config, e.g. 'LlamaForCausalLM' or 'MambaForCausalLM'."""
    if getattr(model.config, 'architectures', None) and len(model.config.architectures) > 0:
        return model.config.architectures[0]
    return type(model).__name__


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


class MambaRepresentationExtractor:
    """
    Extract representations from Mamba model (state-space, no attention/FFN).
    Only supports residual_before and after_block (input/output of each backbone layer).
    """

    def __init__(self, model, representations=None):
        self.model = model
        self.representations = {}
        self.hooks = []
        allowed = {'residual_before', 'after_block'}
        requested = set(representations or ['after_block'])
        self.representations_to_extract = [r for r in requested if r in allowed]
        if requested - allowed:
            import warnings
            warnings.warn(
                f"Mamba does not support {requested - allowed}; only {allowed} are available. Using {self.representations_to_extract}."
            )

    def register_hooks(self):
        layers = self.model.backbone.layers
        for layer_idx, layer in enumerate(layers):
            if 'residual_before' in self.representations_to_extract:
                def _residual_before_hook(idx):
                    def hook(module, input, output):
                        inp = input[0] if isinstance(input, tuple) and input else input
                        if hasattr(inp, 'detach'):
                            self.representations[f'layer_{idx}_residual_before'] = inp.detach().cpu()
                    return hook
                self.hooks.append(layer.register_forward_hook(_residual_before_hook(layer_idx)))
            if 'after_block' in self.representations_to_extract:
                def _after_block_hook(idx):
                    def hook(module, input, output):
                        out = output[0] if isinstance(output, tuple) and output else output
                        if hasattr(out, 'detach'):
                            self.representations[f'layer_{idx}_after_block'] = out.detach().cpu()
                    return hook
                self.hooks.append(layer.register_forward_hook(_after_block_hook(layer_idx)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear_representations(self):
        self.representations = {}


def _get_extractor_class(model):
    arch = _get_architecture(model)
    if arch == "MambaForCausalLM":
        return MambaRepresentationExtractor
    return RepresentationExtractor


def extract_token_representations(model, token_id, representations=None):
    """
    Extract representations for a single token.
    Works for Llama (full rep types) and Mamba (residual_before, after_block only).

    Args:
        model: Causal LM (LlamaForCausalLM or MambaForCausalLM)
        token_id: int, the token ID to extract
        representations: list of representation types to extract

    Returns:
        dict with representations for each layer
    """
    ExtractorClass = _get_extractor_class(model)
    extractor = ExtractorClass(model, representations)
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


def extract_token_representations_from_sequence(model, sequence_token_ids, target_position, representations=None):
    """
    Extract representations for a token at a specific position in a sequence.
    Works for Llama and Mamba (same as extract_token_representations).

    Args:
        model: Causal LM (LlamaForCausalLM or MambaForCausalLM)
        sequence_token_ids: list of token IDs
        target_position: int, position in sequence to extract (0-indexed)
        representations: list of representation types to extract

    Returns:
        dict with representations for each layer (each value shape [dim])
    """
    ExtractorClass = _get_extractor_class(model)
    extractor = ExtractorClass(model, representations)
    extractor.register_hooks()

    input_ids = torch.tensor([sequence_token_ids], dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True
        )

    reps = extractor.representations.copy()
    extractor.remove_hooks()

    batch_size, seq_len = input_ids.shape
    token_reps = {}

    for key, value in reps.items():
        if value.dim() == 3:
            token_reps[key] = value[0, target_position, :].detach().cpu()
        elif value.dim() == 2:
            if value.shape[0] == batch_size * seq_len:
                reshaped = value.view(batch_size, seq_len, -1)
                token_reps[key] = reshaped[0, target_position, :].detach().cpu()
            elif value.shape[0] == seq_len:
                token_reps[key] = value[target_position, :].detach().cpu()
            else:
                token_reps[key] = value[0, :].detach().cpu() if value.shape[0] >= 1 else value.squeeze().detach().cpu()
        else:
            token_reps[key] = value.detach().cpu()

    token_reps['input_embeds'] = outputs.hidden_states[0][0, target_position, :].detach().cpu()
    token_reps['final_hidden'] = outputs.hidden_states[-1][0, target_position, :].detach().cpu()

    return token_reps


def build_token_occurrence_index(dataset_split, token_ids_set, seed=42):
    """
    Build index: for each token_id, list of (seq_idx, position) where token appears.

    Returns:
        dict: token_id -> list of (seq_idx, position)
    """
    token_occurrences = defaultdict(list)
    for seq_idx in tqdm(range(len(dataset_split)), desc="Indexing token occurrences"):
        example = dataset_split[seq_idx]
        ids = example.get('input_ids', example.get('input_id', []))
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        for pos, tid in enumerate(ids):
            if tid in token_ids_set:
                token_occurrences[tid].append((seq_idx, pos))
    return dict(token_occurrences)


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
    parser.add_argument('--upsample', action='store_true',
                      help='Upsample: for each token sample N sequences containing it and average (requires --dataset_dir)')
    parser.add_argument('--upsample_n', type=int, default=8,
                      help='Number of context sequences to sample per token when upsampling (default: 8)')
    parser.add_argument('--dataset_dir', type=str, default=None,
                      help='Dataset directory (HuggingFace load_from_disk); required when --upsample')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for sampling upsampling contexts (default: 42)')
    
    args = parser.parse_args()
    
    if args.upsample and not args.dataset_dir:
        parser.error('--dataset_dir is required when --upsample is set')
    
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
    
    # Load model (auto-detects Llama vs Mamba from saved config.json)
    print(f"\nLoading model from {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.eval()
    arch = _get_architecture(model)
    print(f"  Architecture: {arch}")

    if arch == "MambaForCausalLM":
        num_layers = len(model.backbone.layers)
    else:
        num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    intermediate_size = getattr(model.config, 'intermediate_size', None)
    if arch == "MambaForCausalLM":
        args.representations = [r for r in args.representations if r in ('residual_before', 'after_block')]
        if not args.representations:
            args.representations = ['after_block']
        print(f"  Mamba only supports residual_before, after_block; using: {args.representations}")

    print(f"  Num layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    if intermediate_size is not None:
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
    
    token_ids_set = set(token_ids)
    rng = random.Random(args.seed)
    
    if args.upsample:
        # Load dataset and build token -> (seq_idx, position) index
        print(f"\nUpsampling enabled: N={args.upsample_n} sequences per token (seed={args.seed})")
        print(f"Loading dataset from {args.dataset_dir}")
        dataset = load_from_disk(args.dataset_dir)
        split = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
        print(f"  Split size: {len(split)} sequences")
        token_occurrences = build_token_occurrence_index(split, token_ids_set, seed=args.seed)
        missing = [tid for tid in token_ids if not token_occurrences.get(tid)]
        if missing:
            print(f"  Warning: {len(missing)} tokens never appear in dataset; will use single-token extraction for those")
        
        print(f"\n{'='*60}")
        print("Extracting token representations (upsampled)...")
        print(f"{'='*60}\n")
        
        all_representations = defaultdict(list)
        all_raw_representations = defaultdict(list)  # per-token list of [N, dim] -> stack to [vocab_size, N, dim]
        token_metadata = []
        
        for token_id in tqdm(token_ids, desc="Processing tokens"):
            occurrences = token_occurrences.get(token_id, [])
            if len(occurrences) == 0:
                # Fallback: single-token extraction; repeat to [N, dim] for consistent raw shape
                token_reps = extract_token_representations(model, token_id, args.representations)
                for key, value in token_reps.items():
                    v = value.numpy()
                    all_representations[key].append(v)
                    all_raw_representations[key].append(np.broadcast_to(v[np.newaxis, :], (args.upsample_n, v.shape[0])))  # [N, dim]
            else:
                # Sample N (with replacement if fewer than N occurrences)
                if len(occurrences) >= args.upsample_n:
                    chosen = rng.sample(occurrences, args.upsample_n)
                else:
                    chosen = rng.choices(occurrences, k=args.upsample_n)
                raw_list = defaultdict(list)
                for seq_idx, pos in chosen:
                    seq = split[seq_idx]['input_ids']
                    if isinstance(seq, torch.Tensor):
                        seq = seq.tolist()
                    token_reps = extract_token_representations_from_sequence(
                        model, seq, pos, args.representations
                    )
                    for key, value in token_reps.items():
                        raw_list[key].append(value.numpy())
                # Average for main representation
                for key in raw_list:
                    stacked = np.stack(raw_list[key], axis=0)  # [N, dim]
                    avg = np.mean(stacked, axis=0)  # [dim]
                    all_representations[key].append(avg)
                    all_raw_representations[key].append(stacked)  # [N, dim]
            token_metadata.append({'token_id': int(token_id), 'token_str': str(token_id)})
        
        # Convert to arrays: main [vocab_size, dim], raw [vocab_size, N, dim]
        print(f"\nConverting to arrays...")
        save_dict = {}
        for key, value_list in all_representations.items():
            save_dict[key] = np.stack(value_list)
            print(f"  {key} (averaged): {save_dict[key].shape}")
        raw_save_dict = {}
        for key, value_list in all_raw_representations.items():
            raw_save_dict[key] = np.stack(value_list)  # [vocab_size, N, dim]
            print(f"  {key} (raw upsampled): {raw_save_dict[key].shape}")
    else:
        # Direct extraction: one forward per token
        print(f"\n{'='*60}")
        print("Extracting token representations (direct)...")
        print(f"{'='*60}\n")
        
        all_representations = defaultdict(list)
        token_metadata = []
        
        for token_id in tqdm(token_ids, desc="Processing tokens"):
            token_reps = extract_token_representations(model, token_id, args.representations)
            for key, value in token_reps.items():
                all_representations[key].append(value.numpy())
            token_metadata.append({'token_id': int(token_id), 'token_str': str(token_id)})
        
        print(f"\nConverting to arrays...")
        save_dict = {}
        for key, value_list in all_representations.items():
            save_dict[key] = np.stack(value_list)
            print(f"  {key}: {save_dict[key].shape}")
        raw_save_dict = None
    
    # Save main representations (same filename so downstream is unchanged)
    output_file = output_dir / 'token_representations.npz'
    print(f"\nSaving representations to {output_file}")
    np.savez_compressed(output_file, **save_dict)
    
    if raw_save_dict is not None:
        raw_file = output_dir / 'token_representations_upsampled_raw.npz'
        print(f"Saving upsampled raw representations to {raw_file}")
        np.savez_compressed(raw_file, **raw_save_dict)
    
    # Save metadata
    metadata_file = output_dir / 'token_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(token_metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    # Save summary
    summary = {
        'model_dir': str(args.model_dir),
        'architecture': arch,
        'vocab_file': str(vocab_file),
        'vocab_size': vocab_size,
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'representations_extracted': args.representations,
        'representation_keys': list(save_dict.keys()),
        'representation_shape': f'[vocab_size={vocab_size}, hidden_dim]',
        'upsampled': args.upsample,
    }
    if args.upsample:
        summary['dataset_dir'] = str(args.dataset_dir)
        summary['upsample_n'] = args.upsample_n
        summary['seed'] = args.seed
        summary['upsampled_raw_file'] = 'token_representations_upsampled_raw.npz'
    
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
    if raw_save_dict:
        print(f"\nUpsampled raw (for visualization):")
        for key, arr in raw_save_dict.items():
            print(f"  {key}: {arr.shape}")
    
    print(f"\nTo analyze:")
    print(f"  python visualize_representations.py --representation_dir {output_dir}")


if __name__ == '__main__':
    main()
