"""
Train a causal LM (Llama or Mamba) on graph walk sequences.

Model architecture is read from the config JSON key "architectures", e.g.:
  "architectures": ["LlamaForCausalLM"]  or  ["MambaForCausalLM"]
Config file is typically configs/config_${RUN_NAME}.json (e.g. config_12M_llama.json).
"""
import argparse
import json
import os
import torch
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    MambaConfig,
    MambaForCausalLM,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


# Map architecture name from config to (ConfigClass, ModelClass)
ARCH_REGISTRY = {
    "LlamaForCausalLM": (LlamaConfig, LlamaForCausalLM),
    "MambaForCausalLM": (MambaConfig, MambaForCausalLM),
}


# HuggingFace tokenizer for graph walks
class GraphWalkTokenizer(PreTrainedTokenizer):
    """Tokenizer that treats each graph node ID as a token"""
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(self, vocab_size, **kwargs):
        self._vocab_size = vocab_size
        self._vocab = {str(i): i for i in range(vocab_size)}
        self._id_to_token = {i: str(i) for i in range(vocab_size)}
        super().__init__(**kwargs)
    
    @property
    def vocab_size(self):
        return self._vocab_size
    
    def get_vocab(self):
        return self._vocab.copy()
    
    def _tokenize(self, text, **kwargs):
        """Split text into tokens (space-separated numbers)"""
        return text.split()
    
    def _convert_token_to_id(self, token):
        """Convert a token (string) to an id (integer)"""
        return int(token)
    
    def _convert_id_to_token(self, index):
        """Convert an id (integer) to a token (string)"""
        return str(index)
    
    def convert_tokens_to_string(self, tokens):
        """Convert a sequence of tokens to a single string"""
        return " ".join(tokens)
    
    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save the vocabulary to a directory"""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        with open(vocab_file, 'w') as f:
            json.dump(self._vocab, f, indent=2)
        
        return (vocab_file,)
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """No special tokens, just return the input as-is"""
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1


# Custom data collator for graph walks
class GraphWalkDataCollator:
    """Collator that pads sequences and creates attention masks"""
    def __init__(self, pad_token_id=0, max_seq_length=None):
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
    
    def __call__(self, features):
        # Get max length in batch
        batch_max_length = max(len(f['input_ids']) for f in features)
        
        # If max_seq_length is set, use it; otherwise use batch max
        max_length = self.max_seq_length if self.max_seq_length else batch_max_length
        
        input_ids = []
        labels = []
        attention_mask = []
        
        for f in features:
            seq_len = len(f['input_ids'])
            
            # Truncate if needed
            if seq_len > max_length:
                seq = f['input_ids'][:max_length]
            else:
                seq = f['input_ids']
            
            padding_length = max_length - len(seq)
            
            # Pad input_ids
            padded_input = seq + [self.pad_token_id] * padding_length
            input_ids.append(padded_input)
            
            # Pad labels (use -100 for padding tokens so they're ignored in loss)
            padded_labels = seq + [-100] * padding_length
            labels.append(padded_labels)
            
            # Create attention mask
            mask = [1] * len(seq) + [0] * padding_length
            attention_mask.append(mask)
        
        # Create tensors - Trainer will automatically move them to the correct device
        # Using stack instead of tensor for better performance
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


def count_parameters(model):
    """Count model parameters with embedding breakdown"""
    total_params = 0
    trainable_params = 0
    embedding_params = 0
    non_embedding_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        
        if 'embed_tokens' in name or 'embeddings' in name or 'lm_head' in name:
            embedding_params += num_params
        else:
            non_embedding_params += num_params
    
    return total_params, trainable_params, embedding_params, non_embedding_params


def main():
    parser = argparse.ArgumentParser(description='Train Llama on graph walks')
    parser.add_argument('--dataset_dir', type=str, default='./prepared_dataset',
                      help='Directory with prepared dataset')
    parser.add_argument('--config', type=str, default='config_400K_llama.json',
                      help='Model config JSON file')
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='Output directory for model and checkpoints')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Per-device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                      help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                      help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')
    parser.add_argument('--save_steps', type=int, default=50,
                      help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=50,
                      help='Evaluate every N steps')
    parser.add_argument('--logging_steps', type=int, default=10,
                      help='Log every N steps')
    parser.add_argument('--logging_dir', type=str, default=None,
                      help='Directory for logs (default: {output_dir}/logs)')
    parser.add_argument('--save_total_limit', type=str, default='2',
                      help='Maximum number of checkpoints to keep (integer) or "all" to save all checkpoints')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                      help='Path to checkpoint directory to resume training from')
    parser.add_argument('--no_cuda', action='store_true',
                      help='Force CPU training')
    
    args = parser.parse_args()
    
    # Parse save_total_limit - convert "all" to None, otherwise parse as int
    if args.save_total_limit.lower() == 'all':
        save_total_limit = None
    else:
        try:
            save_total_limit = int(args.save_total_limit)
            if save_total_limit < 1:
                raise ValueError("save_total_limit must be >= 1 or 'all'")
        except ValueError as e:
            raise ValueError(f"Invalid save_total_limit: {args.save_total_limit}. Must be a positive integer or 'all'") from e
    
    # Check CUDA availability and initialize if needed
    cuda_available = torch.cuda.is_available()
    
    print("="*60)
    print("Graph Walk Model Training")
    print("="*60)
    
    # Device information
    print(f"\nDevice availability:")
    print(f"  CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA current device: {torch.cuda.current_device()}")
        
        # Check GPU compute capability
        try:
            props = torch.cuda.get_device_properties(0)
            compute_capability = f"{props.major}.{props.minor}"
            print(f"  Compute capability: sm_{props.major}{props.minor}")
            
            # Check if compute capability is supported
            # PyTorch stable builds typically support up to sm_90
            # sm_100, sm_101, sm_102, sm_103, sm_110, sm_120+ require newer builds
            if props.major >= 10:
                print(f"\n  ⚠ WARNING: GPU has compute capability sm_{props.major}{props.minor}")
                print(f"  This GPU architecture may not be fully optimized by your PyTorch build.")
                print(f"  Current PyTorch version: {torch.__version__}")
                
                if props.major >= 12:
                    print(f"\n  ⚠ WARNING: Blackwell architecture (sm_{props.major}{props.minor}) may not be fully supported.")
                    print(f"  PyTorch {torch.__version__} may not have optimized CUDA kernels for sm_120.")
                    print(f"  Training will proceed with GPU, but you may encounter issues.")
                    print(f"  If you encounter 'no kernel image available' errors, try:")
                    print(f"    1. Use CPU training: USE_CPU=true ./12a_combined_model_training.sh")
                    print(f"    2. Try PyTorch nightly (may have experimental support):")
                    print(f"       pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
                    print(f"  Continuing with GPU training...")
                elif props.major == 10:
                    print(f"  ⚠ Hopper architecture (sm_{props.major}{props.minor}) may require PyTorch 2.1+")
                    print(f"  If you encounter errors, try: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124")
        except Exception as e:
            print(f"  Could not check compute capability: {e}")
        
        # Set default device to GPU 0 if using GPU and not falling back to CPU
        if not args.no_cuda:
            torch.cuda.set_device(0)
            print(f"  ✓ CUDA device set to GPU 0")
    
    # Override no_cuda if CUDA is not available and user didn't explicitly request CPU
    if args.no_cuda:
        actual_device = "CPU"
    elif not cuda_available:
        print(f"\nWarning: CUDA not available, falling back to CPU")
        args.no_cuda = True
        actual_device = "CPU"
    else:
        actual_device = "GPU"
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)
    print(f"  Train samples: {len(dataset['train'])}")
    if 'validation' in dataset:
        print(f"  Validation samples: {len(dataset['validation'])}")
    
    # Load vocabulary info
    vocab_info_path = Path(args.dataset_dir) / 'vocab_info.json'
    with open(vocab_info_path, 'r') as f:
        vocab_info = json.load(f)
    vocab_size = vocab_info['vocab_size']
    print(f"\nVocabulary size: {vocab_size}")
    
    # Load or create tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer_path = Path(args.dataset_dir)
    
    # Check if tokenizer was saved (from combined dataset preparation)
    if (tokenizer_path / 'tokenizer_config.json').exists() or (tokenizer_path / 'vocab.json').exists():
        print(f"  Found saved tokenizer in dataset directory, loading...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            print(f"  ✓ Loaded tokenizer from dataset directory")
            print(f"  Tokenizer vocab size: {len(tokenizer)}")
            # Verify vocab size matches
            if len(tokenizer) != vocab_size:
                print(f"  ⚠ WARNING: Tokenizer vocab size ({len(tokenizer)}) != vocab_info vocab_size ({vocab_size})")
                print(f"  Using tokenizer vocab size: {len(tokenizer)}")
                vocab_size = len(tokenizer)
        except Exception as e:
            print(f"  ✗ Failed to load tokenizer: {e}")
            print(f"  Falling back to GraphWalkTokenizer")
            tokenizer = GraphWalkTokenizer(vocab_size)
    else:
        # Use custom GraphWalkTokenizer for pure synthetic datasets
        print(f"  No saved tokenizer found, using GraphWalkTokenizer")
        tokenizer = GraphWalkTokenizer(vocab_size)
    
    print(f"  Final tokenizer vocab size: {len(tokenizer)}")
    
    # Load model config and resolve architecture from "architectures" in JSON
    print(f"\nLoading config from {args.config}")
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    arch_list = config_dict.get("architectures", ["LlamaForCausalLM"])
    arch = arch_list[0] if arch_list else "LlamaForCausalLM"
    if arch not in ARCH_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}' in config. Supported: {list(ARCH_REGISTRY.keys())}. "
            "Add \"architectures\": [\"LlamaForCausalLM\"] or [\"MambaForCausalLM\"] to your config JSON."
        )
    ConfigClass, ModelClass = ARCH_REGISTRY[arch]
    print(f"  Architecture: {arch}")
    # Build config from dict (omit 'architectures' for Config constructor)
    config_dict = {k: v for k, v in config_dict.items() if k != "architectures"}
    config = ConfigClass(**config_dict)
    config.vocab_size = vocab_size

    if arch == "LlamaForCausalLM":
        if not getattr(config, 'max_position_embeddings', None):
            config.max_position_embeddings = 2048
        if not getattr(config, 'rope_theta', None):
            config.rope_theta = 10000.0

    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    if hasattr(config, 'num_attention_heads'):
        print(f"  Num heads: {config.num_attention_heads}")
    if hasattr(config, 'intermediate_size'):
        print(f"  Intermediate size: {config.intermediate_size}")
    if hasattr(config, 'max_position_embeddings'):
        print(f"  Max position embeddings: {config.max_position_embeddings}")

    # Initialize or load model
    print(f"\nInitializing model...")
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"  Loading model from checkpoint: {args.resume_from_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(args.resume_from_checkpoint)
        print(f"  ✓ Model loaded from checkpoint")
    else:
        model = ModelClass(config)
        print(f"  ✓ Model initialized from config ({arch})")
    print(f"Model: {model}")
    
    # Count parameters
    total, trainable, emb, non_emb = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Total: {total:,} ({total/1e6:.2f}M)")
    print(f"  Trainable: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"  Embedding: {emb:,} ({emb/1e6:.2f}M)")
    print(f"  Non-embedding: {non_emb:,} ({non_emb/1e6:.2f}M)")
    print(f"  Non-emb ratio: {non_emb/total*100:.1f}%")
    
    # Report GPU memory before moving model
    if cuda_available and not args.no_cuda:
        print(f"\nGPU memory before model placement:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Truncate sequences if needed
    def truncate_sequences(examples):
        examples['input_ids'] = [
            seq[:args.max_length] for seq in examples['input_ids']
        ]
        return examples
    
    dataset = dataset.map(truncate_sequences, batched=True)
    
    # Setup training arguments
    print(f"\n{'='*60}")
    print("Training configuration:")
    print(f"{'='*60}")
    
    # Determine device for model placement
    device = torch.device("cuda" if cuda_available and not args.no_cuda else "cpu")
    
    # Check if resuming from checkpoint
    is_resuming = args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint)
    if is_resuming:
        print(f"\n  Resuming training from checkpoint: {args.resume_from_checkpoint}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=not is_resuming,  # Don't overwrite when resuming
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps" if 'validation' in dataset else "no",
        eval_steps=args.eval_steps if 'validation' in dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=save_total_limit,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=args.logging_dir if args.logging_dir else f"{args.output_dir}/logs",
        no_cuda=args.no_cuda,
        report_to="none",
        fp16=False,
        load_best_model_at_end='validation' in dataset,
        metric_for_best_model="loss" if 'validation' in dataset else None,
        ddp_find_unused_parameters=False,  # Better performance for single GPU
        dataloader_pin_memory=True if cuda_available and not args.no_cuda else False,  # Faster data loading on GPU
        dataloader_num_workers=0,  # Avoid multiprocessing issues, let Trainer handle it
    )
    
    # Explicitly move model to device before creating trainer
    print(f"\nMoving model to device: {device}")
    model = model.to(device)
    
    print(f"  Device: {actual_device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Save steps: {args.save_steps}")
    print(f"  Save total limit: {'all' if save_total_limit is None else save_total_limit}")
    print(f"  Eval steps: {args.eval_steps if 'validation' in dataset else 'N/A'}")
    print(f"  Logging steps: {args.logging_steps}")
    print(f"  Logging dir: {args.logging_dir if args.logging_dir else f'{args.output_dir}/logs'}")
    
    # Create data collator
    data_collator = GraphWalkDataCollator(
        pad_token_id=0,
        max_seq_length=args.max_length
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation'),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Verify device placement
    model_device = next(model.parameters()).device
    print(f"\nDevice verification:")
    print(f"  Model device: {model_device}")
    if not args.no_cuda and cuda_available:
        if model_device.type != 'cuda':
            print(f"  ✗ WARNING: Model is not on CUDA device despite CUDA being available!")
        else:
            print(f"  ✓ Model successfully placed on GPU")
            # Report GPU memory after model placement
            print(f"\nGPU memory after model placement:")
            print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3:.2f} GB")
            
            # Verify CUDA is working
            try:
                test_tensor = torch.randn(10, 10).to(device)
                _ = test_tensor @ test_tensor
                print(f"  ✓ CUDA operations verified working")
            except Exception as e:
                print(f"  ✗ ERROR: CUDA operations failed: {e}")
    elif args.no_cuda or not cuda_available:
        if model_device.type != 'cpu':
            print(f"  ✗ WARNING: Model device mismatch!")
        else:
            print(f"  ✓ Model correctly using CPU")
    
    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    # Additional GPU memory check right before training
    if cuda_available and not args.no_cuda:
        print(f"GPU memory before training start:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print("")
    
    # Resume from checkpoint if provided
    resume_from_checkpoint = args.resume_from_checkpoint if is_resuming else None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Report final GPU memory usage
    if cuda_available and not args.no_cuda:
        print(f"\nGPU memory after training:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print("")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    
    # Save model and tokenizer
    final_output_dir = Path(args.output_dir) / "final_model"
    print(f"\nSaving model to {final_output_dir}")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    config.save_pretrained(final_output_dir)
    print(f"Model, tokenizer, and config saved!")


if __name__ == '__main__':
    main()

