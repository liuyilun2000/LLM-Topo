# Stage 02: Model Training and Representation Extraction

This stage trains language models on graph walk sequences and extracts internal representations.

## Overview

Stage 02 consists of two steps:
1. **Model Training** (`02a`) - Train Llama models on walk sequences
2. **Representation Extraction** (`02b`) - Extract internal representations from trained models

## Step 1: Model Training

Train a Llama language model on the prepared walk sequences.

### Script: `02a_model_training.sh`

```bash
./02a_model_training.sh
```

### Configuration

```bash
# Custom training parameters
CONFIG=configs/config_2M_llama.json \
EPOCHS=10 \
BATCH_SIZE=32 \
LEARNING_RATE=5e-4 \
./02a_model_training.sh
```

**Parameters:**
- `CONFIG` - Model configuration file (default: `configs/config_{RUN_NAME}.json`)
- `DATASET_DIR` - Dataset directory (default: `results/{DATASET_NAME}/dataset`)
- `OUTPUT_DIR` - Output directory (default: `results/{DATASET_NAME}/{RUN_NAME}`)
- `EPOCHS` - Number of training epochs (default: 1)
- `BATCH_SIZE` - Batch size (default: 50)
- `GRAD_ACCUM` - Gradient accumulation steps (default: 1)
- `LEARNING_RATE` - Learning rate (default: 5e-4)
- `MAX_LENGTH` - Maximum sequence length (default: 128)
- `SAVE_STEPS` - Save checkpoint every N steps (default: 200)
- `EVAL_STEPS` - Evaluate every N steps (default: 200)
- `LOGGING_STEPS` - Log every N steps (default: 10)
- `USE_CPU` - Force CPU training (default: false)

### Model Configurations

Available presets in `configs/`:

| Config | Parameters | Hidden | Layers | Heads | FFN |
|--------|-----------|--------|--------|-------|-----|
| config_400K_llama.json | ~400K | 64 | 6 | 4 | 256 |
| config_2M_llama.json | ~2M | 128 | 8 | 8 | 512 |
| config_6M_llama.json | ~6M | 256 | 6 | 8 | 1024 |
| config_12M_llama.json | ~12M | 384 | 8 | 8 | 1536 |

### Output Files

Generated in `results/{DATASET_NAME}/{RUN_NAME}/`:

- `checkpoint-{step}/` - Training checkpoints
- `final_model/` - Final trained model
  - `config.json` - Model configuration
  - `model.safetensors` - Model weights
  - `tokenizer_config.json` - Tokenizer configuration
  - `vocab.json` - Vocabulary file

### Training Loss

Expected loss values:
- **Random guessing**: log(vocab_size) ≈ 4.6 (for 100 nodes)
- **Ideal model**: log(neighbors) ≈ 1.95 (for 8 neighbors)
- **Good model**: 1.6-1.8

### Example

```bash
# CPU training (small model)
USE_CPU=true CONFIG=configs/config_400K_llama.json ./02a_model_training.sh

# GPU training (larger model)
CONFIG=configs/config_6M_llama.json \
EPOCHS=10 \
BATCH_SIZE=32 \
./02a_model_training.sh
```

## Step 2: Representation Extraction

Extract internal representations from trained models for each token.

### Script: `02b_representation_extraction.sh`

```bash
./02b_representation_extraction.sh
```

### Configuration

```bash
# Extract multiple representation types
REPRESENTATIONS="after_block ffn_gate ffn_up" ./02b_representation_extraction.sh
```

**Parameters:**
- `MODEL_DIR` - Trained model directory (default: `results/{DATASET_NAME}/{RUN_NAME}/final_model`)
- `REPRESENTATION_DIR` - Output directory (default: `results/{DATASET_NAME}/{RUN_NAME}/token_representations`)
- `REPRESENTATIONS` - Representation types to extract (default: "after_block")

### Representation Types

Available representation types:

- `residual_before` - Residual stream state **before** each decoder block (input to block)
- `after_attention` - Hidden state **after** attention, **before** FFN
- `after_block` - Hidden state **after** entire decoder block (attention + FFN + residuals) **[default]**
- `ffn_gate` - FFN gate projection activations
- `ffn_up` - FFN up projection activations

### Output Files

Generated in `results/{DATASET_NAME}/{RUN_NAME}/token_representations/`:

- `token_representations.npz` - All representations
  - Keys: `layer_{i}_after_block`, `layer_{i}_ffn_gate`, etc.
  - Shape: `[vocab_size, hidden_dim]` for each representation
- `representation_info.json` - Extraction metadata
  - Model information
  - Representation types extracted
  - Vocabulary size

### Data Structure

For a model with 8 layers and vocab size 1200:

```python
import numpy as np

data = np.load('token_representations.npz')

# Access layer 0 hidden states
layer0 = data['layer_0_after_block']  # Shape: [1200, 128]

# Access FFN activations
ffn_gate = data['layer_0_ffn_gate']   # Shape: [1200, 512]
ffn_up = data['layer_0_ffn_up']       # Shape: [1200, 512]
```

### Example

```bash
# Extract default (after_block only)
./02b_representation_extraction.sh

# Extract multiple types
REPRESENTATIONS="after_block ffn_gate ffn_up" ./02b_representation_extraction.sh

# Extract all types
REPRESENTATIONS="residual_before after_attention after_block ffn_gate ffn_up" \
./02b_representation_extraction.sh
```

## Complete Stage 02 Workflow

```bash
# 1. Train model
CONFIG=configs/config_2M_llama.json \
EPOCHS=10 \
BATCH_SIZE=32 \
./02a_model_training.sh

# 2. Extract representations
REPRESENTATIONS="after_block ffn_gate ffn_up" \
./02b_representation_extraction.sh
```

## Output Structure

```
results/{DATASET_NAME}/{RUN_NAME}/
├── checkpoint-{step}/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
│
├── final_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── vocab.json
│
└── token_representations/
    ├── token_representations.npz
    └── representation_info.json
```

## Troubleshooting

### Model Training Issues

**Out of Memory:**
```bash
# Use smaller model
CONFIG=configs/config_400K_llama.json ./02a_model_training.sh

# Reduce batch size
BATCH_SIZE=4 GRAD_ACCUM=8 ./02a_model_training.sh
```

**Slow Training:**
```bash
# Use GPU (if available)
USE_CPU=false ./02a_model_training.sh

# Increase batch size
BATCH_SIZE=64 ./02a_model_training.sh
```

### Representation Extraction Issues

**Model Not Found:**
```bash
# Check model exists
ls results/{DATASET_NAME}/{RUN_NAME}/final_model/

# Re-train if needed
./02a_model_training.sh
```

**Memory Issues:**
```bash
# Extract fewer representations at once
REPRESENTATIONS="after_block" ./02b_representation_extraction.sh
```

## Next Steps

After Stage 02, proceed to:
- [README_03.md](README_03.md) - Dimensionality reduction (PCA, fuzzy neighborhood, UMAP)

