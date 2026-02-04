# Step 9: ViT Encoder

## Goal

Replace the CNN encoder with a Vision Transformer (ViT). This replaces fixed convolutional features with learned patch embeddings and self-attention, allowing the encoder to learn spatial relationships.

## Concept

**Before (CNN + Transformer Decoder):**
- CNN → spatial features (tokens)
- Transformer decoder: queries attend to CNN tokens

**After (ViT + Transformer Decoder):**
- Patch embedding → patch tokens
- Transformer encoder: patches attend to each other
- Transformer decoder: queries attend to encoded patches

## Architecture

```
Image (128x128x3)
    ↓
Patch Embedding (16x16 patches) → 64 tokens of dim 64
    ↓
+ Positional Embeddings
    ↓
Transformer Encoder (2 layers, 4 heads)
    ↓
Transformer Decoder:
  - Queries: 8 learned embeddings (one per action)
  - Keys/Values: encoded patch tokens
  - Cross-attention: queries attend to patches
    ↓
8 output tokens → Linear → 8 actions (each 2D)
```

## Architecture Details

**Final configuration:**
```
Patch Embedding:
  Conv2d(3, 64, 16x16, stride=16) + ReLU
  64 patch tokens (8x8 spatial grid)

Transformer Encoder:
  d_model=64, dim_feedforward=256, nhead=4, num_layers=2

Transformer Decoder:
  d_model=64, dim_feedforward=256, nhead=2, num_layers=2
  8 action queries (one per action in chunk)

Action Head:
  Linear(64, 2) per query → 8 actions
```

## Training Challenges

**ViT requires more data than CNN:**
- With 10k samples: loss stayed flat for 30+ epochs, then decreased slowly
- With 20k samples: loss started decreasing after epoch 12, converged well

This matches findings from the original ViT paper - transformers need more data to outperform CNNs. Without inductive biases from convolutions, ViT must learn spatial relationships from scratch.

**Hyperparameter sensitivity:**
- lr=1e-3: stayed flat through 50 epochs
- lr=1e-4: worked well with sufficient data
- Smaller model (embed 32, 2 heads): stayed flat, insufficient capacity

## Training Setup

- 20k samples (expert + dagger)
- lr=1e-4, batch_size=64
- Train/val split: 14k/6k
- Loss started decreasing at epoch 12
- Converged by epoch 40

## Results

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 12    | 17.94      | 23.48    |
| 20    | 5.43       | 4.84     |
| 30    | 2.99       | 2.62     |
| 40    | 2.57       | 2.40     |

## Parameter Count

| Model | Parameters |
|-------|-----------|
| BC Policy | 4,207,906 |
| Action Chunking Policy | 4,208,816 |
| ACT Policy (CNN) | 69,282 |
| ViT Policy | 287,426 |

ViT is ~4x larger than CNN-based ACT, but still ~15x smaller than BC. The additional parameters are in the transformer encoder layers.

## Status

Complete

---

Location: `policy/vit_policy.py`
