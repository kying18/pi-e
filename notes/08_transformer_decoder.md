# Step 8: Transformer Decoder with Action Queries (ACT-style)

## Goal

Replace the MLP action head with a transformer decoder that uses learned "action queries" to generate the action chunk. This is the key architectural innovation from ACT that carries into Pi0.

## Concept

**Before (action chunking with MLP):**
- CNN → flatten → MLP → action chunk (all at once)

**After (transformer decoder):**
- CNN → spatial features (tokens)
- Learned action queries (one per action in chunk)
- Transformer decoder: queries attend to image tokens
- Each query outputs one action

## Why Action Queries?

Instead of predicting all actions from a single pooled feature, each action in the chunk gets its own "query" that can attend to different parts of the image. This allows:
- Different actions to focus on different spatial regions
- More expressive action generation
- Natural handling of variable-length sequences

## Architecture

```
Image (128x128x3)
    ↓
CNN Encoder → (16x16, 64) features
    ↓
Flatten → 256 tokens of dim 64
    ↓
Transformer Decoder:
  - Queries: 8 learned embeddings (one per action)
  - Keys/Values: image tokens
  - Cross-attention: queries attend to image
    ↓
8 output tokens → Linear → 8 actions (each 2D)
```

## Key Components

1. **CNN Encoder**: Same as before, extracts spatial features
2. **Action Queries**: `nn.Parameter(torch.zeros(1, chunk_size, embed_dim))`
3. **Transformer Decoder**: `nn.TransformerDecoder` with cross-attention
4. **Action Head**: Linear layer per query → 2D action

## Implementation Notes

- PyTorch's `nn.TransformerDecoder` expects:
  - `tgt`: action queries (B, chunk_size, embed_dim)
  - `memory`: image tokens (B, num_tokens, embed_dim)
- Positional embeddings for image tokens (spatial positions)
- No causal masking needed (all actions predicted in parallel)

## Training Setup

- 10k samples (expert + dagger)
- lr=1e-3, batch_size=64
- ~20 epochs (converged early with good hyperparams)
- MSE loss on action chunks

## Architecture Details

**Final configuration:**
```
CNN Encoder:
  Conv2d(3, 32, 3x3) + ReLU + MaxPool(4)   → 32x32x32
  Conv2d(32, 32, 3x3) + ReLU + MaxPool(2)  → 16x16x32

Transformer Decoder:
  d_model=32, dim_feedforward=256, nhead=2, num_layers=2
  256 image tokens (16x16 spatial)
  8 action queries (one per action in chunk)

Action Head:
  Linear(32, 2) per query → 8 actions
```

Small model works great for this simple task - proves the architecture works without needing massive params.

## Hyperparameter Experiments

**dim_feedforward matters:**
- Default dim_feedforward=2048 with d_model=32 is a 64x ratio (way too large)
- Using dim_feedforward=512 with d_model=128 (4x ratio) converges much faster
- Standard ratio is 4x (e.g., d_model=512 → dim_feedforward=2048 in original transformer)

**Larger CNN didn't help:**
- Tried matching BC's CNN (3→16→32, output 32x32)
- Val loss was higher - more capacity just overfits on this simple task
- Smaller CNN (256 tokens) works better than larger (1024 tokens)

## Parameter Count

| Model | Parameters |
|-------|-----------|
| BC Policy | 4,207,906 |
| Action Chunking Policy | 4,208,816 |
| ACT Policy | 69,282 |

ACT is ~60x smaller and performs well on this task.

**Key insight:** The flatten→linear approach doesn't scale.
- Flatten+Linear: params ∝ (H × W × C) × hidden_dim → explodes with image size
- Transformer: params ∝ d_model² × num_layers → independent of num_tokens

BC's `Linear(32768, 128)` alone is 4.2M params. Transformer decoder avoids this by keeping spatial tokens and combining them dynamically via attention.

## Results

**Qualitative observations:**
- Open-loop (actions_per_inference=8): When EE gets close to the ball, it pauses and takes several more inferences to reach it
- Receding horizon (actions_per_inference=4): Mitigates the pausing issue
- Receding horizon (actions_per_inference=2): Much smoother execution

The shorter horizon allows more frequent corrections, which helps when approaching the target.

## Status

Complete

---

Location: `policy/act_policy.py`
