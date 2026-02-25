# Step 11: Flow Matching Action Head

## Goal

Replace direct action regression with a generative flow matching model. Instead of predicting actions directly, the model learns a velocity field that transforms Gaussian noise into action chunks via ODE integration.

## Concept

**Before (direct regression):**

- Encoder → action queries → cross-attention → Linear → action chunk directly

**After (flow matching):**

- Start from x_0 ~ N(0, 1) (Gaussian noise)
- Integrate learned velocity field from t=0 → t=1
- x_1 (the action chunk) emerges at the end of integration

The model never predicts actions directly — it predicts _how fast to move_ through action space at every point along the trajectory.

## Architecture

```
Image (128x128x3)
    ↓
Patch Embedding (16x16 patches) → 64 tokens of dim 64
    ↓
+ Positional Embeddings
    ↓
Transformer Encoder (2 layers, 4 heads)
    ↓ (image memory)
Transformer Decoder:
  - Queries: action_proj(x_t) + sinusoidal_embed(t)  [chunk_size tokens]
  - Keys/Values: encoded image patches
  - Cross-attention: noisy actions attend to image
    ↓
chunk_size output tokens → Linear(64, 2) → predicted velocity
```

## Training: Flow Matching

Given a clean action chunk x_1 and noise x_0 ~ N(0, 1):

1. Sample t ~ Uniform(0, 1)
2. Interpolate: x*t = (1 - t) * x*0 + t * x_1
3. True velocity: v = x_1 - x_0 (constant along the straight-line path)
4. Train the model to predict v given (obs, t, x_t)
5. Loss: MSE(predicted_v, true_v)

The paths are straight lines by construction (linear interpolation between noise and target).

## Inference: Euler Integration

```python
x_t = sample N(0, 1)   # start from noise
for i in range(N):
    t = i / N
    v = model(obs, t, x_t)
    x_t = x_t + v * (1/N)
# x_t is now the predicted action chunk
```

N=20 steps used. Because the flow paths are straight lines, coarse discretization works well.

## Architecture Details

```
Patch Embedding:
  Conv2d(3, 64, 16x16, stride=16) + GELU
  64 patch tokens (8x8 spatial grid)

Transformer Encoder:
  d_model=64, dim_feedforward=256, nhead=4, num_layers=2

Transformer Decoder:
  d_model=64, dim_feedforward=256, nhead=2, num_layers=2
  Queries: action_proj(x_t) + t_embed (one per action in chunk)

Velocity Head:
  Linear(64, 2) per token → 2D velocity per action step
```

## Training Challenges

**Action normalization is necessary.**
Raw actions have std ≈ 4.37 (clipped to [-5, 5]). Noise is N(0, 1) with std=1. Without normalization, the velocity target x_1 - x_0 has std ≈ 4.5 — the noise and target distributions are mismatched, making the velocity regression much harder. Initial runs without normalization produced near-random performance (completion rate 0.16, matching the random baseline).

Fix: normalize actions to [-1, 1] by dividing by 5.0 before computing the flow. Denormalize (multiply by 5.0) after ODE integration at inference.

**Slow initial convergence.**
Loss was flat for ~60 epochs before dropping sharply, then continued to decrease steadily. Loss was still declining at epoch 99, suggesting more training would improve results.

## Training Setup

- 20k samples (expert + dagger)
- lr=1e-4, batch_size=64
- Actions normalized: divide by 5.0 before flow, multiply by 5.0 after inference

## Training Curve (normalized run)

| Epoch | Train Loss | Val Loss |
| ----- | ---------- | -------- |
| 0     | 1.20       | 0.89     |
| 19    | 0.72       | 0.62     |
| 38    | 0.63       | 0.59     |
| 60    | 0.55       | 0.50     |
| 65    | 0.42       | 0.32     |
| 79    | 0.32       | 0.21     |
| 99    | 0.29       | 0.20     |

## Results

| Variant             | Steps to Capture | Path Inefficiency | Completed Rate |
| ------------------- | ---------------: | ----------------: | -------------: |
| Flow Matching       |    40.36 ± 22.17 |       2.10 ± 2.08 |          0.980 |
| Flow Matching (RH4) |    34.91 ± 21.31 |       1.44 ± 1.02 |          0.990 |

ACT (RH4) and ViT (RH4) still outperform on speed and path efficiency. Expected: this is a unimodal environment where direct regression has an inherent advantage — there is one clearly correct action per observation. Flow matching earns its cost on multimodal distributions where regression would predict blurry averages.

## Inference Speed

Flow matching requires N forward passes per chunk inference (N=20), compared to 1 pass for ACT/ViT. With chunk_size=8 and receding horizon, this is ~20x more compute per environment step.

## Key Takeaways

- Flow matching works end-to-end: Gaussian noise is successfully integrated into coherent action chunks via a learned ODE.
- Action normalization is necessary. Mismatched scales between noise and action distribution (4.4× difference) caused near-random performance before the fix.
- The loss had a long flat phase (~60 epochs) before dropping. More training epochs would likely improve results further.
- Inference cost is the main practical limitation: N forward passes per chunk vs. 1 for direct regression.
- On simple unimodal tasks, direct regression wins. Flow matching becomes more valuable as action distributions become multimodal.

## Status

Complete

---

Location: `policy/flow_matching_policy.py`
