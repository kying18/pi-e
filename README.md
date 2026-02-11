# pi-e

Hypothesis: robotic policies would benefit from a memory bank of sorts. If we can successfully demonstrate this with a toy problem using pi0, this provides evidence that having something that resembles memory may benefit the model. I'd love to call this pi-e (pi with experience)

## Goal

Implement Physical Intelligence's Pi architecture from scratch, building up incrementally through the key papers/techniques that led to it:

1. **Behavior cloning** - supervised learning baseline
2. **Action chunking** - predict action sequences (key idea from ACT)
3. **Transformer decoder with action queries** - ACT-style architecture
4. **ViT encoder** - replace CNN with Vision Transformer
5. **Flow matching** - generative action modeling (replaces direct regression)
6. **VLA** - add language conditioning for full Pi0-style model

### Why this progression?

- **ACT** (2023, from Pi team) introduced action chunking + transformer decoder with learned action queries
- **Pi0** builds on ACT, adding flow matching and language conditioning
- Understanding ACT's transformer decoder is key to understanding Pi0

## Environment

Simple 2D ball interception task:
- Red ball bounces around the screen
- Blue end-effector (robot) must intercept it
- Observation: 256x256 RGB image
- Action: (dx, dy) velocity command

## Structure

```
pi/
├── env/                  # Environment
│   └── moving_object.py
├── expert/               # Expert policy for demonstrations
│   └── expert_policy.py
├── policy/               # Learned policies
├── eval/                 # Evaluation framework
├── scripts/              # Training and evaluation
├── data/                 # Collected demonstrations
├── notes/                # Design notes and videos
└── visualize.py          # Visualization
```

## Usage

Visualize with expert policy:
```bash
python visualize.py
```

## Progress

- [x] Environment
- [x] Expert policy
- [x] Data collection
- [x] Behavior cloning policy (single-frame)
- [x] DAgger for single-frame BC
- [x] Action chunking
- [x] Transformer decoder with action queries (ACT-style)
- [x] ViT encoder
- [x] Evaluation framework + baseline metrics
- [ ] Flow matching
- [ ] Language conditioning (VLA)

## Lessons Learned

### Multi-frame BC (frame stacking) - Abandoned

Attempted stacking 3 frames along channel dimension (H, W, 9) to provide temporal information. This approach had several problems:

1. **Zero-frame contamination**: First observations in each episode have zero-padding for older frames, teaching the model to ignore temporal channels.

2. **No temporal inductive bias**: Conv2d treats all 9 channels equally - it doesn't know channels 0-2, 3-5, 6-8 represent different time steps.

3. **DAgger bootstrap failure**: Collecting DAgger data with a poorly-trained policy produces low-quality data that doesn't help.

4. **It's a 2015-era technique**: Frame stacking was popularized by DQN for Atari. Modern approaches (ACT, Pi0) handle temporal information differently:
   - Encode each frame separately with a vision encoder
   - Use transformer attention over frame embeddings
   - Or use action chunking, which implicitly captures dynamics

**Takeaway**: Skip frame stacking. For temporal reasoning, use action chunking or attention over frame sequences.

### Single-frame BC + DAgger

After abandoning multi-frame BC, we returned to single-frame BC and implemented DAgger properly.

**Training setup (same for BC and BC+DAgger):**
- 10k datapoints
- Learning rate: 1e-3
- Train/val split: 70/30
- Batch size: 64
- Epochs: 45
- Simple CNN encoder (2 conv layers → MLP)

**DAgger collection:**
- Run learned policy in environment (which drifts to edges/corners)
- Label those edge states with expert recovery actions
- Aggregate with original expert data and retrain from scratch

**Why it works:**
- Expert is good → stays centered → expert data lacks edge samples
- Learned policy is imperfect → visits edges → DAgger captures edge recovery data
- Result: policy learns to avoid/recover from edges

**Results:**

| Policy | Video |
|--------|-------|
| Random | [00_random_policy.mp4](notes/videos/00_random_policy.mp4) |
| Expert | [01_expert_policy.mp4](notes/videos/01_expert_policy.mp4) |
| BC (single-frame) | [03_bc_policy.mp4](notes/videos/03_bc_policy.mp4) |
| BC + DAgger | [06_bc_policy_dagger.mp4](notes/videos/06_bc_policy_dagger.mp4) |
| Multi-frame BC (abandoned) | [04_multi_img_bc_policy.mp4](notes/videos/04_multi_img_bc_policy.mp4) |
| Action chunking (open-loop) | [07_action_chunking_policy.mp4](notes/videos/07_action_chunking_policy.mp4) |
| Action chunking (RH4 + episode ends + padding) | [07_action_chunking_policy_rh4_episode_ends_padded.mp4](notes/videos/07_action_chunking_policy_rh4_episode_ends_padded.mp4) |

### Transformer Decoder (ACT-style)

Replaced the MLP action head with a transformer decoder using learned action queries. Each action in the chunk gets its own query that attends to image tokens via cross-attention.

**Key result:** 60x fewer parameters than BC/action chunking (69K vs 4.2M) with comparable or better performance. The flatten+linear bottleneck in BC/action chunking explodes with image size, while transformer params scale with d_model, independent of token count.

**Receding horizon helps:** Open-loop (execute all 8) causes pausing near the target. RH4 (predict 8, execute 4) and RH2 are progressively smoother.

### ViT Encoder

Replaced the CNN encoder with a Vision Transformer (patch embeddings + self-attention encoder, then cross-attention decoder).

**Training required more data:** 10k samples stayed flat for 30+ epochs. 20k samples + lower lr (1e-4) converged by epoch 40. This matches the original ViT paper — without convolutional inductive biases, the model needs more data to learn spatial relationships.

### Baseline Metrics

| Policy | Steps to Capture | Path Inefficiency | Dir. Consistency | Mag. Continuity | Completed | Params |
|---|---:|---:|---:|---:|---:|---:|
| Expert | 16.83 ± 10.25 | 1.06 ± 0.07 | 0.982 ± 0.029 | 0.031 ± 0.045 | 1.000 | N/A |
| BC + DAgger | 31.55 ± 26.58 | 1.38 ± 0.95 | 0.929 ± 0.097 | 0.135 ± 0.084 | 0.910 | 4.2M |
| BC | 32.62 ± 31.41 | 2.00 ± 12.79 | 0.969 ± 0.039 | 0.094 ± 0.054 | 0.840 | 4.2M |
| ACT (RH4) | 23.98 ± 15.22 | 1.20 ± 0.26 | 0.911 ± 0.111 | 0.140 ± 0.097 | 1.000 | 69K |
| ViT (RH4) | 25.39 ± 15.90 | 1.24 ± 0.52 | 0.915 ± 0.087 | 0.125 ± 0.076 | 1.000 | 287K |
| Random | 89.99 ± 26.47 | 14.54 ± 13.45 | 0.009 ± 0.159 | 0.681 ± 0.096 | 0.157 | N/A |

ACT (RH4) and ViT (RH4) are the best learned policies — near-expert completion rate with 15-60x fewer parameters than BC.

### Action Chunking

Predict 8 future actions at once instead of 1. Key idea from ACT that carries into Pi0.

**Execution modes:**
- Open-loop: execute all 8 actions, then re-predict
- Receding horizon: predict 8, execute fewer (e.g., 4), then re-predict

**Data improvements:**
- Added `episode_ends` tracking to avoid cross-episode contamination
- Zero-pad action chunks at episode boundaries

**Results:** Receding horizon (4) + clean episode data produces smoother execution. However, for this simple task, not dramatically better than BC + DAgger. Action chunking likely shines more on complex tasks with temporal structure.

### Why Keep the Simple Task?

We're already performing this task quite well with BC + DAgger. But we'll continue with this toy example because it's easier to build and compare architectures when keeping the task the same. Once we understand the full Pi0 architecture (transformers, flow matching, VLA), we can expand to harder scenarios:

- Multi-step tasks (catch ball → carry to goal)
- Partial observability (ball goes behind occluder, needs memory)
- Variable dynamics (ball behavior changes mid-episode)
- Multi-object reasoning (multiple balls, specific order)
- Longer horizon planning
