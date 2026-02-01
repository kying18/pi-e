# pi-e

Hypothesis: robotic policies would benefit from a memory bank of sorts. If we can successfully demonstrate this with a toy problem using pi0, this provides evidence that having something that resembles memory may benefit the model. I'd love to call this pi-e (pi with experience)

## Goal

Implement Physical Intelligence's Pi architecture from scratch, building up incrementally:

1. Behavior cloning (supervised learning)
2. Action chunking (predict action sequences)
3. Transformer-based policy
4. Flow matching for action generation
5. Full Pi0-style vision-language-action model

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
├── scripts/              # Training and evaluation
├── data/                 # Collected demonstrations
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
- [ ] Action chunking
- [ ] Transformer policy
- [ ] Flow matching
- [ ] Pi0-style architecture

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
