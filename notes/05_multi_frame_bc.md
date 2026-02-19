# Step 5: Behavior Cloning (Multi-Frame)

## Goal

Train a policy on stacked frames to capture temporal dynamics.

## Approach

Modified CNN architecture:
- Input: 64x64x9 (3 RGB frames, resized)
- First conv layer: `nn.Conv2d(9, 16, ...)` to handle 9 channels
- Rest of architecture similar to single-frame

## Training Observations

### Gradient Explosion

Around epoch 28, loss suddenly jumped from ~2 to ~21. This is gradient explosion - a large gradient caused a catastrophic weight update.

**Fix**: Add gradient clipping:
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

### Covariate Shift (Getting Stuck in Corners)

**Problem**: The policy gets stuck in corners/edges of the environment.

**Why**: The expert never visits corners - it catches the ball quickly from sensible positions. So the training data has no examples of "how to escape a corner." When the learned policy drifts there by mistake, it's in unfamiliar territory.

```
Expert data: only "good" states
     ↓
Learned policy makes small error
     ↓
Enters unseen state
     ↓
Bad prediction → worse state → stuck
```

This is a fundamental limitation of pure behavior cloning.

## Results

- Less jittery than single-frame when tracking the ball
- But gets stuck when it drifts into unseen states (corners)

## Video

[04_multi_img_bc_policy.mp4](videos/04_multi_img_bc_policy.mp4)

## Status

**Abandoned.** Frame stacking inherits the same fundamental flaw as the dataset approach (step 4): a CNN has no way to distinguish temporal order from channel order. The corner-sticking covariate shift problem is real but is better addressed by DAgger (step 6) than by multi-frame input. DAgger on a single-frame BC policy outperforms this approach with fewer architectural assumptions.

---

Location: `policy/multi_img_bc_policy.py`
Checkpoint: `policy/checkpoints/multi_img_bc_policy.pth`
