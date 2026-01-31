# Step 4: Multi-Frame Dataset

## Goal

Give the policy temporal information so it can infer ball velocity.

## Approach

Stack n consecutive frames (t-2, t-1, t) as input:
- Instead of (H, W, 3), input becomes (H, W, 3*n)
- Stacked along channel dimension: 3 RGB frames = 9 channels
- Model can now see motion by comparing frames

Data collection changes:
- Maintain frame buffer during episode
- Only save when buffer is full
- Initialize buffer with first frame (not zeros)

## Why This Helps

With multiple frames, the model can learn:
- Frame t vs t-1 difference = ball velocity
- Frame t-1 vs t-2 difference = confirms velocity direction
- Better prediction of where to intercept

---

Location: `scripts/collect_data.py` (n_frames parameter)
Data: `data/expert_data_n_frames_3.npz`
