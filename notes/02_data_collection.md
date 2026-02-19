# Step 2: Data Collection

## Goal

Collect (observation, action) pairs from the expert to train a learned policy.

## Approach

Run the expert policy for many episodes, saving:
- **Observation**: RGB image (256x256x3) - what the learned policy will see
- **Action**: (dx, dy) command - what we want the policy to predict

## Dataset Stats

- Steps per episode: ~10-50 (expert captures quickly)
- BC/ACT training: 10k samples (expert + DAgger)
- ViT training: 20k samples (ViT needs more data; no CNN inductive bias)

## Video

See `notes/videos/01_expert_policy.mp4` for the expert generating demonstrations.

---

Location: `scripts/collect_data.py`
Data: `data/expert_demos.npz`
