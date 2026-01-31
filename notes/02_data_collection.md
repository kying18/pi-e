# Step 2: Data Collection

## Goal

Collect (observation, action) pairs from the expert to train a learned policy.

## Approach

Run the expert policy for many episodes, saving:
- **Observation**: RGB image (256x256x3) - what the learned policy will see
- **Action**: (dx, dy) command - what we want the policy to predict

## Dataset Stats

- Episodes: 100-1000
- Steps per episode: varies (~10-50)
- Total samples: ~5,000-50,000

## Video

[TODO: Add video of data collection]

---

Location: `scripts/collect_data.py`
Data: `data/expert_demos.npz`
