# Step 6: DAgger (Dataset Aggregation)

## Goal

Generate more training samples in edge/corner cases that the expert rarely visits.

## The Problem with Behavior Cloning

The expert policy is good - it stays centered and catches the ball efficiently. This means expert data has few samples near edges/corners. When the learned policy (which is imperfect) drifts toward edges, it doesn't know how to recover because it never trained on those situations.

## DAgger Algorithm

```
1. Train initial policy π on expert data D
2. Repeat:
   a. Run π to collect new states
   b. Query expert for actions at those states
   c. Add (new_state, expert_action) to D
   d. Retrain π on full D
```

The key insight: we're collecting states from the *learned* policy but labels from the *expert*. This covers the states the learned policy actually visits.

## Implementation

- [x] Modify data collection to run learned policy
- [x] At each step, also record what expert would do
- [x] Aggregate with existing dataset
- [x] Retrain from scratch

## Training Setup

Same hyperparameters for BC and BC+DAgger for fair comparison:
- 10k datapoints
- Learning rate: 1e-3
- Train/val split: 70/30
- Batch size: 64
- Epochs: 45

## Results

DAgger improves policy robustness. Notable observation: **BC+DAgger tends to avoid edges**, likely because DAgger exposes the model to recovery situations near boundaries that pure expert data doesn't cover.

| Policy | Video |
|--------|-------|
| BC (single-frame) | [03_bc_policy.mp4](videos/03_bc_policy.mp4) |
| BC + DAgger | [06_bc_policy_dagger.mp4](videos/06_bc_policy_dagger.mp4) |

## Status

Complete ✓

---

Location: `scripts/collect_data.py` (`collect_dagger_data` function)
