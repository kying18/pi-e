# Step 6: DAgger (Dataset Aggregation)

## Goal

Fix covariate shift by teaching the policy to recover from mistakes.

## The Problem with Behavior Cloning

Pure BC only trains on expert states. When the learned policy makes a mistake and enters a new state, it doesn't know how to recover because it never trained on that situation.

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

## Implementation Plan

- [ ] Modify data collection to run learned policy
- [ ] At each step, also record what expert would do
- [ ] Aggregate with existing dataset
- [ ] Retrain

## Status

TODO

---

Location: `scripts/collect_dagger.py` (to be created)
