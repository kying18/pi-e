# Pi-E: Exploring Pi0 with Experience

## Project Goal

Implement Physical Intelligence's Pi0 architecture from scratch, building up incrementally from simple behavior cloning to the full vision-language-action model.

## Learning Path

1. [Expert Policy](01_expert_policy.md) - Ground truth baseline
2. [Data Collection](02_data_collection.md) - Gathering demonstrations
3. [BC Single Frame](03_behavior_cloning_single_frame.md) - First learned policy
4. [Multi-Frame Stacking](04_multi_frame_stacking.md) - Adding temporal info
5. [BC Multi-Frame](05_multi_frame_bc.md) - Improved policy, but gets stuck in corners
6. [DAgger](06_dagger.md) - Fix covariate shift (next)
7. Action Chunking - (upcoming)
8. Transformer Policy - (upcoming)
9. Flow Matching - (upcoming)
10. Full Pi0 - (upcoming)

## Key Insights So Far

1. **Single frame lacks velocity**: A static image shows where the ball IS, not where it's GOING. Stacking frames provides temporal context.

2. **Covariate shift**: Pure behavior cloning only trains on expert states. When the learned policy drifts into unseen states (like corners), it doesn't know how to recover. DAgger addresses this by collecting data from the learned policy's actual trajectory.

## Videos

See `notes/videos/` for demonstrations of each stage.
