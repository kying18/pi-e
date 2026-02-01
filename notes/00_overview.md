# Pi-E: Exploring Pi0 with Experience

## Project Goal

Implement Physical Intelligence's Pi0 architecture from scratch, building up incrementally from simple behavior cloning to the full vision-language-action model.

## Learning Path

1. [Expert Policy](01_expert_policy.md) - Ground truth baseline ✓
2. [Data Collection](02_data_collection.md) - Gathering demonstrations ✓
3. [BC Single Frame](03_behavior_cloning_single_frame.md) - First learned policy ✓
4. [Multi-Frame Stacking](04_multi_frame_stacking.md) - Adding temporal info (abandoned)
5. [BC Multi-Frame](05_multi_frame_bc.md) - Abandoned (see below)
6. [DAgger](06_dagger.md) - Fix covariate shift ✓
7. Action Chunking - (next)
8. Transformer Policy - (upcoming)
9. Flow Matching - (upcoming)
10. Full Pi0 - (upcoming)

## Key Insights

1. **Multi-frame stacking is a dead end**: Frame stacking (concatenating frames along channels) is a 2015-era technique from DQN. It doesn't provide proper temporal inductive bias - the CNN treats all channels equally. Modern approaches use attention over frame sequences or action chunking instead.

2. **Expert data lacks edge cases**: The expert is good and stays centered, so expert data has few samples near edges/corners. DAgger fixes this by running the imperfect learned policy (which drifts to edges) and labeling those states with expert recovery actions.

3. **DAgger enables edge avoidance**: After DAgger training, the policy avoids edges because it now has training data showing how to recover from edge situations.

## Videos

See `notes/videos/` for demonstrations:
- `00_random_policy.mp4` - Random baseline
- `01_expert_policy.mp4` - Expert policy
- `03_bc_policy.mp4` - BC single-frame
- `04_multi_img_bc_policy.mp4` - Multi-frame BC (abandoned)
- `06_bc_policy_dagger.mp4` - BC + DAgger
