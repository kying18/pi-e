# pi-e: Policy with Experience

## Project Goal

Build toward a generalist VLA/policy with explicit memory, by implementing each core ingredient from scratch — BC through flow matching — in a controlled simulation where every architectural change is attributable.

## Learning Path

1. [Expert Policy](01_expert_policy.md) - Ground truth baseline ✓
2. [Data Collection](02_data_collection.md) - Gathering demonstrations ✓
3. [BC Single Frame](03_behavior_cloning_single_frame.md) - First learned policy ✓
4. [Multi-Frame Stacking](04_multi_frame_stacking.md) - Abandoned (see below)
5. [BC Multi-Frame](05_multi_frame_bc.md) - Abandoned (see below)
6. [DAgger](06_dagger.md) - Fix covariate shift ✓
7. [Action Chunking](07_action_chunking.md) - Open-loop + receding horizon ✓
8. [Transformer Decoder (ACT)](08_transformer_decoder.md) - Action queries + CNN encoder ✓
9. [ViT Encoder](09_vit_encoder.md) - Patch-based encoder + transformer decoder ✓
10. [Baseline Metrics](10_baseline_metrics.md) - Evaluation harness ✓
11. Flow Matching - (next)
12. Language Conditioning (VLA) - (upcoming)
13. Memory Experiments - (upcoming)

## Key Insights

1. **Multi-frame stacking is a dead end**: Frame stacking (concatenating frames along channels) is a 2015-era technique from DQN. It doesn't provide proper temporal inductive bias — the CNN treats all channels equally. Modern approaches use attention over frame sequences or action chunking instead.

2. **Expert data lacks edge cases**: The expert is good and stays centered, so expert data has few samples near edges/corners. DAgger fixes this by running the imperfect learned policy (which drifts to edges) and labeling those states with expert recovery actions.

3. **DAgger enables edge avoidance**: After DAgger training, the policy avoids edges because it now has training data showing how to recover from edge situations.

4. **Receding horizon matters more than architecture**: Open-loop chunk execution is fragile on dynamic targets. Re-planning every 4 steps (RH4) recovers most of the performance gap at no architectural cost.

5. **Transformer decoder is dramatically more parameter-efficient**: ACT-style learned action queries outperform flatten+MLP BC at 60× fewer parameters. The decoder adds structure the MLP cannot learn.

## Videos

See `notes/videos/` for demonstrations:
- `00_random_policy.mp4` - Random baseline
- `01_expert_policy.mp4` - Expert policy
- `03_bc_policy.mp4` - BC single-frame
- `04_multi_img_bc_policy.mp4` - Multi-frame BC (abandoned)
- `06_bc_policy_dagger.mp4` - BC + DAgger
- `07_action_chunking_policy_rh4.mp4` - Action chunking (RH4)
- `08_act_policy_small_rh4.mp4` - ACT (RH4)
- `09_vit_policy_patch16_rh4.mp4` - ViT (RH4)
