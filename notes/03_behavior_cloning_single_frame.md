# Step 3: Behavior Cloning (Single Frame)

## Goal

Train a neural network to predict actions from images alone (no access to ground truth state).

## Approach

Simple CNN architecture:
- Input: 128x128x3 RGB image
- Conv layers extract visual features
- Linear layers predict 2D action

Training:
- MSE loss between predicted and expert actions
- Adam optimizer
- Train/validation split (70/30)

## Result

**Problem**: Policy was jittery when close to the ball. It could find the ball but didn't know which direction to move.

**Why**: A single frame shows WHERE the ball is, but not WHERE IT'S GOING. Without velocity information, the policy can't predict the interception point.

## Video

[03_bc_policy.mp4](videos/03_bc_policy.mp4)

---

Location: `policy/bc_policy.py`
Checkpoint: `policy/checkpoints/bc_policy.pth`
