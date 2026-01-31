# Step 1: Expert Policy

## Goal

Create a "perfect" policy that always catches the ball, to generate training data.

## Approach

The expert has access to ground truth state that the learned policy won't have:
- `ball_pos` - current ball position
- `ball_vel` - current ball velocity
- `ee_pos` - current end-effector position

Strategy: move toward where the ball **will be**, not where it is:
```python
future_ball_pos = ball_pos + ball_vel
action = future_ball_pos - ee_pos
action = np.clip(action, -5, 5)
```

## Result

Expert catches the ball reliably in ~10-50 steps depending on starting positions.

## Video

[TODO: Add video of expert policy]

---

Location: `expert/expert_policy.py`
