# Step 7: Action Chunking

## Goal

Instead of predicting one action at a time, predict a sequence of actions (chunk). This captures temporal structure and is a key idea from ACT (Action Chunking with Transformers) that carries into Pi0.

## Concept

**Before (single action):**
- Input: observation
- Output: action (2D)
- Execute, get new obs, repeat

**After (action chunk):**
- Input: observation
- Output: action sequence [a₁, a₂, ..., aₖ] (k×2D)
- Execute all k actions (open-loop), then get new obs and predict again

## Implementation

- Same CNN encoder as BC
- Output layer predicts `chunk_size * action_dim` values
- Reshape to `(chunk_size, 2)` for execution
- Dataset pairs each observation with next `chunk_size` actions

## Execution Modes

Controlled via `actions_per_inference` parameter:

**Open-loop:** `actions_per_inference=8` (default)
- Execute entire chunk blindly, then re-observe
- Simple, but errors compound over k steps

**Receding horizon:** `actions_per_inference=1`
- Predict chunk every step, only use first action
- More reactive, more compute

**Partial:** `actions_per_inference=4`
- Middle ground - predict 8, execute 4, then re-predict
- Good balance of reactivity and efficiency

## Episode Boundary Handling

**Problem:** Original data collection flattened all episodes into one array. Action chunks near episode boundaries would include actions from the next episode.

**Solution:**
1. Data collection now saves `episode_ends` array marking where each episode ends
2. Dataset pads chunks with zeros at episode boundaries instead of crossing into next episode
3. This teaches the model to predict "do nothing" near success states

## Training Setup

- chunk_size: 8
- Same hyperparameters as BC (lr=1e-3, batch_size=64)
- Data: expert + dagger (10k samples)
- Train/val split: 0.7
- Best checkpoint: epoch 6 (val_loss=3.25)
- Observed overfitting after epoch 6 (train kept dropping, val plateaued ~3.5)

## Results

**Without episode_ends (v1):**
- Policy works but shows jittering/overshooting
- Open-loop errors compound, expert data noise amplified

**With episode_ends + receding horizon (v2):**
- `actions_per_inference=4` smooths out jitter
- Cleaner data + more frequent re-prediction helps

**Honest assessment:**
- Episode ends training improved data quality
- Receding horizon (4) made execution noticeably smoother
- But overall performance is not dramatically better than BC + DAgger for this simple task
- Action chunking likely shines more on complex tasks with temporal structure (multi-step manipulation, longer horizons) rather than this simple ball-catching task

## Videos

| Description | Video |
|-------------|-------|
| Initial open-loop, some jitter | [07_action_chunking_policy.mp4](videos/07_action_chunking_policy.mp4) |
| Receding horizon (4), smoother | [07_action_chunking_policy_rh4.mp4](videos/07_action_chunking_policy_rh4.mp4) |
| With episode ends | [07_action_chunking_policy_episode_ends.mp4](videos/07_action_chunking_policy_episode_ends.mp4) |
| Episode ends + RH4 | [07_action_chunking_policy_rh4_episode_ends.mp4](videos/07_action_chunking_policy_rh4_episode_ends.mp4) |
| Episode ends + zero padding | [07_action_chunking_policy_episode_ends_padded.mp4](videos/07_action_chunking_policy_episode_ends_padded.mp4) |
| Episode ends + zero padding + RH4 | [07_action_chunking_policy_rh4_episode_ends_padded.mp4](videos/07_action_chunking_policy_rh4_episode_ends_padded.mp4) |

## Status

Complete

---

Location: `policy/action_chunking_policy.py`, `data/dataset.py`
