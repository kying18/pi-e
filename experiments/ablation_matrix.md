# Ablation Matrix (Week 1)

Goal: produce a tight, fair set of comparisons that hiring managers can audit quickly.

## A1: Capacity Ablation (Parameter Efficiency)

Question: how much performance do we buy with model size?

| ID | Policy | Variant | Expected Params | Notes |
|---|---|---|---:|---|
| A1-1 | BC | current | ~4.2M | high-capacity baseline |
| A1-2 | BC | compact CNN/MLP | TBD | reduced channels + smaller head |
| A1-3 | ACT | RH4 | ~69K | token-based decoder |
| A1-4 | ViT | RH4 | ~287K | encoder+decoder transformer |

Primary metrics: steps to capture, path inefficiency, completion rate.  
Secondary: direction consistency, magnitude continuity.

## A2: Horizon Ablation (Control Strategy)

Question: how much does execution horizon matter vs architecture?

| ID | Policy | Mode | actions_per_inference |
|---|---|---|---:|
| A2-1 | Action Chunking | Open-loop | 8 |
| A2-2 | Action Chunking | RH4 | 4 |
| A2-3 | ACT | Open-loop | 8 |
| A2-4 | ACT | RH4 | 4 |
| A2-5 | ViT | Open-loop | 8 |
| A2-6 | ViT | RH4 | 4 |

Primary metrics: steps to capture, completion rate.  
Secondary: smoothness metrics, trajectory length.

## A3: Data Scale Ablation (Transformer Data Sensitivity)

Question: do transformer policies need more demonstrations in this environment?

| ID | Policy | Train Samples | Notes |
|---|---|---:|---|
| A3-1 | ACT | 10k | baseline |
| A3-2 | ACT | 20k | scale effect |
| A3-3 | ViT | 10k | expected weaker |
| A3-4 | ViT | 20k | expected stronger |

Primary metrics: all rollout metrics + val loss trend snapshot.

## Evaluation Standard (Use for All Rows)

- seed: 42 (plus optional multi-seed extension)
- episodes: 300
- max steps: 100
- same environment settings across all runs
- store final metrics in `experiments/runs.csv`

