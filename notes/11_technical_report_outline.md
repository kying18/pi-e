# Step 11: Technical Report Outline

Use this outline for a 4-6 page lab-style report.

## 1. Problem and Motivation

- Goal: build toward generalist VLA/policy ingredients in a controlled setting.
- Why this environment: isolate architectural/control effects before scaling complexity.

## 2. Methods

### 2.1 Environment and Data
- Observation/action spec
- Expert and DAgger data generation
- Dataset sizes and train/val splits

### 2.2 Policy Architectures
- BC baseline
- Action chunking baseline
- ACT-style decoder
- ViT encoder + decoder

### 2.3 Evaluation Protocol
- Fixed seed / episodes / max steps
- Metrics and definitions
- Parameter counting method

## 3. Results

### 3.1 Main Table
- Steps, path inefficiency, completion, smoothness, params

### 3.2 Ablation A1: Capacity
- Performance vs parameter count

### 3.3 Ablation A2: Horizon
- Open-loop vs RH4 for chunking/ACT/ViT

### 3.4 Ablation A3: Data Scale
- 10k vs 20k sensitivity for ACT/ViT

## 4. Failure Analysis

- 3-5 representative failure cases with short notes
- Which metrics capture each failure mode

## 5. Discussion

- What transferred from toy setting to generalist policy intuition
- What did not transfer
- Concrete next steps (flow matching, language conditioning)

## 6. Limitations

- Environment simplicity
- No generative action model yet
- No language-conditioned policy yet

## 7. Reproducibility Appendix

- checkpoint names
- config table
- run IDs from `experiments/runs.csv`
