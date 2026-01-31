# pi-e

Exploring Pi0 policies with experience.

## Goal

Implement Physical Intelligence's Pi0 architecture from scratch, building up incrementally:

1. Behavior cloning (supervised learning)
2. Action chunking (predict action sequences)
3. Transformer-based policy
4. Flow matching for action generation
5. Full Pi0-style vision-language-action model

## Environment

Simple 2D ball interception task:
- Red ball bounces around the screen
- Blue end-effector (robot) must intercept it
- Observation: 256x256 RGB image
- Action: (dx, dy) velocity command

## Structure

```
pi/
├── env/                  # Environment
│   └── moving_object.py
├── expert/               # Expert policy for demonstrations
│   └── expert_policy.py
├── policy/               # Learned policies
├── scripts/              # Training and evaluation
├── data/                 # Collected demonstrations
└── visualize.py          # Visualization
```

## Usage

Visualize with expert policy:
```bash
python visualize.py
```

## Progress

- [x] Environment
- [x] Expert policy
- [ ] Data collection
- [ ] Behavior cloning policy
- [ ] Action chunking
- [ ] Transformer policy
- [ ] Flow matching
- [ ] Pi0-style architecture
