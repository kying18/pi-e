import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.bc_policy import BcPolicy
from data.dataset import create_dataloaders


def load_data(*paths):
    """Load and concatenate data from one or more .npz files."""
    all_obs = []
    all_actions = []

    for path in paths:
        if not os.path.exists(path):
            print(f"Skipping {path} (not found)")
            continue
        data = np.load(path)
        all_obs.append(data["observations"])
        all_actions.append(data["actions"])
        print(f"Loaded {path}: {data['observations'].shape[0]} samples")

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    print(f"Total: {len(observations)} samples")

    return observations, actions


def train_bc_policy(data_paths=None, max_samples=5000):
    if data_paths is None:
        data_paths = ["data/datasets/expert_data.npz"]
    observations, actions = load_data(*data_paths)

    train_loader, val_loader = create_dataloaders(observations, actions, batch_size=64, train_split=0.7, max_samples=max_samples)
    policy = BcPolicy(use_best_model=False)
    policy.train(train_loader, val_loader)


def train_bc_with_dagger(data_paths=None, max_samples=5000):
    if data_paths is None:
        data_paths = ["data/datasets/expert_data.npz", "data/datasets/expert_data_dagger.npz"]
    observations, actions = load_data(*data_paths)

    train_loader, val_loader = create_dataloaders(observations, actions, batch_size=64, train_split=0.7, max_samples=max_samples)

    policy = BcPolicy(use_best_model=False)
    policy.train(train_loader, val_loader)


if __name__ == "__main__":
    # train_bc_policy(max_samples=10000)
    train_bc_with_dagger(max_samples=10000)