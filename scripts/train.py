import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.bc_policy import BcPolicy
from policy.action_chunking_policy import ActionChunkingPolicy
from policy.act_policy import ActPolicy
from data.dataset import create_dataloaders, create_action_chunking_dataloaders


def load_data(*paths):
    """Load and concatenate data from one or more .npz files."""
    all_obs = []
    all_actions = []
    all_episode_ends = []
    offset = 0

    for path in paths:
        if not os.path.exists(path):
            print(f"Skipping {path} (not found)")
            continue
        data = np.load(path)
        all_obs.append(data["observations"])
        all_actions.append(data["actions"])

        # Handle episode_ends if present (adjust indices by offset)
        if "episode_ends" in data:
            episode_ends = data["episode_ends"] + offset
            all_episode_ends.extend(episode_ends.tolist())

        offset += len(data["observations"])
        print(f"Loaded {path}: {data['observations'].shape[0]} samples")

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    episode_ends = all_episode_ends if all_episode_ends else None
    print(f"Total: {len(observations)} samples")

    return observations, actions, episode_ends


def train_bc_policy(data_paths=None, max_samples=5000, checkpoint_name=None):
    if data_paths is None:
        data_paths = ["data/datasets/expert_data.npz"]
    observations, actions, _ = load_data(*data_paths)

    train_loader, val_loader = create_dataloaders(observations, actions, batch_size=64, train_split=0.7, max_samples=max_samples)
    if checkpoint_name is None:
        policy = BcPolicy(use_checkpoint=False)
    else:
        policy = BcPolicy(use_checkpoint=False, checkpoint_name=checkpoint_name)
    policy.train(train_loader, val_loader)


def train_bc_with_dagger(data_paths=None, max_samples=5000, checkpoint_name=None):
    if data_paths is None:
        data_paths = ["data/datasets/expert_data.npz", "data/datasets/expert_data_dagger.npz"]
    observations, actions, _ = load_data(*data_paths)

    train_loader, val_loader = create_dataloaders(observations, actions, batch_size=64, train_split=0.7, max_samples=max_samples)

    if checkpoint_name is None:
        policy = BcPolicy(use_checkpoint=False)
    else:
        policy = BcPolicy(use_checkpoint=False, checkpoint_name=checkpoint_name)
    policy.train(train_loader, val_loader)

def train_action_chunking_policy(data_paths=None, max_samples=5000, checkpoint_name=None):
    if data_paths is None:
        data_paths = ["data/datasets/expert_data_with_episode_ends.npz", "data/datasets/expert_data_bc_dagger_with_episode_ends.npz"]
    observations, actions, episode_ends = load_data(*data_paths)

    train_loader, val_loader = create_action_chunking_dataloaders(
        observations, actions, batch_size=64, train_split=0.7,
        max_samples=max_samples, episode_ends=episode_ends
    )

    if checkpoint_name is None:
        policy = ActionChunkingPolicy(use_checkpoint=False)
    else:
        policy = ActionChunkingPolicy(use_checkpoint=False, checkpoint_name=checkpoint_name)
    policy.train(train_loader, val_loader)

def train_act_policy(data_paths=None, max_samples=5000, checkpoint_name=None):
    if data_paths is None:
        data_paths = ["data/datasets/expert_data_with_episode_ends.npz", "data/datasets/expert_data_bc_dagger_with_episode_ends.npz"]
    observations, actions, episode_ends = load_data(*data_paths)

    train_loader, val_loader = create_action_chunking_dataloaders(
        observations, actions, batch_size=64, train_split=0.7,
        max_samples=max_samples, episode_ends=episode_ends
    )

    if checkpoint_name is None:
        policy = ActPolicy(use_checkpoint=False)
    else:
        policy = ActPolicy(use_checkpoint=False, checkpoint_name=checkpoint_name)
    policy.train(train_loader, val_loader)

if __name__ == "__main__":
    # train_bc_policy(max_samples=10000)
    # train_bc_with_dagger(max_samples=10000)
    # train_action_chunking_policy(max_samples=10000, checkpoint_name="episode_ends_padded_action_chunking_policy")
    train_act_policy(max_samples=10000, checkpoint_name="act_policy_small")