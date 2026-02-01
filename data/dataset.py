import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ObservationActionDataset(Dataset):
    """PyTorch Dataset for observation-action pairs with lazy preprocessing."""

    def __init__(self, observations, actions, img_size=128):
        self.observations = observations
        self.actions = actions.astype(np.float32)
        self.img_size = img_size

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]
        action = self.actions[idx]

        # Resize and normalize (lazy - only when requested)
        obs = cv2.resize(obs, (self.img_size, self.img_size))
        obs = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
        action = torch.from_numpy(action)

        return obs, action

class ActionChunkingDataset(Dataset):
    """PyTorch Dataset for action chunking."""

    def __init__(self, observations, actions, img_size=128, chunk_size=8):
        self.observations = observations
        self.actions = actions.astype(np.float32)
        self.chunk_size = chunk_size
        self.img_size = img_size

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = cv2.resize(self.observations[idx], (self.img_size, self.img_size))
        obs = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
        action = torch.from_numpy(action[idx:idx+self.chunk_size])

        return obs, action


def create_dataloaders(observations, actions, batch_size=64, train_split=0.8, max_samples=-1, chunk_size=8):
    """
    Create train and validation DataLoaders from observation-action data.

    Observations and actions can come from load_data() which already handles
    concatenating multiple datasets (expert + dagger).
    """
    # Sample if needed
    if max_samples > 0 and len(observations) > max_samples:
        indices = np.random.choice(len(observations), max_samples, replace=False)
        observations = observations[indices]
        actions = actions[indices]

    # Shuffle
    indices = np.random.permutation(len(observations))
    observations = observations[indices]
    actions = actions[indices]

    # Split
    train_size = int(len(observations) * train_split)
    train_obs, train_act = observations[:train_size], actions[:train_size]
    val_obs, val_act = observations[train_size:], actions[train_size:]

    # Create datasets and loaders
    train_dataset = ObservationActionDataset(train_obs, train_act)
    val_dataset = ObservationActionDataset(val_obs, val_act)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    return train_loader, val_loader

def create_action_chunking_dataloaders(observations, actions, batch_size=64, train_split=0.8, max_samples=-1, chunk_size=8):
    """
    Create train and validation DataLoaders from observation-action data for action chunking.
    """
    # Sample if needed
    if max_samples > 0 and len(observations) > max_samples:
        indices = np.random.choice(len(observations), max_samples, replace=False)
        observations = observations[indices]
        actions = actions[indices]

    # Shuffle
    indices = np.random.permutation(len(observations))
    observations = observations[indices]
    actions = actions[indices]

    # Split
    train_size = int(len(observations) * train_split)
    train_obs, train_act = observations[:train_size], actions[:train_size]
    val_obs, val_act = observations[train_size:], actions[train_size:]

    # Create datasets and loaders
    train_dataset = ActionChunkingDataset(train_obs, train_act, chunk_size=chunk_size)
    val_dataset = ActionChunkingDataset(val_obs, val_act, chunk_size=chunk_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    return train_loader, val_loader
