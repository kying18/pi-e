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

    def __init__(self, observations, actions, img_size=128, chunk_size=8, episode_ends=None):
        self.chunk_size = chunk_size
        self.img_size = img_size
        self.observations = observations
        actions = actions.astype(np.float32)

        # Build episode end set for quick lookup
        episode_end_set = set(episode_ends) if episode_ends is not None else set()

        # Create chunks, padding with zeros at episode boundaries
        self.action_chunks = []
        for i in range(len(observations)):
            # Find how many actions we can take before hitting episode end or data end
            actions_available = 0
            for j in range(chunk_size):
                if i + j >= len(actions):
                    break
                actions_available += 1
                # Stop if this action is the last in an episode
                if (i + j + 1) in episode_end_set:
                    break

            # Build chunk: real actions + zero padding
            if actions_available == chunk_size:
                chunk = actions[i:i+chunk_size]
            else:
                chunk = actions[i:i+actions_available]
                padding = np.zeros((chunk_size - actions_available, 2), dtype=np.float32)
                chunk = np.concatenate([chunk, padding])

            self.action_chunks.append(chunk)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = cv2.resize(self.observations[idx], (self.img_size, self.img_size))
        obs = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
        action = torch.from_numpy(self.action_chunks[idx].flatten())

        return obs, action


def create_dataloaders(observations, actions, batch_size=64, train_split=0.8, max_samples=-1):
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

def create_action_chunking_dataloaders(observations, actions, batch_size=64, train_split=0.8, max_samples=-1, chunk_size=8, episode_ends=None):
    """
    Create train and validation DataLoaders from observation-action data for action chunking.

    NOTE: Don't shuffle before chunking - data must be sequential.
    DataLoader shuffle=True handles shuffling of (obs, chunk) pairs.
    """
    # Split first (keep sequential within each split)
    train_size = int(len(observations) * train_split)
    train_obs, train_act = observations[:train_size], actions[:train_size]
    val_obs, val_act = observations[train_size:], actions[train_size:]

    # Split episode_ends for train/val
    train_episode_ends = None
    val_episode_ends = None
    if episode_ends is not None:
        # Episode ends within train split
        train_episode_ends = [e for e in episode_ends if e <= train_size]
        # Episode ends within val split, adjusted to local indices
        val_episode_ends = [e - train_size for e in episode_ends if e > train_size]

    # Create datasets (chunking happens here on sequential data)
    train_dataset = ActionChunkingDataset(train_obs, train_act, chunk_size=chunk_size, episode_ends=train_episode_ends)
    val_dataset = ActionChunkingDataset(val_obs, val_act, chunk_size=chunk_size, episode_ends=val_episode_ends)

    # Sample after chunking if needed (maintain train/val split ratio)
    if max_samples > 0 and len(train_dataset) + len(val_dataset) > max_samples:
        total_samples = len(train_dataset) + len(val_dataset)
        ratio = len(train_dataset) / total_samples
        
        n_train = int(max_samples * ratio)
        n_val = max_samples - n_train
        
        # Sample from each dataset separately
        train_indices = np.random.choice(len(train_dataset), min(n_train, len(train_dataset)), replace=False)
        val_indices = np.random.choice(len(val_dataset), min(n_val, len(val_dataset)), replace=False)
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # DataLoader shuffles the chunked pairs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    return train_loader, val_loader
