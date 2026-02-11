import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from .policy import Policy


class ActionChunkingPolicy(Policy):
    """Action chunking policy with simple CNN.

    Args:
        chunk_size: Number of actions to predict at once.
        actions_per_inference: How many actions to execute before re-predicting.
            - chunk_size (default): Open-loop execution of entire chunk.
            - 1: Receding horizon (predict every step, use first action only).
            - Any value in between for partial open-loop.
    """

    def __init__(self, use_checkpoint=False, checkpoint_name="action_chunking_policy",
                 chunk_size=8, actions_per_inference=None):
        super().__init__()
        self.device = self._get_device()
        self.chunk_size = chunk_size
        self.actions_per_inference = actions_per_inference if actions_per_inference else chunk_size

        # Internal state for action chunk execution
        self._action_chunk = None
        self._action_idx = 0

        policy_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_path = os.path.join(policy_dir, "checkpoints", f"{checkpoint_name}.pth")

        self.model = nn.Sequential(
            # input: 128x128x3
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output: 64x64x16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output: 32x32x32
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # Rather than predicting 2D action, predict action chunk
            nn.Linear(64, 2 * self.chunk_size),
        ).to(self.device)

        if use_checkpoint:
            self.model.load_state_dict(torch.load(self.checkpoint_path))

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def act(self, obs, env):
        """Return next action. Re-predicts when actions_per_inference exhausted."""
        # Need new prediction?
        if self._action_chunk is None or self._action_idx >= self.actions_per_inference:
            self._action_chunk = self._predict_chunk(obs)
            self._action_idx = 0

        action = self._action_chunk[self._action_idx]
        self._action_idx += 1
        return action

    def _predict_chunk(self, obs):
        """Run model inference to get action chunk."""
        self.model.eval()
        with torch.no_grad():
            obs = cv2.resize(obs, (128, 128))
            obs = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
            obs = obs.unsqueeze(0).to(self.device)
            chunk = self.model(obs).squeeze(0).view(self.chunk_size, 2)
            chunk = np.clip(chunk.cpu().numpy(), -5, 5)
        return chunk

    def reset(self):
        """Clear action chunk state. Call between episodes."""
        self._action_chunk = None
        self._action_idx = 0
