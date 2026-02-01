import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from policy.policy import Policy


class BcPolicy(Policy):
    """Behavior cloning policy with simple CNN."""

    def __init__(self, use_checkpoint=False, checkpoint_name="bc_policy"):
        super().__init__()
        self.device = self._get_device()

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
            nn.Linear(64, 2),
        ).to(self.device)

        if use_checkpoint:
            self.model.load_state_dict(torch.load(self.checkpoint_path))

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def act(self, obs):
        self.model.eval()
        with torch.no_grad():
            obs = cv2.resize(obs, (128, 128))
            obs = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
            obs = obs.unsqueeze(0).to(self.device)
            action = self.model(obs).squeeze(0)
            action = np.clip(action.cpu().numpy(), -5, 5)
        return action
