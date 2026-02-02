import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

from policy.policy import Policy

class ActModel(nn.Module):
    def __init__(self, chunk_size=8):
        super().__init__()
        self.embed_dim = 32
        self.cnn_encoder = nn.Sequential(
            # input: 128x128x3
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            # output: 32x32x32
            nn.Conv2d(32, self.embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output: 16x16x32
        )
        # self.cnn_encoder = nn.Sequential(
        #   # input: 128x128x3
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     # output: 64x64x16
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     # output: 32x32x32
        # )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.embed_dim, dim_feedforward=256, nhead=2, batch_first=True),
            num_layers=2,
        )
        self.action_chunk_head = nn.Linear(self.embed_dim, 2) # one action per query
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, self.embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 16*16, self.embed_dim) * 0.02)
    
    def forward(self, x):
        memory = self.cnn_encoder(x)
        # output (B, 32, 16, 16)
        memory = memory.flatten(2).transpose(1, 2)
        # output (B, 32, 256)
        memory = memory + self.pos_embed

        batch_size = x.shape[0]
        tgt = self.action_queries.expand(batch_size, -1, -1)
        # output (B, chunk_size, embed_dim)

        x = self.transformer_decoder(tgt, memory)
        # output (B, chunk_size, embed_dim)
        x = self.action_chunk_head(x)
        # output (B, chunk_size, 2)
        return x.flatten(1) # (B, chunk_size * 2)


# TODO: combine with ActionChunkingPolicy to avoid code duplication for act().
class ActPolicy(Policy):
    """Action chunking policy with transformer decoder.

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

        self.model = ActModel(chunk_size=chunk_size).to(self.device)

        if use_checkpoint:
            self.model.load_state_dict(torch.load(self.checkpoint_path))

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def act(self, obs):
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

if __name__ == "__main__":
    model = ActModel(chunk_size=8)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")