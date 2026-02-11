import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import torch.optim as optim

from .policy import Policy

class ViTModel(nn.Module):
    def __init__(self, chunk_size=8):
        super().__init__()
        self.embed_dim = 64
        self.patch_size = 16
        self.patch_embed = nn.Sequential(
          nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size),
          nn.ReLU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, (128 // self.patch_size) ** 2, self.embed_dim) * 0.02)
        self.transformer_encoder = nn.TransformerEncoder(
          nn.TransformerEncoderLayer(d_model=self.embed_dim, dim_feedforward=256, nhead=4, batch_first=True),
          num_layers=2,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.embed_dim, dim_feedforward=256, nhead=2, batch_first=True),
            num_layers=2,
        )
        self.action_chunk_head = nn.Linear(self.embed_dim, 2) # one action per query
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, self.embed_dim) * 0.02)
    
    def forward(self, x):
        patches = self.patch_embed(x)
        # output (B, embed_dim, 128 // patch_size, 128 // patch_size)
        # output (B, 32, 8, 8)
        patches = patches.flatten(2).transpose(1, 2)
        # output (B, 64, 32)
        patches = patches + self.pos_embed
        # output (B, 64, 32)

        # encode patches
        patches = self.transformer_encoder(patches)
        # output (B, 64, 32)

        batch_size = x.shape[0]
        tgt = self.action_queries.expand(batch_size, -1, -1)
        # output (B, chunk_size, embed_dim)
  
        x = self.transformer_decoder(tgt, patches)
        # output (B, chunk_size, embed_dim)
        x = self.action_chunk_head(x)
        # output (B, chunk_size, 2)
        return x.flatten(1) # (B, chunk_size * 2)


# TODO: combine with ActionChunkingPolicy to avoid code duplication for act().
class ViTPolicy(Policy):
    """Action chunking policy with transformer decoder.

    Args:
        chunk_size: Number of actions to predict at once.
        actions_per_inference: How many actions to execute before re-predicting.
            - chunk_size (default): Open-loop execution of entire chunk.
            - 1: Receding horizon (predict every step, use first action only).
            - Any value in between for partial open-loop.
    """
    def __init__(self, use_checkpoint=False, checkpoint_name="vit_policy",
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

        self.model = ViTModel(chunk_size=chunk_size).to(self.device)

        if use_checkpoint:
            self.model.load_state_dict(torch.load(self.checkpoint_path))

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

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

if __name__ == "__main__":
    model = ViTModel(chunk_size=8)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")