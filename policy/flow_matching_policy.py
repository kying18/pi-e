import torch
from torch import nn, optim
import os
import cv2
import numpy as np

from .policy import Policy

class FlowModel(nn.Module):
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
        self.velocity_head = nn.Linear(self.embed_dim, 2)
        self.action_proj = nn.Linear(2, self.embed_dim)
    
    # embed t using sinusoidal embedding
    def sinusoidal_embedding(self, t, embed_dim):
        # t is a tensor of shape (B,) or scalar
        batch_size = t.shape[0] if t.ndim > 0 else 1
        # output (B, embed_dim)
        embeddings = torch.zeros((batch_size, embed_dim), device=t.device)

        # for all even dimensions, use sin(t/10000^(2i / embed_dim))
        embeddings[:, ::2] = torch.sin(t.unsqueeze(1) / 10000.0 ** (2 * torch.arange(embed_dim // 2, device=t.device) / embed_dim))
        # for all odd dimensions, use cos(t/10000^(2i / embed_dim))
        embeddings[:, 1::2] = torch.cos(t.unsqueeze(1) / 10000.0 ** (2 * torch.arange(embed_dim // 2, device=t.device) / embed_dim))

        return embeddings

    # obs is the image obs, t is the timestep (0-1), and x_t represents the noisy action chunk at timestep t
    def forward(self, obs, t, x_t):
        t_embed = self.sinusoidal_embedding(t, self.embed_dim)
        # output (B, embed_dim)

        patches = self.patch_embed(obs)
        # output (B, embed_dim, 128 // patch_size, 128 // patch_size)
        # output (B, 32, 8, 8)
        patches = patches.flatten(2).transpose(1, 2)
        # output (B, 64, 32)
        patches = patches + self.pos_embed
        # output (B, 64, 32)

        patches = self.transformer_encoder(patches)
        # output (B, 64, 32)

        # x_t is (B, chunk_size, 2)
        # tgt should be (B, chunk_size, embed_dim)
        tgt = self.action_proj(x_t) + t_embed[:, None, :]
        # output (B, chunk_size, embed_dim)

        x = self.transformer_decoder(tgt, patches)

        x = self.velocity_head(x)

        return x.flatten(1) # (B, chunk_size * 2)

# TODO: combine with ActionChunkingPolicy to avoid code duplication for act().
class FlowMatchingPolicy(Policy):
    def __init__(self, use_checkpoint=False, checkpoint_name="flow_matching_policy",
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

        self.model = FlowModel(chunk_size=chunk_size).to(self.device)

        if use_checkpoint:
            self.model.load_state_dict(torch.load(self.checkpoint_path))

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def sample_t_and_x_t(self, batch_size, actions):
        # Expect actions to be (B, chunk_size, 2)

        # Sample t uniformly from [0, 1]
        t = torch.rand(batch_size, device=self.device)

        # Sample x_0 from normal distribution with mean 0 and std 1
        x_0 = torch.normal(mean=0, std=1, size=(batch_size, self.chunk_size, 2), device=self.device)

        # Compute x_t using x_0, x_1, and t (x_0 is the initial action chunk, x_1 is the action chunk at t=1)
        x_t = (1 - t)[:, None, None] * x_0 + t[:, None, None] * actions

        return t, x_t, x_0

    def train(self, train_loader, val_loader, num_epochs=100):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for obs, actions in train_loader:
                batch_size = obs.shape[0]

                obs, actions = obs.to(self.device), actions.to(self.device)
                unflattened_actions = actions.unflatten(1, (self.chunk_size, 2))
                t, x_t, x_0 = self.sample_t_and_x_t(batch_size, unflattened_actions)

                self.optimizer.zero_grad()
                outputs = self.model(obs, t, x_t)
                velocity = unflattened_actions - x_0
                loss = self.loss_fn(outputs, velocity.flatten(1))
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for obs, actions in val_loader:
                    batch_size = obs.shape[0]
                    obs, actions = obs.to(self.device), actions.to(self.device)
                    unflattened_actions = actions.unflatten(1, (self.chunk_size, 2))
                    t, x_t, x_0 = self.sample_t_and_x_t(batch_size, unflattened_actions)
                    outputs = self.model(obs, t, x_t)
                    velocity = unflattened_actions - x_0
                    val_loss += self.loss_fn(outputs, velocity.flatten(1)).item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"Saved checkpoint to {self.checkpoint_path}")

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

            x_t = torch.normal(mean=0, std=1, size=(1, self.chunk_size, 2), device=self.device)
            dt = 0.1
            for t in torch.arange(0, 1 + dt, dt, device=self.device):
                # t is a scalar (0d tensor), so we need to unsqueeze it to make it a 1d tensor
                predicted_velocity = self.model(obs, t.unsqueeze(0), x_t).view(1, self.chunk_size, 2)
                x_t = x_t + predicted_velocity * dt

            chunk = x_t.squeeze(0).view(self.chunk_size, 2)
            chunk = np.clip(chunk.cpu().numpy(), -5, 5)

        return chunk

    def reset(self):
        """Clear action chunk state. Call between episodes."""
        self._action_chunk = None
        self._action_idx = 0