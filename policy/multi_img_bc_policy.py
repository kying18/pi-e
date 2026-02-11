import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import policy.policy as policy
import os

class MultiImgBcPolicy(policy.Policy):
    def __init__(self, use_checkpoint=False, checkpoint_name="multi_img_bc_policy"):
        super().__init__()
        if torch.backends.mps.is_available():
          self.device = torch.device("mps")
        elif torch.cuda.is_available():
          self.device = torch.device("cuda")
        else:
          self.device = torch.device("cpu")
        
        # Set checkpoint path relative to this file
        policy_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_path = os.path.join(policy_dir, "checkpoints", f"{checkpoint_name}.pth")
        self.model = nn.Sequential(
            # 3 images stacked as input: 128x128x9
            nn.Conv2d(9, 16, kernel_size=3, stride=1, padding=1),
            # output: 128x128x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output: 64x64x16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # output: 64x64x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output: 32x32x32
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            # output: 128
            nn.ReLU(),
            nn.Linear(128, 64),
            # output: 64
            nn.ReLU(),
            nn.Linear(64, 2),
            # output: 2
          ).to(self.device)
        if use_checkpoint:
          self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.loss_fn = nn.MSELoss()

    def train(self, observations, actions, num_epochs=100, batch_size=64, train_split=0.8, max_samples=-1):
      # create checkpoints directory if it doesn't exist
      os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

      # Sample first to reduce memory
      if max_samples > 0 and len(observations) > max_samples:
        indices = np.random.choice(len(observations), max_samples, replace=False)
        observations = observations[indices]
        actions = actions[indices]
      
      # Shuffle
      indices = np.random.permutation(len(observations))
      observations = observations[indices]
      actions = actions[indices]
      
      # Split into train and validation BEFORE converting to torch
      train_size = int(len(observations) * train_split)
      train_obs_np = observations[:train_size]
      train_act_np = actions[:train_size]
      valid_obs_np = observations[train_size:]
      valid_act_np = actions[train_size:]
      
      # Resize and convert validation set (keep in memory for evaluation)
      print("Processing validation set...")
      valid_obs_resized = np.array([cv2.resize(obs, (128, 128)) for obs in valid_obs_np])
      valid_obs = torch.from_numpy(valid_obs_resized).permute(0, 3, 1, 2).float() / 255.0
      valid_act = torch.from_numpy(valid_act_np).float()
      
      # Don't keep training data in memory as tensors - process per batch
      del observations, actions  # Free memory

      print(f"Starting training with {len(train_obs_np)} training samples and {len(valid_obs)} validation samples...")   

      # batch data
      best_loss = float('inf')
      for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        # Process training data in batches (don't keep all in GPU memory)
        for i in range(0, len(train_obs_np), batch_size):
          # Resize and convert batch on-the-fly
          batch_obs_np = train_obs_np[i:i+batch_size]
          batch_obs_resized = np.array([cv2.resize(obs, (128, 128)) for obs in batch_obs_np])
          batch_obs = torch.from_numpy(batch_obs_resized).permute(0, 3, 1, 2).float().to(self.device) / 255.0
          batch_actions = torch.from_numpy(train_act_np[i:i+batch_size]).float().to(self.device)

          # forward pass
          outputs = self.model(batch_obs)
          loss = self.loss_fn(outputs, batch_actions)

          # backward pass
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          total_loss += loss.item()
          num_batches += 1

        avg_train_loss = total_loss / num_batches

        # evaluate on validation set (in batches to avoid OOM)
        self.model.eval()
        with torch.no_grad():
          valid_loss = 0
          valid_batches = 0
          for i in range(0, len(valid_obs), batch_size):
            batch_valid_obs = valid_obs[i:i+batch_size].to(self.device)
            batch_valid_act = valid_act[i:i+batch_size].to(self.device)
            valid_pred = self.model(batch_valid_obs)
            valid_loss += self.loss_fn(valid_pred, batch_valid_act).item()
            valid_batches += 1
          valid_loss = valid_loss / valid_batches
        self.model.train()
        
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {valid_loss:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(self.model.state_dict(), self.checkpoint_path)
            print(f"Saved model to {self.checkpoint_path}")

    def act(self, obs, env):
      # set model to evaluation mode
      self.model.eval()
      with torch.no_grad():
        resized_obs = cv2.resize(obs, (128, 128))
        current_obs = torch.from_numpy(resized_obs).permute(2, 0, 1).float() / 255.0
        current_obs = current_obs.unsqueeze(0).to(self.device)
        action = self.model(current_obs).squeeze(0)
        action = np.clip(action.cpu().numpy(), -5, 5)
      return action
    
