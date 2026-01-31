import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import policy.policy as policy
import os

class BcPolicy(policy.Policy):
    def __init__(self, use_best_model=False):
        super().__init__()
        if torch.backends.mps.is_available():
          self.device = torch.device("mps")
        elif torch.cuda.is_available():
          self.device = torch.device("cuda")
        else:
          self.device = torch.device("cpu")
        
        # Set checkpoint path relative to this file
        policy_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_path = os.path.join(policy_dir, "checkpoints", "bc_policy.pth")
        self.model = nn.Sequential(
            # input: 128x128x3
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
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
        if use_best_model:
          self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def train(self, observations, actions, num_epochs=100, batch_size=64, train_split=0.7, max_samples=-1):
      # create checkpoints directory if it doesn't exist
      os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

      if max_samples > 0:
        observations = observations[:max_samples]
        actions = actions[:max_samples]

      # shuffle observations and actions and split into train and validation
      indices = np.random.permutation(len(observations))
      # resize observations to 128x128
      resized_obs = [cv2.resize(obs, (128, 128)) for obs in observations]
      resized_obs = np.array(resized_obs)
      observations = torch.from_numpy(resized_obs[indices]).permute(0, 3, 1, 2).float() / 255.0
      actions = torch.from_numpy(actions[indices]).float()
      train_size = int(len(observations) * train_split)
      train_obs = observations[:train_size]
      train_act = actions[:train_size]
      valid_obs = observations[train_size:]
      valid_act = actions[train_size:]

      print(f"Starting training with {len(train_obs)} training samples and {len(valid_obs)} validation samples...")   

      # batch data
      best_loss = float('inf')
      for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_obs), batch_size):
          batch_obs = train_obs[i:i+batch_size].to(self.device)
          batch_actions = train_act[i:i+batch_size].to(self.device)

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

        # evaluate on validation set
        with torch.no_grad():
          valid_obs_tensor = valid_obs.to(self.device)
          valid_act_tensor = valid_act.to(self.device)
          valid_pred = self.model(valid_obs_tensor)
          valid_loss = self.loss_fn(valid_pred, valid_act_tensor)
          print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {valid_loss.item():.4f}")

          if valid_loss.item() < best_loss:
            best_loss = valid_loss.item()
            torch.save(self.model.state_dict(), self.checkpoint_path)
            print(f"Saved model to {self.checkpoint_path}")

    def act(self, obs):
      # set model to evaluation mode
      self.model.eval()
      with torch.no_grad():
        resized_obs = cv2.resize(obs, (128, 128))
        current_obs = torch.from_numpy(resized_obs).permute(2, 0, 1).float() / 255.0
        current_obs = current_obs.unsqueeze(0).to(self.device)
        action = self.model(current_obs).squeeze(0)
        action = np.clip(action.cpu().numpy(), -5, 5)
      return action
    
