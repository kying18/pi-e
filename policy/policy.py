import os
import torch
import torch.nn as nn
import torch.optim as optim


class Policy:
    """Base class for learned policies."""

    def __init__(self):
        # Subclasses must set: self.model, self.device, self.checkpoint_path
        self.model = None
        self.device = None
        self.checkpoint_path = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def train(self, train_loader, val_loader, num_epochs=100):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for obs, actions in train_loader:
                obs, actions = obs.to(self.device), actions.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(obs)
                loss = self.loss_fn(outputs, actions)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for obs, actions in val_loader:
                    obs, actions = obs.to(self.device), actions.to(self.device)
                    outputs = self.model(obs)
                    val_loss += self.loss_fn(outputs, actions).item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"Saved checkpoint to {self.checkpoint_path}")

    def act(self, obs):
        raise NotImplementedError
