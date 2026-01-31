import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.bc_policy import BcPolicy
from policy.multi_img_bc_policy import MultiImgBcPolicy


def train_bc_policy():
  data = np.load("data/expert_data.npz")
  observations = data["observations"]
  actions = data["actions"]
  print(observations.shape, actions.shape)
  policy = BcPolicy(use_best_model=False)
  policy.train(observations, actions, num_epochs=100, batch_size=64, train_split=0.7, max_samples=5000)

def train_multi_img_bc_policy():
  data = np.load("data/expert_data_n_frames_3.npz")
  observations = data["observations"]
  actions = data["actions"]
  print(observations.shape, actions.shape)
  policy = MultiImgBcPolicy(use_best_model=False)
  policy.train(observations, actions, num_epochs=100, batch_size=64, train_split=0.7, max_samples=5000)

if __name__ == "__main__":
  # train_bc_policy()
  train_multi_img_bc_policy()