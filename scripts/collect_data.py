import numpy as np                                                                                                                                                                            
import pygame
import os
import sys
                                                                                                                                                                                              
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.moving_object import MovingObjectEnv
from expert.expert_policy import ExpertPolicy


def collect_data(num_episodes=1000, n_frames=3):
  pygame.init()
  env = MovingObjectEnv()
  observations = []
  actions = []
  policy = ExpertPolicy()
  for episode in range(num_episodes):
    observation, info = env.reset()
    frames = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(n_frames-1)]
    for t in range(1000):
      frames.append(observation)

      action = policy.act(env)
      observation, reward, terminated, truncated, info = env.step(action)
      
      # Stack frames along channel dimension: (H, W, C*n_frames)
      observations.append(np.concatenate(frames, axis=2))
      actions.append(action)
      frames.pop(0)  # Remove the oldest frame

      if terminated or truncated:
        break
    if episode % 100 == 0:
      print(f"Collected {episode} episodes")
  env.close()

  return observations, actions

def save_data(observations, actions, filename):
  observations = np.array(observations)
  actions = np.array(actions)
  np.savez(filename, observations=observations, actions=actions)
  print(f"Saved data to {filename}")

if __name__ == "__main__":
  # collect data
  observations, actions = collect_data(num_episodes=1000, n_frames=3)
  save_data(observations, actions, "data/expert_data_n_frames_3.npz")

  # load data
  data = np.load("data/expert_data_n_frames_3.npz")
  observations = data["observations"]
  actions = data["actions"]
  print(observations.shape, actions.shape)