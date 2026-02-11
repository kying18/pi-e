import numpy as np
import random

class RunResults:
  def __init__(self):
    self.steps_to_capture = 0
    self.path_inefficiency = 0
    self.direction_consistency = []
    self.magnitude_continuity = []
    self.completed = False
    self.num_steps = 0
    self.trajectory_length = 0

  def get_steps_to_capture(self):
    return self.steps_to_capture
  
  def get_path_inefficiency(self):
    return self.path_inefficiency
  
  def get_average_direction_consistency(self):
    return np.mean(self.direction_consistency) if self.direction_consistency else 1.0

  def get_average_magnitude_continuity(self):
    return np.mean(self.magnitude_continuity) if self.magnitude_continuity else 0.0
  
  def get_completed(self):
    return self.completed
  
  def get_trajectory_length(self):
    return self.trajectory_length

class Run:
  def __init__(self, env, policy, max_steps=100, seed=None):
    self.env = env
    self.policy = policy
    self.max_steps = max_steps
    self.seed = seed if seed is not None else random.randint(0, 1000000)
    self.rng = np.random.default_rng(self.seed)
    self.previous_action = None
    self.start_ee_pos = None
    self.end_ee_pos = None
    self.start_ball_pos = None
    self.end_ball_pos = None
    self.results = RunResults()
  
  def _calculate_direction_consistency(self, action):
    if self.previous_action is None:
      return None
    return np.dot(action, self.previous_action) / (np.linalg.norm(action) * np.linalg.norm(self.previous_action) + 1e-8)

  def _calculate_magnitude_continuity(self, action):
    if self.previous_action is None:
      return None
    return np.linalg.norm(action - self.previous_action) / (np.linalg.norm(action) + np.linalg.norm(self.previous_action) + 1e-8)

  def _calculate_path_inefficiency(self):
    return self.results.trajectory_length / (np.linalg.norm(self.start_ee_pos - self.end_ee_pos) + 1e-8)

  def run(self):
    obs, _ = self.env.reset(rng=self.rng)
    self.previous_action = None
    self.start_ee_pos = self.env.ee_pos.copy()
    self.start_ball_pos = self.env.ball_pos.copy()

    for _ in range(self.max_steps):
      timestep_ee_pos = self.env.ee_pos.copy()

      if self.policy is None:
        action = self.env.action_space.sample()
      else:
        action = self.policy.act(obs, self.env)
      obs, _, terminated, _, _ = self.env.step(action)

      self.results.steps_to_capture += 1
      self.results.trajectory_length += np.linalg.norm(self.env.ee_pos - timestep_ee_pos)
      direction_consistency = self._calculate_direction_consistency(action)
      if direction_consistency is not None:
        self.results.direction_consistency.append(direction_consistency)
      magnitude_continuity = self._calculate_magnitude_continuity(action)
      if magnitude_continuity is not None:
        self.results.magnitude_continuity.append(magnitude_continuity)

      self.previous_action = action

      if terminated:
        self.results.completed = True
        break

    self.end_ee_pos = self.env.ee_pos.copy()
    self.end_ball_pos = self.env.ball_pos.copy()
    self.results.path_inefficiency = self._calculate_path_inefficiency()

  def get_results(self):
    return self.results