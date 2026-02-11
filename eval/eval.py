import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .run import Run
import random

class EvalResults:
  def __init__(self):
    self.run_results = []

  def summarize_steps_to_capture(self):
    stats = [result.get_steps_to_capture() for result in self.run_results]
    mean = np.mean(stats)
    std = np.std(stats)
    return mean, std
  
  def summarize_path_inefficiency(self):
    stats = [result.get_path_inefficiency() for result in self.run_results]
    mean = np.mean(stats)
    std = np.std(stats)
    return mean, std
  
  def summarize_direction_consistency(self):
    stats = [result.get_average_direction_consistency() for result in self.run_results]
    mean = np.mean(stats)
    std = np.std(stats)
    return mean, std
  
  def summarize_magnitude_continuity(self):
    stats = [result.get_average_magnitude_continuity() for result in self.run_results]
    mean = np.mean(stats)
    std = np.std(stats)
    return mean, std

  def summarize_completed_rate(self):
    stats = [result.get_completed() for result in self.run_results]
    mean = np.mean(stats)
    std = np.std(stats)
    return mean, std

  def summarize_trajectory_length(self):
    stats = [result.get_trajectory_length() for result in self.run_results]
    mean = np.mean(stats)
    std = np.std(stats)
    return mean, std

  def calculate_all_metrics(self):
    self.average_steps_to_capture, self.std_steps_to_capture = self.summarize_steps_to_capture()
    self.average_path_inefficiency, self.std_path_inefficiency = self.summarize_path_inefficiency()
    self.average_direction_consistency, self.std_direction_consistency = self.summarize_direction_consistency()
    self.average_magnitude_continuity, self.std_magnitude_continuity = self.summarize_magnitude_continuity()
    self.completed_rate, self.std_completed_rate = self.summarize_completed_rate()
    self.average_trajectory_length, self.std_trajectory_length = self.summarize_trajectory_length()

  def __str__(self) -> str:
    return f"Steps to Capture: {self.average_steps_to_capture} ± {self.std_steps_to_capture}\nPath Inefficiency: {self.average_path_inefficiency} ± {self.std_path_inefficiency}\nDirection Consistency: {self.average_direction_consistency} ± {self.std_direction_consistency}\nMagnitude Continuity: {self.average_magnitude_continuity} ± {self.std_magnitude_continuity}\nCompleted Rate: {self.completed_rate} ± {self.std_completed_rate}\nTrajectory Length: {self.average_trajectory_length} ± {self.std_trajectory_length}"

class Eval:
  def __init__(self, env, policy, num_runs=10, max_steps=100, seed=None):
    self.env = env
    self.policy = policy
    self.num_runs = num_runs
    self.max_steps = max_steps
    self.seed = seed if seed is not None else random.randint(0, 1000000)
    np.random.seed(self.seed)
    self.results = EvalResults()

  def _run_single(self, i):
    env = self.env.copy()
    run = Run(env, self.policy, self.max_steps, seed=self.seed + i)
    run.run()
    return run.get_results()

  def eval(self, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
      futures = [executor.submit(self._run_single, i) for i in range(self.num_runs)]
      self.results.run_results = [f.result() for f in futures]
    return self.results

  def get_results(self):
    return self.results