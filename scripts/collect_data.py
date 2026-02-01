import numpy as np
import pygame
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.moving_object import MovingObjectEnv
from expert.expert_policy import ExpertPolicy
from policy.bc_policy import BcPolicy


def collect_expert_data(num_episodes=1000):
    """Collect demonstrations from expert policy."""
    pygame.init()
    env = MovingObjectEnv()
    expert = ExpertPolicy()

    observations = []
    actions = []
    episode_ends = []

    for episode in range(num_episodes):
        obs, _ = env.reset()

        for t in range(1000):
            action = expert.act(env)
            observations.append(obs)
            actions.append(action)

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        episode_ends.append(len(observations))

        if episode % 100 == 0:
            print(f"Collected {episode}/{num_episodes} episodes")

    env.close()
    return observations, actions, episode_ends


def collect_dagger_data(policy, num_episodes=100):
    """
    Collect DAgger data: run learned policy, label with expert actions.

    The learned policy controls the agent (determining state distribution),
    but we record what the expert would have done (correct labels).
    """
    pygame.init()
    env = MovingObjectEnv()
    expert = ExpertPolicy()

    observations = []
    actions = []
    episode_ends = []

    for episode in range(num_episodes):
        obs, _ = env.reset()

        for t in range(1000):
            # Get actions from both policies
            learned_action = policy.act(obs)
            expert_action = expert.act(env)

            # Record observation with EXPERT label
            observations.append(obs)
            actions.append(expert_action)

            # Step with LEARNED policy action (this is the key DAgger insight)
            obs, _, terminated, truncated, _ = env.step(learned_action)
            if terminated or truncated:
                break

        episode_ends.append(len(observations))

        if episode % 100 == 0:
            print(f"Collected {episode}/{num_episodes} episodes")

    env.close()
    return observations, actions, episode_ends


def save_data(observations, actions, filename, episode_ends=None):
    """Save observations and actions to .npz file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    observations = np.array(observations)
    actions = np.array(actions)
    if episode_ends is not None:
        episode_ends = np.array(episode_ends)
        np.savez(filename, observations=observations, actions=actions, episode_ends=episode_ends)
    else:
        np.savez(filename, observations=observations, actions=actions)
    print(f"Saved {len(observations)} samples to {filename}")


if __name__ == "__main__":
    # Collect expert data
    # observations, actions, episode_ends = collect_expert_data(num_episodes=1000)
    # save_data(observations, actions, "data/datasets/expert_data_with_episode_ends.npz", episode_ends)

    # Collect DAgger data with current learned policy
    learned_policy = BcPolicy(use_checkpoint=True, checkpoint_name="bc_policy")
    observations, actions, episode_ends = collect_dagger_data(learned_policy, num_episodes=300)
    save_data(observations, actions, "data/datasets/expert_data_bc_dagger_with_episode_ends.npz", episode_ends)
