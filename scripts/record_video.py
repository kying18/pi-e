import pygame
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.moving_object import MovingObjectEnv


def record(policy, output_path, num_episodes=3, fps=20):
    """
    Record a video of the policy running.

    Args:
        policy: callable that takes (env) and returns action, or None for random
        output_path: where to save the video (e.g., "notes/videos/expert.mp4")
        num_episodes: how many episodes to record
        fps: frames per second
    """
    pygame.init()

    env = MovingObjectEnv()

    # Collect video frames for output
    video_frames = []
    episodes_done = 0

    obs, _ = env.reset()

    while episodes_done < num_episodes:
        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy.act(obs)
            # action = policy.act(env)

        obs, reward, terminated, truncated, info = env.step(action)

        # Convert observation to BGR for OpenCV
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        video_frames.append(frame)

        if terminated:
            episodes_done += 1
            print(f"Episode {episodes_done}/{num_episodes} done")
            obs, _ = env.reset()

    pygame.quit()

    # Write video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in video_frames:
        out.write(frame)

    out.release()
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    # Random policy
    record(None, "notes/videos/00_random_policy.mp4", num_episodes=10)

    # from expert.expert_policy import ExpertPolicy
    # expert = ExpertPolicy()
    # record(expert, "notes/videos/01_expert_policy.mp4", num_episodes=10)

    # from policy.bc_policy import BcPolicy
    # bc_policy = BcPolicy(use_checkpoint=True, checkpoint_name="bc_policy")
    # record(bc_policy, "notes/videos/03_bc_policy.mp4", num_episodes=10)
    # bc_policy = BcPolicy(use_checkpoint=True, checkpoint_name="bc_policy_dagger")
    # record(bc_policy, "notes/videos/06_bc_policy_dagger.mp4", num_episodes=10)
