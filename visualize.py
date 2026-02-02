import pygame
from env.moving_object import MovingObjectEnv
from policy.policy import Policy
from policy.bc_policy import BcPolicy
from expert.expert_policy import ExpertPolicy
from policy.multi_img_bc_policy import MultiImgBcPolicy
from policy.action_chunking_policy import ActionChunkingPolicy
from policy.act_policy import ActPolicy
import numpy as np

def run(policy=None):
    """
    Visualize the environment with a given policy.
    If policy is None, uses random actions.

    policy: Policy object
    """
    pygame.init()

    screen = pygame.display.set_mode((256, 256))
    pygame.display.set_caption("Moving Object Env")

    env = MovingObjectEnv()
    obs, _ = env.reset()

    running = True
    episode = 0
    steps = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy.act(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        pygame.time.delay(50)
        steps += 1

        if terminated:
            episode += 1
            print(f"Episode {episode} completed in {steps} steps")
            steps = 0
            obs, _ = env.reset()

    pygame.quit()


if __name__ == "__main__":
    # print("Running with random policy...")
    # run(policy=None)
    # print("Running with expert policy...")
    # run(policy=ExpertPolicy())
    # print("Running with BC policy...")
    # run(policy=BcPolicy(use_checkpoint=True, checkpoint_name="bc_policy"))
    # print("Running with action chunking policy...")
    # run(policy=ActionChunkingPolicy(use_checkpoint=True, checkpoint_name="action_chunking_policy"))
    print("Running with ACT policy...")
    run(policy=ActPolicy(use_checkpoint=True, checkpoint_name="act_policy"))