import gymnasium as gym
import numpy as np
import pygame


class MovingObjectEnv(gym.Env):
    """
    Robot arm must intercept/grasp a moving object
    - Ball rolls across screen
    - Robot sees RGB image
    - Must predict where ball will be and intercept
    """

    def __init__(self):
        self.width, self.height = 256, 256

        # Ball state: position + velocity
        self.ball_pos = np.array([128., 128.])
        self.ball_vel = np.array([2., 1.])  # moves each step
        self.ball_radius = 10

        # Robot end-effector position
        self.ee_pos = np.array([50., 128.])

        # Observation: RGB image
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(256, 256, 3),
            dtype=np.uint8
        )

        # Action: delta x, delta y for end-effector
        self.action_space = gym.spaces.Box(
            low=-5, high=5,
            shape=(2,),
            dtype=np.float32
        )

    def copy(self):
        return MovingObjectEnv()

    def reset(self, seed=None, rng=None):
        super().reset(seed=seed)
        if rng is None:
            rng = np.random.default_rng(seed)
        # Random ball starting position and velocity
        self.ball_pos = rng.random(2) * 200 + 28
        angle = rng.random() * 2 * np.pi
        speed = rng.random() * 3 + 1
        self.ball_vel = np.array([np.cos(angle), np.sin(angle)]) * speed

        # Random end-effector starting position
        self.ee_pos = rng.random(2) * 200 + 28

        return self._get_obs(), {}

    def step(self, action):
        # Move ball
        self.ball_pos += self.ball_vel

        # Bounce off walls
        if self.ball_pos[0] < 0 or self.ball_pos[0] > self.width:
            self.ball_vel[0] *= -1
        if self.ball_pos[1] < 0 or self.ball_pos[1] > self.height:
            self.ball_vel[1] *= -1

        # Move end-effector
        self.ee_pos += action
        self.ee_pos = np.clip(self.ee_pos, 0, [self.width, self.height])

        # Reward: negative distance to ball
        dist = np.linalg.norm(self.ee_pos - self.ball_pos)
        reward = -dist

        # Success: within 15 pixels of ball
        success = dist < 15

        terminated = success
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {'success': success}

    def _get_obs(self):
        # Render scene as RGB image
        surface = pygame.Surface((self.width, self.height))
        surface.fill((255, 255, 255))  # white background

        # Draw moving ball (red)
        pygame.draw.circle(surface, (255, 0, 0),
                           self.ball_pos.astype(int),
                           self.ball_radius)

        # Draw end-effector (blue)
        pygame.draw.circle(surface, (0, 0, 255),
                           self.ee_pos.astype(int),
                           8)

        # Convert to numpy array
        img = pygame.surfarray.array3d(surface)
        img = np.transpose(img, (1, 0, 2))  # fix orientation

        return img
