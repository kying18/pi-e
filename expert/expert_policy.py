import numpy as np

class ExpertPolicy:
    def __init__(self):
        super().__init__()

    def act(self, env):
        """
        Expert policy that uses true state to compute optimal action.
        """

        ball_pos = env.ball_pos
        ball_vel = env.ball_vel
        ee_pos = env.ee_pos

        # End effector should move towards where the ball will be
        future_ball_pos = ball_pos + ball_vel
        action = future_ball_pos - ee_pos
        action = np.clip(action, -5, 5)

        return action