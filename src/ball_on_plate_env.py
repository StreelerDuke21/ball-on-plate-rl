"""
Ball-on-Plate Environment
Inspired by Kóczi, Németh & Sárosi (2023)
Electronics 2023, 12, 671
https://www.mdpi.com/2079-9292/12/3/671

Simplified physics: independent X/Y axes, frictionless inclined plane approximation.
Sampling rate: 20 Hz (dt = 0.05 s) as in the original paper.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BallOnPlateEnv(gym.Env):
    """
    Entorno de simulación Ball-on-Plate controlado por 2 servomotores.
    Objetivo principal: mantener la bolita en el centro de la placa.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        max_episode_steps: int = 200,
        plate_half_size: float = 0.10,
        max_angle_deg: float = 12.0,
        dt: float = 0.05,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.g = 9.81
        self.dt = dt
        self.plate_half_size = plate_half_size
        self.max_angle = np.deg2rad(max_angle_deg)
        self.max_servo_speed = np.deg2rad(80.0)

        self.observation_space = spaces.Box(
            low=np.array([-1.5, -1.5, -3.0, -3.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.5, 1.5, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x = self.np_random.uniform(-0.035, 0.035)
        self.y = self.np_random.uniform(-0.035, 0.035)
        self.vx = 0.0
        self.vy = 0.0

        self.alpha = 0.0
        self.beta = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.steps = 0

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        obs = np.array(
            [
                self.x / self.plate_half_size,
                self.y / self.plate_half_size,
                self.vx / 1.5,
                self.vy / 1.5,
                self.alpha / self.max_angle,
                self.beta / self.max_angle,
                self.last_action[0],
                self.last_action[1],
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -3.0, 3.0)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action.copy()

        scale = 0.35
        d_alpha = action[0] * self.max_angle * scale
        d_beta = action[1] * self.max_angle * scale

        max_d = self.max_servo_speed * self.dt
        d_alpha = np.clip(d_alpha, -max_d, max_d)
        d_beta = np.clip(d_beta, -max_d, max_d)

        self.alpha = np.clip(self.alpha + d_alpha, -self.max_angle, self.max_angle)
        self.beta = np.clip(self.beta + d_beta, -self.max_angle, self.max_angle)

        ax = -self.g * np.sin(self.alpha)
        ay = -self.g * np.sin(self.beta)

        self.vx += ax * self.dt
        self.vy += ay * self.dt
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        self.steps += 1

        dist = np.sqrt(self.x**2 + self.y**2)

        reward = -10.0 * dist
        reward -= 0.08 * (abs(self.alpha) + abs(self.beta))
        reward -= 0.02 * np.sum(np.square(action))

        if dist < 0.012:
            reward += 1.5

        terminated = False
        if abs(self.x) > self.plate_half_size or abs(self.y) > self.plate_half_size:
            reward = -25.0
            terminated = True

        truncated = self.steps >= self.max_episode_steps

        info = {
            "dist": float(dist),
            "x": float(self.x),
            "y": float(self.y),
            "alpha_deg": float(np.rad2deg(self.alpha)),
            "beta_deg": float(np.rad2deg(self.beta)),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        pass


def make_env(**kwargs):
    return BallOnPlateEnv(**kwargs)
