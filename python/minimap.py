"""Minimap observation wrappers.

The colony env provides a flat 209-dim observation (counts, resources,
catalog). Spatial information (where land/resources are relative to the
colony) is lost. These wrappers add a 2D minimap tensor:
    [8 channels, 2R+1, 2R+1] — one-hot land types + occupied, window
    centered on the colony start (init_sel), radius R (default 14).

Usage:
    env = MinimapVecEnvWrapper(CppVecEnv(...))          # training
    env = MinimapSingleEnvWrapper(CppColonyEnv(...))    # eval / watch
    obs = env.minimap_obs()  # (n_envs, 8, 29, 29) float32
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym


class MinimapVecEnvWrapper:
    """Adds a minimap() method to a CppVecEnv-like vec env."""

    def __init__(self, venv):
        self.venv = venv
        self.radius = int(venv.venv.minimap_radius())
        self.channels = 8
        self.grid = 2 * self.radius + 1
        self.minimap_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.channels, self.grid, self.grid),
            dtype=np.float32,
        )

    @property
    def minimap_shape(self) -> tuple[int, int, int]:
        return (self.channels, self.grid, self.grid)

    def minimap_obs(self) -> np.ndarray:
        """Current minimaps for all envs: (n_envs, C, H, W) float32."""
        mm = self.venv.venv.minimap_batch()
        return np.ascontiguousarray(mm, dtype=np.float32)


class MinimapSingleEnvWrapper:
    """Adds a minimap() method to a CppColonyEnv-like single env."""

    def __init__(self, env):
        self.env = env
        self.radius = int(env.cpp_env.minimap_radius())
        self.channels = 8
        self.grid = 2 * self.radius + 1
        self.minimap_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.channels, self.grid, self.grid),
            dtype=np.float32,
        )

    @property
    def minimap_shape(self) -> tuple[int, int, int]:
        return (self.channels, self.grid, self.grid)

    def minimap_obs(self) -> np.ndarray:
        """Current minimap: (C, H, W) float32."""
        mm = self.env.cpp_env.minimap()
        return np.ascontiguousarray(mm, dtype=np.float32)
