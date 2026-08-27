from __future__ import annotations

import torch
import numpy as np
from typing import Optional


class RolloutBuffer:
    """GPU-resident rollout buffer with GAE.

    Stores obs, actions, rewards, log_probs, values, dones as GPU tensors.
    GAE computed on GPU.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_size: int,
        n_actions: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        total = n_steps * n_envs
        self.obs = torch.empty(total, obs_size, dtype=torch.float32, device=device)
        self.actions = torch.empty(total, dtype=torch.long, device=device)
        self.rewards = torch.empty(total, dtype=torch.float32, device=device)
        self.log_probs = torch.empty(total, dtype=torch.float32, device=device)
        self.values = torch.empty(total, dtype=torch.float32, device=device)
        self.dones = torch.empty(total, dtype=torch.bool, device=device)

        self.advantages = torch.empty(total, dtype=torch.float32, device=device)
        self.returns = torch.empty(total, dtype=torch.float32, device=device)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ):
        """Add one step of data. All tensors shape [n_envs]."""
        if self.pos >= self.n_steps:
            raise RuntimeError("Buffer full, call reset() first")
        start = self.pos * self.n_envs
        end = start + self.n_envs
        self.obs[start:end] = obs
        self.actions[start:end] = action
        self.rewards[start:end] = reward
        self.log_probs[start:end] = log_prob
        self.values[start:end] = value
        self.dones[start:end] = done
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True

    def reset(self):
        self.pos = 0
        self.full = False

    def compute_gae(self, last_value: torch.Tensor, last_done: torch.Tensor):
        """Compute GAE advantages and returns.

        last_value: [n_envs] value of the state after the last step
        last_done:  [n_envs] whether the last step terminated
        """
        n = self.n_steps * self.n_envs
        adv = torch.zeros_like(self.advantages)
        last_gae = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)

        next_adv = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
        for t in range(self.n_steps - 1, -1, -1):
            start = t * self.n_envs
            end = start + self.n_envs
            if t == self.n_steps - 1:
                next_values = last_value
                next_dones = last_done
            else:
                next_start = (t + 1) * self.n_envs
                next_values = self.values[next_start:next_start + self.n_envs]
                next_dones = self.dones[next_start:next_start + self.n_envs]

            delta = (
                self.rewards[start:end]
                + self.gamma * next_values * (~next_dones)
                - self.values[start:end]
            )
            next_adv = delta + self.gamma * self.gae_lambda * (~next_dones) * next_adv
            adv[start:end] = next_adv

        self.advantages[:n] = adv
        self.returns[:n] = adv + self.values[:n]

        mean_adv = self.advantages[:n].mean()
        std_adv = self.advantages[:n].std()
        if std_adv > 1e-8:
            self.advantages[:n] = (self.advantages[:n] - mean_adv) / (std_adv + 1e-8)

    def get_batches(self, batch_size: int):
        """Yield mini-batches. Each batch: dict of tensors."""
        n = self.n_steps * self.n_envs
        indices = torch.randperm(n, device=self.device)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            yield {
                "obs": self.obs[idx],
                "actions": self.actions[idx],
                "rewards": self.rewards[idx],
                "old_log_probs": self.log_probs[idx],
                "values": self.values[idx],
                "advantages": self.advantages[idx],
                "returns": self.returns[idx],
                "dones": self.dones[idx],
            }

    def clear_gpu_memory(self):
        torch.cuda.empty_cache()
