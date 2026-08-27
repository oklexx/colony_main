from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ActorCritic(nn.Module):
    """MLP actor-critic for discrete actions.

    Input:  [B, obs_size] float32
    Output: (logits [B, n_actions], values [B, 1])
    """

    def __init__(
        self,
        obs_size: int,
        n_actions: int,
        hidden_sizes: List[int],
        device: torch.device,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.device = device

        layers: List[nn.Module] = []
        prev = obs_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        self.trunk = nn.Sequential(*layers).to(device)
        self.actor_head = nn.Linear(prev, n_actions).to(device)
        self.critic_head = nn.Linear(prev, 1).to(device)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.actor_head(h)
        values = self.critic_head(h)
        return logits, values

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_probs = dist.log_prob(action)
        return action, log_probs, values.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, values = self.forward(obs)
        return values.squeeze(-1)

    @property
    def params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def state_dict_for_env(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def load_state_dict_from_env(self, state: dict):
        self.load_state_dict(state)
