from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Optional

import torch
import torch.nn as nn


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain=gain)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data, gain=gain)
        if module.bias is not None:
            module.bias.data.zero_()


class ActorCriticHybrid(nn.Module):
    """Hybrid actor-critic: flat MLP branch + CNN branch, merged trunk.

    The flat branch handles global stats (money, taxes, credits, day count).
    The CNN branch handles spatial layout (land types, buildings, resources).
    Both branches feed into a shared trunk MLP, then actor/critic heads.

    Input:  flat_obs [B, obs_size] float32 + minimap [B, C, H, W] float32
    Output: (logits [B, n_actions], values [B, 1])
    """

    def __init__(
        self,
        obs_size: int,
        n_channels: int,
        grid_size: int,
        n_actions: int,
        hidden_sizes: Sequence[int] | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.obs_size = obs_size
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.n_actions = n_actions
        hidden = list(hidden_sizes) if hidden_sizes else [256, 256]
        self.hidden_sizes = hidden

        # Flat branch: obs_size -> hidden[0]
        self.flat_proj = nn.Linear(obs_size, hidden[0])

        # CNN branch: n_channels x grid x grid -> hidden[0]
        conv1 = nn.Conv2d(n_channels, 32, 4, 2, 1)   # 32 x ((grid+2)//2-1)
        conv2 = nn.Conv2d(32, 64, 2, 2)              # 64 x ((g1+2)//2-1)
        self.cnn = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU())
        g1 = (grid_size + 2) // 2 - 1
        g2 = (g1 + 2) // 2 - 1
        self.cnn_proj = nn.Linear(64 * g2 * g2, hidden[0])

        # Shared trunk
        trunk: List[nn.Module] = []
        prev = hidden[0]
        for h in hidden[1:]:
            trunk.append(nn.Linear(prev, h))
            trunk.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*trunk)

        self.actor = nn.Linear(prev, n_actions)
        self.critic = nn.Linear(prev, 1)

        # Init
        _orthogonal_init(self.flat_proj, gain=1.0)
        for m in self.cnn:
            _orthogonal_init(m, gain=math.sqrt(2))
        _orthogonal_init(self.cnn_proj, gain=1.0)
        for m in self.trunk:
            _orthogonal_init(m, gain=1.0)
        _orthogonal_init(self.actor, gain=1.0)
        _orthogonal_init(self.critic, gain=1.0)

        self.to(device)
        self.device = torch.device(device)

    def forward(
        self, flat_obs: torch.Tensor, minimap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_feat = self.flat_proj(flat_obs)
        cnn_feat = self.cnn_proj(self.cnn(minimap).flatten(1))
        x = torch.relu(flat_feat + cnn_feat)
        x = self.trunk(x)
        logits = self.actor(x)
        values = self.critic(x)
        return logits, values

    def act(
        self,
        flat_obs: torch.Tensor,
        minimap: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits, values = self.forward(flat_obs, minimap)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if not deterministic else logits.argmax(-1)
        log_prob = dist.log_prob(action)
        return action, log_prob, values.squeeze(-1)

    def get_action_and_value(
        self,
        flat_obs: torch.Tensor,
        minimap: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(flat_obs, minimap)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_probs = dist.log_prob(action)
        return action, log_probs, values.squeeze(-1)

    def get_value(
        self, flat_obs: torch.Tensor, minimap: torch.Tensor
    ) -> torch.Tensor:
        _, values = self.forward(flat_obs, minimap)
        return values.squeeze(-1)

    @property
    def params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def state_dict_for_env(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def load_state_dict_from_env(self, state: dict):
        self.load_state_dict(state)
