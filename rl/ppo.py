from __future__ import annotations

import torch
import torch.nn as nn
from torch import distributions as D
from typing import Optional, Dict, Any

from rl.actor_critic import ActorCritic
from rl.rollout_buffer import RolloutBuffer


class PPO:
    """PPO with GPU-resident buffer, AMP, optional torch.compile."""

    def __init__(
        self,
        model: ActorCritic,
        buffer: RolloutBuffer,
        lr: float = 3e-4,
        gamma: float = 0.995,
        gae_lambda: float = 0.98,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 8192,
        use_amp: bool = True,
        amp_dtype: str = "bfloat16",
        torch_compile: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.buffer = buffer
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.device = device or model.device

        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == "float16"))

        self.optimizer = torch.optim.Adam(
            model.params, lr=lr, eps=1e-5, weight_decay=0.0
        )

        if torch_compile:
            self.model = torch.compile(model, mode="reduce-overhead")
            self._compiled = True
        else:
            self._compiled = False

    def collect_step(
        self, obs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get action, log_prob, value for current obs (no grad)."""
        self.model.eval()
        with torch.no_grad():
            action, log_prob, value = self.model.get_action_and_value(obs)
        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
        }

    def update(
        self, last_value: torch.Tensor, last_done: torch.Tensor
    ) -> Dict[str, float]:
        """Run PPO update after buffer is full.

        last_value: [n_envs] value of terminal state
        last_done:  [n_envs] whether terminal
        """
        self.buffer.compute_gae(last_value, last_done)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_batches = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]

                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        policy_loss, value_loss, entropy, approx_kl = self._compute_loss_components(
                            obs, actions, old_log_probs, advantages, returns, old_values
                        )
                        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                else:
                    policy_loss, value_loss, entropy, approx_kl = self._compute_loss_components(
                        obs, actions, old_log_probs, advantages, returns, old_values
                    )
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.params, self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.params, self.max_grad_norm)
                    self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl.item()
                n_batches += 1

        self.model.train()
        self.buffer.reset()

        return {
            "policy_loss": total_policy_loss / max(n_batches, 1),
            "value_loss": total_value_loss / max(n_batches, 1),
            "entropy": total_entropy / max(n_batches, 1),
            "approx_kl": total_kl / max(n_batches, 1),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def _compute_loss_components(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ):
        logits, values = self.model(obs)
        dist = D.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        values = values.squeeze(-1)
        value_loss = 0.5 * ((values - returns) ** 2).mean()

        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().abs()

        return policy_loss, value_loss, entropy, approx_kl

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "buffer_pos": self.buffer.pos,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
