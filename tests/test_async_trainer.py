import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.config import Config
from rl.actor_critic import ActorCritic
from rl.rollout_buffer import RolloutBuffer
from rl.ppo import PPO
from rl.async_trainer import AsyncTrainer


class FakeEnvManager:
    """Minimal env manager that simulates 2 envs with known episode returns."""

    def __init__(self, n_envs=2, obs_size=10, n_actions=5, n_steps=3):
        self.n_envs = n_envs
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.device = torch.device("cpu")
        self._step_count = 0
        self._episode_returns = {0: 100.0, 1: 200.0}

        torch.manual_seed(0)
        self.model = ActorCritic(obs_size, n_actions, [16], self.device)
        self.buffer = RolloutBuffer(
            n_steps=n_steps, n_envs=n_envs, obs_size=obs_size,
            n_actions=n_actions, gamma=0.99, gae_lambda=0.95,
            device=self.device,
        )
        self.ppo = PPO(
            model=self.model, buffer=self.buffer, lr=1e-3,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            n_epochs=1, batch_size=n_envs, use_amp=False,
            device=self.device,
        )

    def reset(self):
        return torch.randn(self.n_envs, self.obs_size)

    def collect_step(self, obs):
        self._step_count += 1
        with torch.no_grad():
            action, log_prob, value = self.model.get_action_and_value(obs)
        new_obs = torch.randn(self.n_envs, self.obs_size)
        rewards = torch.randn(self.n_envs)
        dones = torch.zeros(self.n_envs, dtype=torch.bool)
        self.buffer.add(
            obs=obs, action=action, reward=rewards,
            log_prob=log_prob, value=value, done=dones,
        )
        return new_obs, self.get_infos()

    def get_infos(self):
        if self._step_count >= 3:
            return [
                {"episode": {"r": self._episode_returns[0], "l": 50}},
                {"episode": {"r": self._episode_returns[1], "l": 80}},
            ]
        return [{}, {}]

    def close(self):
        pass


def test_per_env_episode_tracking(tmp_path):
    """Verify that episodes from different envs are tracked separately."""
    cfg = Config(
        n_envs=2,
        n_steps=3,
        total_timesteps=6,
        save_freq=0,
        eval_freq=0,
        use_amp=False,
        model_dir=str(tmp_path),
    )
    em = FakeEnvManager(n_envs=2, n_steps=3)
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)

    trainer.train(total_timesteps=6)

    # Both episodes should be tracked separately
    assert len(trainer._ep_returns) == 2, f"Expected 2 episodes, got {len(trainer._ep_returns)}"
    # best_reward should be the max of individual episodes, not their sum
    assert trainer.best_reward == 200.0, f"Expected best_reward=200.0, got {trainer.best_reward}"
    assert 100.0 in trainer._ep_returns
    assert 200.0 in trainer._ep_returns


if __name__ == "__main__":
    test_per_env_episode_tracking()
    print("PASS: per-env episode tracking")
