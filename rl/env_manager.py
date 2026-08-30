from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List

from rl.config import Config
from rl.actor_critic import ActorCritic
from rl.actor_critic_cnn import ActorCriticCNN
from rl.rollout_buffer import RolloutBuffer, _TensorRolloutBuffer
from rl.ppo import PPO


class EnvManager:
    """Manages CppVecEnv + weight sync to/from PyTorch model."""

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device

        # Ensure model weight initialization is reproducible for a given seed.
        torch.manual_seed(cfg.seed)

        import sys
        project_root = Path(__file__).resolve().parent.parent
        python_dir = project_root / "python"
        if str(python_dir) not in sys.path:
            sys.path.insert(0, str(python_dir))

        from cpp_vecenv import make_cpp_vec_env

        reward_dict = cfg.reward.to_dict()
        self.env = make_cpp_vec_env(
            n_envs=cfg.n_envs,
            map_size=cfg.map_size,
            curriculum_stage=cfg.curriculum_stage,
            unlock_ids=cfg.unlock_ids or None,
            disable_net_worth=cfg.reward.disable_net_worth,
            disable_daily_income=cfg.reward.disable_daily_income,
            reward_config=reward_dict,
            seed=cfg.seed,
            n_threads=cfg.cpp_threads,
        )

        self.n_envs = cfg.n_envs
        self.obs_size = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.obs_mode = getattr(cfg, "obs_mode", "flat")
        if self.obs_mode == "minimap":
            from minimap import MinimapVecEnvWrapper
            self.mm_env = MinimapVecEnvWrapper(self.env)
            C, H, W = self.mm_env.minimap_shape
            self.model = ActorCriticCNN(
                n_channels=C,
                grid_size=W,
                n_actions=self.n_actions,
                hidden_sizes=cfg.net_arch,
                device=device,
            )
        else:
            self.mm_env = None
            self.model = ActorCritic(
                obs_size=self.obs_size,
                n_actions=self.n_actions,
                hidden_sizes=cfg.net_arch,
                device=device,
            )

        if self.mm_env is not None:
            C, H, W = self.mm_env.minimap_shape
            self.buffer = _TensorRolloutBuffer(
                n_steps=cfg.n_steps,
                n_envs=cfg.n_envs,
                obs_shape=(C, H, W),
                n_actions=self.n_actions,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                device=device,
            )
        else:
            self.buffer = RolloutBuffer(
                n_steps=cfg.n_steps,
                n_envs=cfg.n_envs,
                obs_size=self.obs_size,
                n_actions=self.n_actions,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                device=device,
            )

        self.ppo = PPO(
            model=self.model,
            buffer=self.buffer,
            lr=cfg.learning_rate,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            use_amp=cfg.use_amp,
            amp_dtype=cfg.amp_dtype,
            torch_compile=cfg.torch_compile,
            device=device,
        )

        self._obs_gpu = None
        self._pinned_obs = None

    def _policy_obs(self, obs_np: np.ndarray) -> torch.Tensor:
        """Convert raw env observations to the tensor the policy consumes."""
        if self.mm_env is not None:
            mm = self.mm_env.minimap_obs()
            return torch.from_numpy(np.ascontiguousarray(mm, dtype=np.float32)).to(self.device, non_blocking=True)
        return torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(self.device, non_blocking=True)

    def reset(self) -> torch.Tensor:
        obs_np = self.env.reset()
        return self._policy_obs(obs_np)

    def step(self, actions: np.ndarray) -> Dict[str, Any]:
        """Step env with actions (CPU numpy int array [n_envs])."""
        actions = np.asarray(actions, dtype=np.int32)
        self.env.step_async(actions)
        obs_np, rewards_np, dones_np, infos = self.env.step_wait()

        obs = self._policy_obs(obs_np)
        rewards = torch.from_numpy(np.asarray(rewards_np, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.asarray(dones_np, dtype=bool)).to(self.device)
        # `terminated` is the true terminal flag (without truncation). The RL
        # layer uses it to bootstrap GAE so that time-limit truncation does not
        # cut the value bootstrap.
        terminated_np = np.asarray(
            getattr(self.env, "_last_terminateds", dones_np), dtype=bool
        )
        terminated = torch.from_numpy(terminated_np).to(self.device)

        return {
            "obs": obs,
            "rewards": rewards,
            "dones": dones,
            "terminated": terminated,
            "infos": infos,
        }

    def collect_step(self, obs: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
        """One full step: policy → env step → buffer add. Returns (new_obs, infos)."""
        policy_out = self.ppo.collect_step(obs)
        action_gpu = policy_out["action"]
        action_np = action_gpu.cpu().numpy().astype(np.int32)

        env_out = self.step(action_np)
        new_obs = env_out["obs"]
        rewards = env_out["rewards"]
        dones = env_out["dones"]
        terminated = env_out["terminated"]
        infos = env_out["infos"]

        self.buffer.add(
            obs=obs,
            action=action_gpu,
            reward=rewards,
            log_prob=policy_out["log_prob"],
            value=policy_out["value"],
            done=dones,
            terminated=terminated,
        )

        return new_obs, infos

    def finish_episode(self, last_obs: torch.Tensor, last_dones: torch.Tensor):
        """Compute last values and GAE."""
        with torch.no_grad():
            last_value = self.ppo.model.get_value(last_obs)
        self.ppo.update(last_value=last_value, last_done=last_dones)

    def get_stats(self) -> Dict[str, float]:
        stats = {}
        for i in range(self.n_envs):
            try:
                ep_info = self.env._env_infos[i] if hasattr(self.env, "_env_infos") else None
            except Exception:
                pass
        return stats

    def set_curriculum_stage(self, stage: int):
        """Switch curriculum stage on the C++ env (1-3, 0=all buildings)."""
        self.env.venv.set_curriculum_stage(stage)
        self.cfg.curriculum_stage = stage

    def close(self):
        self.env.close()
