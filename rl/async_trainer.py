from __future__ import annotations

import time
import threading
import queue
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field

from rl.config import Config
from rl.env_manager import EnvManager


@dataclass
class TrainMetrics:
    total_timesteps: int = 0
    fps: float = 0.0
    episode_return: float = 0.0
    episode_length: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    learning_rate: float = 0.0
    gpu_mem_mb: float = 0.0
    wall_time_s: float = 0.0
    n_episodes: int = 0
    best_reward: float = float("-inf")


class AsyncTrainer:
    """Async PPO trainer: producer (env steps) + consumer (GPU PPO update)."""

    def __init__(
        self,
        cfg: Config,
        env_manager: EnvManager,
        logger: Optional[Any] = None,
        progress_callback: Optional[Callable[[TrainMetrics], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ):
        self.cfg = cfg
        self.em = env_manager
        self.device = env_manager.device
        self.logger = logger
        self.progress_callback = progress_callback
        self.stop_check = stop_check

        self._stop = False
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=cfg.queue_size)

        self.metrics = TrainMetrics()
        self.best_reward = float("-inf")
        self._ep_returns: list[float] = []
        self._ep_lengths: list[int] = []

    def _request_stop(self):
        self._stop = True

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

    def _check_stop(self) -> bool:
        if self._stop:
            return True
        if self.stop_check and self.stop_check():
            self._stop = True
            return True
        return False

    def _collect_rollout(self, obs: torch.Tensor) -> Dict[str, Any]:
        """Collect n_steps of transitions."""
        n_envs = self.em.n_envs

        for _ in range(self.cfg.n_steps):
            if self._check_stop():
                break

            new_obs, infos = self.em.collect_step(obs)

            # True termination flag (excludes truncation) for the bootstrap mask.
            terminated = self.em.buffer.terminated[
                (self.em.buffer.pos - 1) * n_envs : self.em.buffer.pos * n_envs
            ].cpu().numpy()

            # Track per-episode returns from C++ info (per-env, no cross-env
            # summing artifacts).
            for info in infos:
                ep = info.get("episode")
                if ep is not None:
                    r = ep["r"]
                    l = ep["l"]
                    self._ep_returns.append(r)
                    self._ep_lengths.append(l)
                    if r > self.best_reward:
                        self.best_reward = r

            obs = new_obs

        with torch.no_grad():
            last_value = self.em.ppo.model.get_value(obs)
            # last_done must be the TRUE termination flag of the LAST collected
            # step (== buffer.terminated[T-1] == terminal(s_T)), NOT an OR over
            # the whole rollout and NOT including time-limit truncation. This is
            # what the GAE bootstrap mask needs.
            last_done = torch.tensor(terminated, dtype=torch.bool, device=self.device)

        return {
            "last_value": last_value.cpu().numpy(),
            "last_done": last_done,
            "final_obs": obs,
        }

    def _update_ppo(self, rollout: Dict[str, Any]) -> Dict[str, float]:
        """Run PPO update on GPU."""
        last_value = torch.tensor(rollout["last_value"], dtype=torch.float32, device=self.device)
        last_done = torch.tensor(rollout["last_done"], dtype=torch.bool, device=self.device)

        t0 = time.perf_counter()
        stats = self.em.ppo.update(last_value=last_value, last_done=last_done)
        update_time = time.perf_counter() - t0

        stats["update_time_s"] = update_time
        if torch.cuda.is_available() and self.device.type == "cuda":
            stats["gpu_mem_mb"] = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            stats["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(self.device) / (1024 * 1024)

        return stats

    def train(self, total_timesteps: Optional[int] = None) -> TrainMetrics:
        total = total_timesteps or self.cfg.total_timesteps
        n_envs = self.em.n_envs
        steps_per_rollout = self.cfg.n_steps * n_envs

        self._log(f"[Trainer] Starting: total={total:,} steps, n_envs={n_envs}, "
                  f"n_steps={self.cfg.n_steps}, batch={self.cfg.batch_size}, "
                  f"epochs={self.cfg.n_epochs}, device={self.device}")
        self._log(f"[Trainer] AMP={self.cfg.use_amp} ({self.cfg.amp_dtype}), "
                  f"compile={self.cfg.torch_compile}")

        obs = self.em.reset()
        t_start = time.perf_counter()
        total_done = 0
        rollout_idx = 0
        # Save every `save_every` rollouts (robust to any save_freq/steps_per_rollout
        # combination; the old `total_done % save_freq < steps_per_rollout` condition
        # almost never fired for the default hyperparameters).
        save_every = (
            max(1, int(round(self.cfg.save_freq / steps_per_rollout)))
            if self.cfg.save_freq > 0
            else 0
        )

        while total_done < total and not self._stop:
            t_rollout_start = time.perf_counter()
            rollout = self._collect_rollout(obs)
            rollout_time = time.perf_counter() - t_rollout_start

            if self._stop:
                break

            stats = self._update_ppo(rollout)

            total_done += steps_per_rollout
            rollout_idx += 1
            elapsed = time.perf_counter() - t_start
            fps = total_done / max(elapsed, 1e-10)

            self.metrics.total_timesteps = total_done
            self.metrics.fps = fps
            self.metrics.policy_loss = stats.get("policy_loss", 0.0)
            self.metrics.value_loss = stats.get("value_loss", 0.0)
            self.metrics.entropy = stats.get("entropy", 0.0)
            self.metrics.approx_kl = stats.get("approx_kl", 0.0)
            self.metrics.learning_rate = stats.get("learning_rate", 0.0)
            self.metrics.gpu_mem_mb = stats.get("gpu_mem_mb", 0.0)
            self.metrics.wall_time_s = elapsed
            self.metrics.n_episodes = len(self._ep_returns)
            self.metrics.best_reward = self.best_reward

            self._log(
                f"[Step {total_done:,}/{total:,}] "
                f"FPS={fps:,.0f} | "
                f"episodes={self.metrics.n_episodes} "
                f"best={self.best_reward:.2f} | "
                f"p_loss={stats.get('policy_loss', 0):.4f} "
                f"v_loss={stats.get('value_loss', 0):.4f} "
                f"ent={stats.get('entropy', 0):.4f} "
                f"KL={stats.get('approx_kl', 0):.5f} | "
                f"GPU_mem={stats.get('gpu_mem_mb', 0):.0f}MB"
            )

            if self.progress_callback:
                try:
                    self.progress_callback(self.metrics)
                except Exception:
                    pass

            save_dir = Path(self.cfg.model_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            if save_every > 0 and rollout_idx % save_every == 0:
                ckpt_path = save_dir / f"checkpoint_{total_done}_steps.pt"
                self.em.ppo.save(str(ckpt_path))
                self._log(f"[Save] {ckpt_path}")

        elapsed = time.perf_counter() - t_start
        final_fps = total_done / max(elapsed, 1e-10)

        save_dir = Path(self.cfg.model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / "final_model.pt"
        self.em.ppo.save(str(final_path))
        self._log(f"[Save] Final model: {final_path}")

        self.metrics.total_timesteps = total_done
        self.metrics.fps = final_fps
        self.metrics.wall_time_s = elapsed
        self.metrics.best_reward = self.best_reward

        self._log(f"[Done] {total_done:,} steps in {elapsed:.1f}s "
                  f"({final_fps:,.0f} FPS), best_reward={self.best_reward:.2f}, "
                  f"episodes={self.metrics.n_episodes}")

        return self.metrics

    def close(self):
        self.em.close()
