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

        radius = cfg.minimap_radius
        if radius != 14:
            self.env.venv.set_minimap_radius(radius)

        self.obs_mode = getattr(cfg, "obs_mode", "flat")
        if self.obs_mode in ("minimap", "hybrid"):
            from minimap import MinimapVecEnvWrapper
            self.mm_env = MinimapVecEnvWrapper(self.env)
            self.mm_env.refresh()
            C, H, W = self.mm_env.minimap_shape
            if self.obs_mode == "hybrid":
                from rl.actor_critic_hybrid import ActorCriticHybrid
                self.model = ActorCriticHybrid(
                    obs_size=self.obs_size,
                    n_channels=C,
                    grid_size=W,
                    n_actions=self.n_actions,
                    hidden_sizes=cfg.net_arch,
                    device=device,
                )
            else:
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

        if self.obs_mode == "hybrid":
            C, H, W = self.mm_env.minimap_shape
            self.buffer = _TensorRolloutBuffer(
                n_steps=cfg.n_steps,
                n_envs=cfg.n_envs,
                obs_shape=(C, H, W),
                n_actions=self.n_actions,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                device=device,
                flat_dim=self.obs_size,
            )
        elif self.mm_env is not None:
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
        self._last_flat = None

    def _policy_obs(self, obs_np: np.ndarray):
        """Convert raw env observations to the tensor(s) the policy consumes.

        Returns a (flat, minimap) tuple in hybrid mode, a single minimap
        tensor in minimap mode, or a single flat tensor in flat mode.
        """
        flat = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(self.device, non_blocking=True)
        if self.obs_mode == "hybrid":
            mm = self.mm_env.minimap_obs()
            mm_t = torch.from_numpy(np.ascontiguousarray(mm, dtype=np.float32)).to(self.device, non_blocking=True)
            return flat, mm_t
        if self.mm_env is not None:
            mm = self.mm_env.minimap_obs()
            return torch.from_numpy(np.ascontiguousarray(mm, dtype=np.float32)).to(self.device, non_blocking=True)
        return flat

    def reset(self):
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

        out = {
            "obs": obs,
            "rewards": rewards,
            "dones": dones,
            "terminated": terminated,
            "infos": infos,
        }
        if isinstance(obs, tuple):
            out["flat"] = obs[0]
            out["minimap"] = obs[1]
        return out

    def collect_step(self, obs) -> tuple:
        """One full step: policy → env step → buffer add. Returns (new_obs, infos)."""
        # Get action masks for current obs (computed after last reset/step)
        action_masks_np = getattr(self.env, "action_masks", None)
        if action_masks_np is not None:
            action_masks = torch.from_numpy(action_masks_np).to(self.device)
        else:
            action_masks = None

        if self.obs_mode == "hybrid":
            flat, minimap = obs
            self._last_flat = flat
            policy_out = self.ppo.collect_step(flat, minimap, action_masks=action_masks)
        else:
            policy_out = self.ppo.collect_step(obs, action_masks=action_masks)
        action_gpu = policy_out["action"]
        action_np = action_gpu.cpu().numpy().astype(np.int32)

        env_out = self.step(action_np)
        new_obs = env_out["obs"]
        rewards = env_out["rewards"]
        dones = env_out["dones"]
        terminated = env_out["terminated"]
        infos = env_out["infos"]

        if self.obs_mode == "hybrid":
            flat, minimap = obs
            self.buffer.add(
                obs=minimap,
                action=action_gpu,
                reward=rewards,
                log_prob=policy_out["log_prob"],
                value=policy_out["value"],
                done=dones,
                terminated=terminated,
                flat=flat,
                action_masks=action_masks,
            )
        else:
            self.buffer.add(
                obs=obs,
                action=action_gpu,
                reward=rewards,
                log_prob=policy_out["log_prob"],
                value=policy_out["value"],
                done=dones,
                terminated=terminated,
                action_masks=action_masks,
            )

        return new_obs, infos

    def finish_episode(self, last_obs, last_dones: torch.Tensor):
        """Compute last values and GAE."""
        with torch.no_grad():
            if self.obs_mode == "hybrid":
                last_flat, last_minimap = last_obs
                last_value = self.ppo.model.get_value(last_flat, last_minimap)
            else:
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

    def get_allowed_buildings_for_stage(self, stage_id: int) -> List[str]:
        """Get list of building names allowed in current curriculum stage.
        
        Args:
            stage_id: Curriculum stage (0=unlock all, 1-3=limited set)
            
        Returns:
            List of building names available in this stage
        """
        # Default curriculum schedule - each stage unlocks specific buildings
        curriculum_schedule = {
            0: ["INITIALIZE", "MOVE_RIGHT", "MOVE_LEFT", "MOVE_UP", "MOVE_DOWN",
                "BUILD_HOUSE", "BUILD_FURNITURE", "GATHER_WOOD", "GATHER_STONE", 
                "GATHER_IRON", "PRESERVE", "WAIT"],  # All buildings
            1: ["INITIALIZE", "MOVE_RIGHT", "MOVE_LEFT", "MOVE_UP", "MOVE_DOWN",
                "BUILD_HOUSE", "PRESERVE", "WAIT"],  # Basic structures only
            2: ["INITIALIZE", "MOVE_RIGHT", "MOVE_LEFT", "MOVE_UP", "MOVE_DOWN",
                "BUILD_HOUSE", "GATHER_WOOD", "GATHER_STONE", "PRESERVE", "WAIT"],  # + gathering
            3: ["INITIALIZE", "MOVE_RIGHT", "MOVE_LEFT", "MOVE_UP", "MOVE_DOWN",
                "BUILD_HOUSE", "BUILD_FURNITURE", "GATHER_WOOD", "GATHER_STONE", 
                "GATHER_IRON", "PRESERVE", "WAIT"]  # All again for fine-tuning
        }
        
        return curriculum_schedule.get(stage_id, ["INITIALIZE", "MOVE_RIGHT", "BUILD_HOUSE", "PRESERVE", "WAIT"])

    def get_curriculum_progress(self, current_step: int) -> Dict[str, Any]:
        """Calculate progress in current curriculum stage.
        
        Args:
            current_step: Current training step
            
        Returns:
            Dict with:
            - stage: Current stage (0-3)
            - progress_percent: 0.0-1.0 progress to next stage transition
            - available_actions: First 5 building names for display
            - next_stage_at_step: Step when next stage begins or None
            - upcoming_stages: List of upcoming stages with thresholds
        """
        schedule = getattr(self.cfg, "curriculum_schedule", [])
        if not schedule or len(schedule) < 2:
            # Default schedule: 3 stages at 100k, 500k, 1M steps
            default_schedule = [
                (100000, 1),
                (500000, 2),
                (1000000, 3)
            ]
            # Convert to step->stage format
            schedule = [(t, s+1) for t, s in enumerate(default_schedule)]
        
        current_stage = self.cfg.curriculum_stage
        
        # Find current stage boundaries
        prev_threshold = 0
        for threshold, stage in schedule:
            if threshold > current_step:
                break
            prev_threshold = threshold
        
        # Find next transition
        next_threshold = None
        next_stage = None
        for threshold, stage in schedule:
            if threshold > current_step and stage > current_stage:
                next_threshold = threshold
                next_stage = stage
                break
        
        # Calculate progress percent to next transition
        stage_length = next_threshold - prev_threshold if next_threshold else max(100000, 1000000)
        steps_in_stage = current_step - prev_threshold
        progress_percent = min(1.0, max(0.0, steps_in_stage / stage_length))
        
        # Get available actions (first 5)
        all_actions = self.get_allowed_buildings_for_stage(current_stage)
        available_actions = " | ".join(all_actions[:5])
        
        # Calculate upcoming stages
        upcoming_stages = []
        for threshold, stage in schedule:
            if stage > current_stage and (next_threshold is None or threshold > next_threshold):
                upcoming_stages.append({
                    "stage": stage,
                    "at_step": threshold
                })
        
        # Limit to 3 upcoming stages
        upcoming_stages = upcoming_stages[:3]
        
        return {
            "stage": current_stage,
            "progress_percent": float(progress_percent),
            "available_actions": available_actions,
            "next_stage_at_step": int(next_threshold) if next_threshold else None,
            "upcoming_stages": upcoming_stages,
            "current_schedule": schedule,
        }

    def set_curriculum_stage(self, stage: int):
        """Switch curriculum stage on the C++ env (1-3, 0=all buildings)."""
        self.env.venv.set_curriculum_stage(stage)
        self.cfg.curriculum_stage = stage

    def close(self):
        self.env.close()
